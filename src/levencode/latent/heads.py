"""Latent JEPA heads: AR code priors (discrete anchor), residual heads
(continuous detail, energy-trained), and energy scorers (CFG guidance).

Each granularity level is a JEPA: a causal prior over the RVQ *plan codes* of
chunk latents (predictor), whose last hidden states condition an energy-based
residual head that predicts the continuous detail r = z - z_q. A level's score
function is an energy scorer over candidate latents, trained with condition
dropout so sampling can use the classifier-free guided combination

    score(z) = (1 + lambda) * E(cond, z) - lambda * E(null, z)

Sequence layout of a level's AR prior (causal transformer):
    [ctx, b1_0, b2_0, b1_1, b2_1, ...]
Outputs at even positions predict book-1 of the same-index chunk; odd positions
predict book-2 (RVQ-AR chaining within a chunk, teacher-forced)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rvq import RVQ


def causal_mask(sz: int, device) -> torch.Tensor:
    return torch.triu(torch.full((sz, sz), float("-inf"), device=device), diagonal=1)


class SwiGLUBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, 2 * dim)
        self.w2 = nn.Linear(dim, dim)
        self.c = nn.Linear(dim, 2 * dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        g = self.w1(x) + self.c(cond)
        g1, g2 = g.chunk(2, dim=-1)
        return x + self.w2(g1 * F.silu(g2))


class ResidualHead(nn.Module):
    """CALM-style single-step continuous generator: noise -> residual r."""

    def __init__(self, cond_dim: int, latent_dim: int, blocks: int = 3, hidden: int = 512):
        super().__init__()
        self.cond_proj = nn.Linear(cond_dim, hidden)
        self.noise_proj = nn.Linear(latent_dim, hidden)
        self.blocks = nn.ModuleList([SwiGLUBlock(hidden) for _ in range(blocks)])
        self.out = nn.Linear(hidden, latent_dim)
        self.null = nn.Parameter(torch.zeros(cond_dim))

    def forward(
        self, cond: torch.Tensor, noise: torch.Tensor, dropped: torch.Tensor | None = None
    ) -> torch.Tensor:
        if dropped is None:
            c = cond + self.null
        else:
            c = torch.where(dropped[:, None].bool(), self.null.expand_as(cond), cond)
        c = self.cond_proj(c)  # cond_dim -> hidden so the blocks can use it
        x = self.noise_proj(noise)
        for blk in self.blocks:
            x = blk(x, c)
        return self.out(x)

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, n: int) -> torch.Tensor:
        """n candidate residuals from U[-0.5, 0.5] noise (CALM Sec. 3.3.3)."""
        B = cond.shape[0]
        d = self.noise_proj.in_features
        eps = torch.rand(B, n, d, device=cond.device) - 0.5
        cond = cond.unsqueeze(1).expand(B, n, -1).reshape(B * n, -1)
        out = self(cond, eps.reshape(B * n, d))
        return out.reshape(B, n, d)


class EnergyScorer(nn.Module):
    """E(cond, z) -> scalar; lower = more plausible. Condition dropout makes
    E(null, z) available for classifier-free guided candidate selection."""

    def __init__(self, cond_dim: int, latent_dim: int, hidden: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim + latent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.null = nn.Parameter(torch.zeros(cond_dim))

    def forward(
        self, cond: torch.Tensor, z: torch.Tensor, dropped: torch.Tensor | None = None
    ) -> torch.Tensor:
        if dropped is None:
            c = cond + self.null
        else:
            c = torch.where(dropped[:, None].bool(), self.null.expand_as(cond), cond)
        return self.mlp(torch.cat([c, z], dim=-1)).squeeze(-1)


class LevelHeads(nn.Module):
    """One granularity level: AR code prior + residual head + energy scorer."""

    def __init__(
        self,
        name: str,
        ar_dim: int,
        codebook_size: int,
        latent_dim: int,
        ar_layers: int = 3,
        ar_heads: int = 8,
        residual_blocks: int = 3,
        energy_hidden: int = 256,
    ):
        super().__init__()
        self.name = name
        self.code_embed = nn.Embedding(codebook_size, ar_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=ar_dim,
            nhead=ar_heads,
            dim_feedforward=4 * ar_dim,
            batch_first=True,
            norm_first=True,
            activation="gelu",
            dropout=0.0,
        )
        self.prior = nn.TransformerEncoder(layer, ar_layers)
        self.b1_head = nn.Linear(ar_dim, codebook_size)
        self.b2_head = nn.Linear(ar_dim, codebook_size)
        self.residual = ResidualHead(ar_dim, latent_dim, blocks=residual_blocks)
        self.energy = EnergyScorer(ar_dim, latent_dim, hidden=energy_hidden)
        self.bias_b1 = nn.Parameter(torch.zeros(1, codebook_size))
        self.bias_b2 = nn.Parameter(torch.zeros(1, codebook_size))

    def forward(
        self,
        head: torch.Tensor,  # [B, D] sequence start (ctx embed, + coarse embed for fine)
        codes: torch.Tensor,  # [B, T, 2] teacher-forced RVQ indices
        mask: torch.Tensor,   # [B, T] float 1=real chunk
    ) -> dict:
        B, T, _ = codes.shape
        emb = self.code_embed(codes)  # [B, T, 2, D]
        seq = torch.cat([head.unsqueeze(1), emb.reshape(B, 2 * T, -1)], dim=1)
        L = 1 + 2 * T
        att = causal_mask(L, head.device)
        pad = torch.zeros(B, L, device=head.device)
        # positions beyond the real chunks are masked; head position is always real
        chunk_flags = mask.unsqueeze(2).expand(B, T, 2).reshape(B, 2 * T)
        pad[:, 1:] = (chunk_flags == 0).float()
        out = self.prior(seq, mask=att, src_key_padding_mask=pad.bool())
        # even positions 0,2,.. predict book-1 of chunk j; odd positions predict book-2
        b1_logits = self.b1_head(out[:, 0::2]) + self.bias_b1  # [B, T, C]
        b2_logits = self.b2_head(out[:, 1::2]) + self.bias_b2  # [B, T, C]
        conds = out[:, 1::2]  # hidden after each chunk's b2 -> residual/energy cond
        return {"b1_logits": b1_logits, "b2_logits": b2_logits, "conds": conds, "hidden": out}


class LatentHeads(nn.Module):
    """Container for the full multi-granularity JEPA stack."""

    def __init__(
        self,
        student_hidden: int,
        latent_dim: int,
        codebook_size: int,
        ar_dim: int = 512,
        ar_layers: int = 3,
        ar_heads: int = 8,
        residual_blocks: int = 3,
        energy_hidden: int = 256,
    ):
        super().__init__()
        self.ctx_pool_proj = nn.Linear(student_hidden, ar_dim)
        self.coarse = LevelHeads(
            "coarse", ar_dim, codebook_size, latent_dim, ar_layers, ar_heads, residual_blocks, energy_hidden
        )
        self.fine = LevelHeads(
            "fine", ar_dim, codebook_size, latent_dim, ar_layers, ar_heads, residual_blocks, energy_hidden
        )
        self.coarse_cond_proj = nn.Linear(latent_dim, ar_dim)

    def ctx_embed(self, pooled_hidden: torch.Tensor) -> torch.Tensor:
        return self.ctx_pool_proj(pooled_hidden)
