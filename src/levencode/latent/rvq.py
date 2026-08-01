"""RVQ (residual vector quantizer): the discrete anchor of the hybrid latent.

The teacher latent z is quantized into R book indices (the *plan codes* — a
small AR prior over these restores exact likelihood and gives cheap
temperature/beam search), and the quantization residual r = z - z_q carries the
*fidelity detail* that the energy head predicts. Discrete carries the mode
choice; continuous carries the detail.

EMA codebook updates (VQGAN/SoundStream-style) with straight-through gradients
and a commitment loss. In eval mode quantization is exact and does not touch
the EMA statistics."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RVQ(nn.Module):
    def __init__(
        self,
        num_books: int,
        codebook_size: int,
        dim: int,
        ema_decay: float = 0.99,
        commitment: float = 0.25,
    ):
        super().__init__()
        self.num_books = num_books
        self.codebook_size = codebook_size
        self.dim = dim
        self.ema_decay = ema_decay
        self.commitment = commitment
        scale = dim**-0.5
        self.register_buffer("codebooks", torch.randn(num_books, codebook_size, dim) * scale)
        self.register_buffer("ema_count", torch.zeros(num_books, codebook_size))
        self.register_buffer("ema_sum", self.codebooks.clone())
        self.register_buffer("initialized", torch.zeros(num_books, dtype=torch.bool))

    def _l2(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a2 = (a * a).sum(-1, keepdim=True)
        b2 = (b * b).sum(-1)
        return a2 + b2 - 2.0 * a @ b.t()

    def quantize(self, z: torch.Tensor, update: bool = False) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """z [..., D] -> (z_q [..., D], codes [..., R], stats)."""
        flat = z.reshape(-1, self.dim)
        B = flat.shape[0]
        residual = flat
        z_q = torch.zeros_like(flat)
        codes = torch.zeros(flat.shape[0], self.num_books, device=z.device, dtype=torch.long)
        losses: list[torch.Tensor] = []
        for b in range(self.num_books):
            d2 = self._l2(residual, self.codebooks[b])  # [B, C]
            idx = d2.argmin(-1)
            codes[:, b] = idx
            cb = self.codebooks[b][idx]
            losses.append(F.mse_loss(cb.detach(), residual))
            if update and self.training:
                self._ema_update(b, residual.detach(), idx)
            z_q = z_q + cb
            residual = residual - cb
        # straight-through: gradients flow to z as if z_q were z
        z_q_out = flat + (z_q - flat).detach()
        if len(losses):
            commit = sum(losses) / len(losses)
        else:
            commit = torch.zeros((), device=z.device)
        stats = {"commit": commit.detach(), "dist": (z_q.detach() - flat.detach()).norm(dim=-1).mean().detach()}
        return z_q_out.reshape(z.shape), codes.reshape(*z.shape[:-1], self.num_books), stats

    @torch.no_grad()
    def _ema_update(self, b: int, residual: torch.Tensor, idx: torch.Tensor) -> None:
        onehot = F.one_hot(idx, self.codebook_size).to(residual.dtype)
        count = onehot.sum(0)
        summed = onehot.t() @ residual
        decay = self.ema_decay
        if self.initialized[b]:
            self.ema_count[b].mul_(decay).add_(count, alpha=1 - decay)
            self.ema_sum[b].mul_(decay).add_(summed, alpha=1 - decay)
        else:
            # first batch: overwrite only rows that won an assignment; zeroing
            # the rest (old behavior) parked every unused code at the origin
            assigned = count > 0
            self.ema_count[b] = count
            self.ema_sum[b] = torch.where(assigned.unsqueeze(-1), summed, self.codebooks[b])
            self.initialized[b] = True
        cnt = self.ema_count[b].clamp_min(1.0)
        self.codebooks[b] = self.ema_sum[b] / cnt.unsqueeze(-1)
        # dead-code revival: re-seed only codes whose EMA usage decayed to ~0
        # from random batch vectors. (The old dedup jolted the ENTIRE codebook
        # with unit-norm noise whenever any two rows collided — destroying the
        # live codes — and paid an O(C^2 d) torch.unique per update for it.)
        dead = self.ema_count[b] < 1e-2
        if dead.any():
            k = int(dead.sum())
            src = residual[torch.randint(0, residual.shape[0], (k,), device=residual.device)]
            self.codebooks[b][dead] = src
            self.ema_sum[b][dead] = src
            self.ema_count[b][dead] = 1.0

    @torch.no_grad()
    def quantize_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Embedding lookup for already-selected codes (inference path):
        codes [..., R] -> z_q [..., D]."""
        out = None
        for b in range(self.num_books):
            e = self.codebooks[b][codes[..., b]]
            out = e if out is None else out + e
        return out

    @torch.no_grad()
    def warm(self, z: torch.Tensor, steps: int = 20, batch: int = 2048) -> dict:
        """One-shot EMA warm-up pass over a sample of latents (precompute time)."""
        flat = z.reshape(-1, self.dim)
        for _ in range(steps):
            perm = torch.randperm(flat.shape[0], device=flat.device)[:batch]
            self.quantize(flat[perm], update=True)
        return {"warmed": int(self.initialized.sum().item())}
