"""Concept level: an LCM-style planner living in Loom's OWN hidden space.

Lessons from the Levencode stage-5 postmortem baked in here:
- Concepts are pooled from the model's own final hidden states, so the latent
  space is native — no frozen external teacher, no separate decodability
  adapter, no space mismatch.
- Injection is state conditioning (zero-init FiLM on the loop state), never
  logit mixing: an untrained/bad plan is exactly a no-op, not an override.
- CAUSALITY CONTRACT: the modulation vector for segment j must be a function
  of segments < j only. Callers pass either ConceptPredictor outputs (which
  are causally shifted by construction) or `shift_concepts(pooled)`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LoomConfig
from .layers import DenseBlock, RMSNorm, build_rope_cache


def rms_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Parameter-free RMS normalization (scale-stable concept targets)."""
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps).to(x.dtype)


def pool_segments(hidden: torch.Tensor, segment_len: int) -> torch.Tensor:
    """Mean-pool [B, T, D] into per-segment concepts [B, S, D] (fixed-length
    segments; the tail segment pools over however many tokens remain)."""
    B, T, D = hidden.shape
    S = (T + segment_len - 1) // segment_len
    pad = S * segment_len - T
    if pad:
        hidden = F.pad(hidden, (0, 0, 0, pad))
    seg = hidden.view(B, S, segment_len, D)
    if pad:
        denom = torch.full((S,), float(segment_len), device=hidden.device)
        denom[-1] = float(segment_len - pad)
        pooled = seg.sum(2) / denom[None, :, None]
    else:
        pooled = seg.mean(2)
    return rms_normalize(pooled)


def shift_concepts(pooled: torch.Tensor) -> torch.Tensor:
    """Predictor-free causal conditioning: segment j sees pooled c_{j-1}
    (zeros for segment 0). The cheap ablation arm against the predictor."""
    return F.pad(pooled, (0, 0, 1, 0))[:, :-1]


class ConceptPredictor(nn.Module):
    """Small causal transformer over the concept sequence. Output at position
    j is the prediction for concept j given concepts < j (BOS-shifted), so
    predictions are causally valid conditioning by construction."""

    def __init__(self, cfg: LoomConfig):
        super().__init__()
        d = cfg.d_model
        self.bos = nn.Parameter(torch.zeros(1, 1, d))
        self.in_proj = nn.Linear(d, d, bias=False)
        self.blocks = nn.ModuleList(
            DenseBlock(d, cfg.concept_heads, cfg.concept_heads, cfg.concept_ff, cfg.norm_eps)
            for _ in range(cfg.concept_layers)
        )
        self.norm = RMSNorm(d, cfg.norm_eps)
        self.out_proj = nn.Linear(d, d, bias=False)
        hd = d // cfg.concept_heads
        cos, sin = build_rope_cache(cfg.max_segments + 1, hd, cfg.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, concepts: torch.Tensor) -> torch.Tensor:
        """concepts [B, S, D] -> predictions [B, S+1, D]; preds[:, j] is the
        model's guess for concept j (preds[:, S] plans the next, unseen one)."""
        B, S, D = concepts.shape
        x = torch.cat([self.bos.expand(B, 1, D), self.in_proj(concepts)], dim=1)
        cos, sin = self.cos[: S + 1], self.sin[: S + 1]
        for blk in self.blocks:
            x, _ = blk(x, cos, sin)
        return self.out_proj(self.norm(x))


class ConceptModulator(nn.Module):
    """FiLM over the loop state: (gamma, beta) from the segment's concept.
    Zero-init => phase-1 pretraining and an untrained planner are exact
    no-ops; the concept pathway fades in only as this projection learns."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, 2 * d_model, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, concept_per_token: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gamma, beta = self.proj(concept_per_token).chunk(2, dim=-1)
        return gamma, beta


def concept_loss(
    pred: torch.Tensor,    # [B, S, D] predictor outputs for segments 0..S-1
    target: torch.Tensor,  # [B, S, D] pooled hiddens (already RMS-normalized)
    mask: torch.Tensor | None = None,  # [B, S] 1 = supervise
) -> torch.Tensor:
    """Smooth-L1 regression on normalized targets (the stage-3 JEPA loss
    form). Guidance-only usage degrades gracefully under mean-regression;
    swap in a distributional head only if ablations demand it."""
    if mask is None:
        return F.smooth_l1_loss(pred.float(), target.float())
    m = mask.bool()
    if not m.any():
        return pred.sum() * 0.0
    return F.smooth_l1_loss(pred.float()[m], target.float()[m])
