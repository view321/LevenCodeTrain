"""Losses for the latent stack: code CE (AR prior), CALM energy loss,
energy-scorer ranking (CFG-trained), and auxiliaries."""

from __future__ import annotations

import torch
import torch.nn.functional as F

IGNORE = -100


def code_ce_loss(
    b1_logits: torch.Tensor,  # [B, T+1, C] (position 0 predicts chunk 0's b1; the
    #                            extra trailing position predicts the next chunk)
    b2_logits: torch.Tensor,  # [B, T, C]
    codes: torch.Tensor,      # [B, T, 2]
    mask: torch.Tensor,       # [B, T] 1=real
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, _ = codes.shape
    C = b1_logits.shape[-1]
    lab = codes.reshape(B * T, 2)
    l1 = b1_logits[:, :T].reshape(B * T, C).float()  # drop the next-chunk slot
    l2 = b2_logits.reshape(B * T, C).float()
    m = mask.reshape(B * T).bool()
    if not m.any():
        return (b1_logits.sum() * 0.0), torch.zeros((), device=b1_logits.device), torch.zeros((), device=b1_logits.device)
    loss = F.cross_entropy(l1[m], lab[m, 0]) + F.cross_entropy(l2[m], lab[m, 1])
    acc1 = (l1[m].argmax(-1) == lab[m, 0]).float().mean()
    acc2 = (l2[m].argmax(-1) == lab[m, 1]).float().mean()
    return loss, acc1.detach(), acc2.detach()


def energy_loss(
    samples: torch.Tensor,  # [B, N, D] draws from the residual head
    targets: torch.Tensor,  # [B, M, D] draws from the target posterior
) -> torch.Tensor:
    """CALM energy score MC estimate (eq. 10, alpha=1). Strictly proper.

    Row-chunked: the M x N x D distance tensor is cheap per row but enormous
    over the batch at latent dims of ~2k, so we accumulate over slices."""
    B, N, D = samples.shape
    _, M, _ = targets.shape
    total = 0.0
    chunk = max(1, min(B, 4096 // max(M * N, 1)))
    for c0 in range(0, B, chunk):
        s = samples[c0 : c0 + chunk]
        t = targets[c0 : c0 + chunk]
        diff = t.unsqueeze(2) - s.unsqueeze(1)  # [C, M, N, D]
        d = diff.norm(dim=-1)  # [C, M, N]
        term1 = 2.0 * d.mean(dim=(1, 2))
        d2 = s.unsqueeze(2) - s.unsqueeze(1)  # [C, N, N, D]
        d2 = d2.norm(dim=-1)
        triu = torch.triu(torch.ones(N, N, device=samples.device), diagonal=1).bool()
        term2 = d2[:, triu].mean(dim=-1)
        total = total + (term1 - term2).sum()
    return total / B


def scorer_loss(
    energy_pos: torch.Tensor,  # [B] E(cond, z_pos) — must be low
    energy_neg: torch.Tensor,  # [B] E(cond, z_neg) — must be high
    margin: float = 1.0,
) -> torch.Tensor:
    """Hinge ranking: E_pos + margin < E_neg."""
    return F.relu(energy_pos - energy_neg + margin).mean()


def neg_hinge(energy_neg: torch.Tensor) -> torch.Tensor:
    """Push unconditional energies up (calibration of the null branch)."""
    return F.relu(-energy_neg + 0.5).mean()
