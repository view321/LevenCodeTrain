"""Loss functions for the diffusion SFT objective, edit heads, and JEPA."""

from __future__ import annotations

import torch
import torch.nn.functional as F

IGNORE = -100


def diffusion_fill_loss(
    logits: torch.Tensor,   # [B, L, V]
    labels: torch.Tensor,   # [B, L] with IGNORE off-block
    t: torch.Tensor,        # [B] mask rates
    block_len: torch.Tensor,  # [B]
) -> tuple[torch.Tensor, torch.Tensor]:
    """LLaDA-style weighted masked CE: per-sample (1/t) * sum(CE_masked) / block_len,
    averaged over the batch. Also returns the unweighted mean CE for logging."""
    B, L, V = logits.shape
    ce = F.cross_entropy(
        logits.reshape(B * L, V).float(), labels.reshape(B * L), ignore_index=IGNORE, reduction="none"
    ).reshape(B, L)
    valid = labels != IGNORE
    per_sample = ce.sum(dim=1)
    weighted = (per_sample / t.clamp_min(1e-4)) / block_len.clamp_min(1).float()
    loss = weighted.mean()
    n_valid = valid.sum().clamp_min(1)
    mean_ce = (ce * valid).sum() / n_valid
    return loss, mean_ce.detach()


def diffusion_fill_loss_sparse(
    logits_sel: torch.Tensor,  # [N, V] logits at supervised positions only
    labels_sel: torch.Tensor,  # [N]
    b_idx: torch.Tensor,       # [N] batch row of each supervised position
    t: torch.Tensor,           # [B]
    block_len: torch.Tensor,   # [B]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Identical value to diffusion_fill_loss, but computed from gathered
    logits so the full [B, L, V] tensor never has to exist in fp32."""
    B = t.shape[0]
    ce = F.cross_entropy(logits_sel.float(), labels_sel, reduction="none")  # [N]
    per_sample = torch.zeros(B, device=ce.device, dtype=ce.dtype).index_add_(0, b_idx, ce)
    weighted = (per_sample / t.clamp_min(1e-4)) / block_len.clamp_min(1).float()
    return weighted.mean(), ce.mean().detach()


def masked_ce_loss_sparse(
    logits_sel: torch.Tensor, labels_sel: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if labels_sel.numel() == 0:
        return logits_sel.sum() * 0.0, torch.zeros((), device=logits_sel.device)
    loss = F.cross_entropy(logits_sel.float(), labels_sel)
    acc = (logits_sel.argmax(-1) == labels_sel).float().mean()
    return loss, acc.detach()


def masked_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Plain masked CE (used for the FILL view where all placeholders count
    equally). Returns (loss, accuracy)."""
    B, L, V = logits.shape
    flat_logits = logits.reshape(B * L, V).float()
    flat_labels = labels.reshape(B * L)
    loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=IGNORE)
    valid = flat_labels != IGNORE
    if valid.any():
        acc = (flat_logits.argmax(-1)[valid] == flat_labels[valid]).float().mean()
    else:
        acc = torch.zeros((), device=logits.device)
        loss = logits.sum() * 0.0
    return loss, acc.detach()


def delete_loss(
    del_logits: torch.Tensor, labels: torch.Tensor, pos_weight: float | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """BCE over per-token junk labels. labels [B, L] in {0, 1}, IGNORE for pad.
    `pos_weight` upweights the rare junk class (labels are ~90% "keep", and an
    unweighted head learns to never delete — a no-op editor)."""
    valid = labels != IGNORE
    if not valid.any():
        return del_logits.sum() * 0.0, torch.zeros((), device=del_logits.device)
    target = labels.clamp_min(0).float()
    pw = None
    if pos_weight is not None and pos_weight != 1.0:
        pw = torch.tensor(float(pos_weight), device=del_logits.device)
    raw = F.binary_cross_entropy_with_logits(
        del_logits.float(), target, reduction="none", pos_weight=pw
    )
    loss = (raw * valid).sum() / valid.sum()
    pred = (torch.sigmoid(del_logits.float()) > 0.5).long()
    acc = (pred[valid] == labels[valid]).float().mean()
    return loss, acc.detach()


def insert_loss(
    ins_logits: torch.Tensor, labels: torch.Tensor, zero_weight: float | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """CE over per-gap insertion counts. ins_logits [B, G, K+1], labels [B, G].
    `zero_weight` downweights the dominant count-0 class (same imbalance story
    as the delete head, from the insertion side)."""
    B, G, C = ins_logits.shape
    labels = labels[:, :G]
    weight = None
    if zero_weight is not None and zero_weight != 1.0:
        weight = torch.ones(C, device=ins_logits.device)
        weight[0] = float(zero_weight)
    loss = F.cross_entropy(
        ins_logits.reshape(B * G, C).float(),
        labels.reshape(B * G),
        ignore_index=IGNORE,
        weight=weight,
    )
    valid = labels != IGNORE
    if valid.any():
        pred = ins_logits.argmax(-1)
        acc = (pred[valid] == labels[valid]).float().mean()
    else:
        acc = torch.zeros((), device=ins_logits.device)
        loss = ins_logits.sum() * 0.0
    return loss, acc.detach()


def jepa_loss(
    predicted: torch.Tensor,  # [B, L, H] predictor output on the corrupted/masked view
    target: torch.Tensor,     # [B, L, H] EMA-encoder hiddens on the clean view (no grad)
    positions: torch.Tensor,  # [B, L] bool: where prediction is supervised
) -> torch.Tensor:
    if not positions.any():
        return predicted.sum() * 0.0
    tgt = F.layer_norm(target.float(), target.shape[-1:])
    return F.smooth_l1_loss(predicted.float()[positions], tgt[positions])
