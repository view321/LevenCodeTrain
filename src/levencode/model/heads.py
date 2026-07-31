"""Levenshtein edit heads on top of the encoder's hidden states.

delete: per-token binary logit — "this token is junk, remove it".
insert: per-gap (K+1)-way logits over how many placeholder masks to insert
        between adjacent tokens; the pretrained MLM head then fills them."""

from __future__ import annotations

import torch
import torch.nn as nn


class EditHeads(nn.Module):
    def __init__(self, hidden_size: int, insert_max: int):
        super().__init__()
        self.insert_max = insert_max
        mid = max(hidden_size // 2, 32)
        self.delete = nn.Sequential(
            nn.Linear(hidden_size, mid), nn.GELU(), nn.Linear(mid, 1)
        )
        self.insert = nn.Sequential(
            nn.Linear(2 * hidden_size, mid), nn.GELU(), nn.Linear(mid, insert_max + 1)
        )

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        del_logits = self.delete(h).squeeze(-1)  # [B, L]
        pair = torch.cat([h[:, :-1, :], h[:, 1:, :]], dim=-1)  # [B, L-1, 2H]
        ins_logits = self.insert(pair)  # [B, L-1, K+1]
        return del_logits, ins_logits
