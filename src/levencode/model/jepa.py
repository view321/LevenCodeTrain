"""JEPA auxiliary module: EMA target encoder + latent predictor.

The target encoder is a frozen EMA copy of the online base encoder. On the
FILL/SFT views (where the corrupted input is position-aligned with the clean
sequence), the predictor maps online hidden states to predictions of the
target encoder's hidden states at the masked positions."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


class JepaModule(nn.Module):
    def __init__(
        self,
        base_encoder: nn.Module,
        hidden_size: int,
        predictor_layers: int = 2,
        predictor_heads: int = 8,
    ):
        super().__init__()
        self.target = copy.deepcopy(base_encoder)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.target.eval()
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=predictor_heads,
            dim_feedforward=4 * hidden_size,
            batch_first=True,
            norm_first=True,
            activation="gelu",
            dropout=0.0,
        )
        self.predictor = nn.TransformerEncoder(layer, predictor_layers)

    def train(self, mode: bool = True):
        super().train(mode)
        self.target.eval()  # the EMA target never leaves eval mode
        return self

    @torch.no_grad()
    def ema_update(self, online_base: nn.Module, momentum: float) -> None:
        for pt, po in zip(self.target.parameters(), online_base.parameters()):
            pt.mul_(momentum).add_(po.detach(), alpha=1.0 - momentum)
        for bt, bo in zip(self.target.buffers(), online_base.buffers()):
            bt.copy_(bo)

    @torch.no_grad()
    def targets(
        self, clean_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        out = self.target(
            input_ids=clean_ids, attention_mask=attention_mask, return_dict=True
        )
        return out.last_hidden_state

    def predict(
        self, online_hidden: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        kpm = None
        if attention_mask is not None:
            kpm = attention_mask == 0
        return self.predictor(online_hidden, src_key_padding_mask=kpm)


def ema_momentum(step: int, total_steps: int, start: float, end: float) -> float:
    frac = min(max(step / max(total_steps, 1), 0.0), 1.0)
    return start + (end - start) * frac
