"""LevencodeEditor: backbone + edit heads (+ optional JEPA), with checkpoint IO.

Checkpoint layout under <dir>/:
  backbone/   HF save_pretrained (weights + config + custom modeling code)
  heads.pt    edit-head weights + meta
  jepa.pt     predictor weights (target encoder is rebuilt from the backbone)
"""

from __future__ import annotations

import inspect
import shutil
from pathlib import Path

import torch
import torch.nn as nn

from .backbone import hidden_and_logits, load_backbone
from .heads import EditHeads
from .jepa import JepaModule


class LevencodeEditor(nn.Module):
    def __init__(self, backbone, insert_max: int, jepa: JepaModule | None = None):
        super().__init__()
        self.backbone = backbone
        self.heads = EditHeads(backbone.config.hidden_size, insert_max)
        self.jepa = jepa

    @property
    def insert_max(self) -> int:
        return self.heads.insert_max

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> dict:
        h, mlm_logits = hidden_and_logits(self.backbone, input_ids, attention_mask)
        del_logits, ins_logits = self.heads(h)
        return {
            "hidden": h,
            "mlm_logits": mlm_logits,
            "del_logits": del_logits,
            "ins_logits": ins_logits,
        }

    # ---- callables for the model-agnostic samplers ----
    def mlm_call(self):
        def call(x: torch.Tensor) -> torch.Tensor:
            return hidden_and_logits(self.backbone, x)[1]

        return call

    def editor_call(self):
        def call(x: torch.Tensor) -> dict:
            return self.forward(x)

        return call

    # ---- checkpoint IO ----
    def save(self, ckpt_dir: str | Path) -> None:
        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        bdir = ckpt_dir / "backbone"
        self.backbone.save_pretrained(bdir)
        # belt and braces: make sure the custom modeling file rides along so the
        # checkpoint stays loadable from a bare directory
        src = inspect.getsourcefile(type(self.backbone))
        if src and not (bdir / Path(src).name).exists():
            shutil.copy2(src, bdir / Path(src).name)
        torch.save(
            {"heads": self.heads.state_dict(), "insert_max": self.heads.insert_max},
            ckpt_dir / "heads.pt",
        )
        if self.jepa is not None:
            torch.save({"predictor": self.jepa.predictor.state_dict()}, ckpt_dir / "jepa.pt")

    @classmethod
    def load(
        cls,
        ckpt_dir: str | Path,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        insert_max: int | None = None,
        with_jepa: bool = False,
        jepa_kwargs: dict | None = None,
    ) -> "LevencodeEditor":
        ckpt_dir = Path(ckpt_dir)
        backbone = load_backbone(str(ckpt_dir / "backbone"), dtype=dtype, device=device)
        heads_path = ckpt_dir / "heads.pt"
        meta = torch.load(heads_path, map_location="cpu", weights_only=True) if heads_path.exists() else None
        k = insert_max or (meta["insert_max"] if meta else 8)
        jepa = None
        if with_jepa:
            jepa = JepaModule(backbone.lfm2, backbone.config.hidden_size, **(jepa_kwargs or {}))
            jpath = ckpt_dir / "jepa.pt"
            if jpath.exists():
                jstate = torch.load(jpath, map_location="cpu", weights_only=True)
                jepa.predictor.load_state_dict(jstate["predictor"])
        editor = cls(backbone, k, jepa)
        if meta:
            editor.heads.load_state_dict(meta["heads"])
        return editor.to(device)


def build_editor(
    repo_or_ckpt: str | Path,
    insert_max: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    with_jepa: bool = False,
    jepa_kwargs: dict | None = None,
) -> LevencodeEditor:
    """Build from either a levencode checkpoint dir (has backbone/) or a fresh
    HF repo id / local snapshot (backbone only, heads randomly initialized)."""
    p = Path(repo_or_ckpt)
    if (p / "backbone").exists():
        return LevencodeEditor.load(
            p, device=device, dtype=dtype, insert_max=insert_max,
            with_jepa=with_jepa, jepa_kwargs=jepa_kwargs,
        )
    backbone = load_backbone(str(repo_or_ckpt), dtype=dtype, device=device)
    jepa = None
    if with_jepa:
        jepa = JepaModule(backbone.lfm2, backbone.config.hidden_size, **(jepa_kwargs or {}))
    return LevencodeEditor(backbone, insert_max, jepa).to(device)
