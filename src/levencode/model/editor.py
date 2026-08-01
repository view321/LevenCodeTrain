"""LevencodeEditor: backbone + edit heads (+ optional JEPA), with checkpoint IO.

Checkpoint layout under <dir>/:
  backbone/   HF save_pretrained (weights + config + custom modeling code)
  heads.pt    edit-head weights + meta
  jepa.pt     predictor weights (target encoder is rebuilt from the backbone)
  latent.pt   latent JEPA stack (RVQ + decodability adapter + level heads)
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
    def __init__(
        self,
        backbone,
        insert_max: int,
        jepa: JepaModule | None = None,
        latent=None,
        latent_kwargs: dict | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.heads = EditHeads(backbone.config.hidden_size, insert_max)
        self.jepa = jepa
        if latent is None and latent_kwargs:
            from ..latent.bundle import LatentBundle

            latent = LatentBundle(
                latent_dim=latent_kwargs.get("latent_dim", backbone.config.hidden_size),
                student_hidden=backbone.config.hidden_size,
                **{k: v for k, v in latent_kwargs.items() if k != "latent_dim"},
            )
        self.latent = latent

    @property
    def insert_max(self) -> int:
        return self.heads.insert_max

    def hidden(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Base-encoder hidden states only — no LM-head logits. The training
        path uses this and applies lm_head just at supervised positions; the
        full-vocab logits tensor is the largest allocation in a step and most
        views never need it."""
        return self.backbone.lfm2(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        ).last_hidden_state

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
        if self.latent is not None:
            torch.save({"latent": self.latent.state_dict()}, ckpt_dir / "latent.pt")

    @classmethod
    def load(
        cls,
        ckpt_dir: str | Path,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        insert_max: int | None = None,
        with_jepa: bool = False,
        jepa_kwargs: dict | None = None,
        with_latent: bool = False,
        latent_kwargs: dict | None = None,
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
        latent = None
        if with_latent:
            from ..latent.bundle import LatentBundle

            lk = dict(latent_kwargs or {})
            ldim = lk.pop("latent_dim", backbone.config.hidden_size)
            latent = LatentBundle(latent_dim=ldim, student_hidden=backbone.config.hidden_size, **lk)
            lpath = ckpt_dir / "latent.pt"
            if lpath.exists():
                lstate = torch.load(lpath, map_location="cpu", weights_only=True)
                latent.load_state_dict(lstate["latent"])
        editor = cls(backbone, k, jepa, latent)
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
    with_latent: bool = False,
    latent_kwargs: dict | None = None,
) -> LevencodeEditor:
    """Build from either a levencode checkpoint dir (has backbone/) or a fresh
    HF repo id / local snapshot (backbone only, heads randomly initialized)."""
    p = Path(repo_or_ckpt)
    if (p / "backbone").exists():
        return LevencodeEditor.load(
            p, device=device, dtype=dtype, insert_max=insert_max,
            with_jepa=with_jepa, jepa_kwargs=jepa_kwargs,
            with_latent=with_latent, latent_kwargs=latent_kwargs,
        )
    backbone = load_backbone(str(repo_or_ckpt), dtype=dtype, device=device)
    jepa = None
    if with_jepa:
        jepa = JepaModule(backbone.lfm2, backbone.config.hidden_size, **(jepa_kwargs or {}))
    latent = None
    if with_latent:
        from ..latent.bundle import LatentBundle

        lk = dict(latent_kwargs or {})
        ldim = lk.pop("latent_dim", backbone.config.hidden_size)
        latent = LatentBundle(latent_dim=ldim, student_hidden=backbone.config.hidden_size, **lk)
    return LevencodeEditor(backbone, insert_max, jepa, latent).to(device)
