"""Roll-in training: edit-head supervision on the model's OWN outputs.

Synthetic corruption teaches the heads to undo random edits of correct code;
inference confronts them with the model's actual mistakes — partially-repaired
states and self-authored decoherence — a distribution they never saw. The
buffer periodically regenerates hypothesis/reference pairs with the *current*
model (LevT-style roll-in), recovers edit labels via Levenshtein alignment
(data/alignment.py), and serves batches through the same view builder as the
synthetic collator.

Two hypothesis modes:
  repair-mode: corrupt clean code, let the current editor repair it (1 round,
    temperature > 0) -> trains fixing what a round left broken.
  fill-mode: mask random positions of clean code, let the model fill them
    stochastically -> model-authored plausible-but-wrong tokens (the draft
    error class: decohered identifiers, glued names).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterator

import torch

from ..data.alignment import align_interior
from ..data.collators import build_edit_views
from ..data.corruption import Corruption, CorruptionCfg, corrupt, make_junk_sampler
from ..data.tokens import TokenizerBundle
from ..sampling.edit_sampler import EditSamplerCfg, _fill_masks, repair


@dataclass
class RollinCfg:
    enabled: bool = False
    frac: float = 0.35          # fraction of edit micro-batches drawn from the buffer
    refresh_every: int = 200    # optimizer steps between buffer regenerations
    buffer_size: int = 256      # pairs per regeneration
    max_len: int = 512          # roll-in sequence cap (alignment is O(n^2))
    mode_mix: float = 0.5       # P(repair-mode) vs fill-mode
    fill_t_min: float = 0.2
    fill_t_max: float = 0.6
    sampler: dict = field(default_factory=dict)  # EditSamplerCfg overrides for rollouts

    @classmethod
    def from_dict(cls, d: dict) -> "RollinCfg":
        d = d or {}
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


class RollinBuffer:
    def __init__(
        self,
        cfg: RollinCfg,
        bundle: TokenizerBundle,
        corruption_cfg: CorruptionCfg,
        insert_max: int,
        max_seq_len: int,
        seed: int,
    ):
        self.cfg = cfg
        self.b = bundle
        self.ccfg = corruption_cfg
        self.insert_max = insert_max
        self.max_seq_len = max_seq_len
        self.rng = random.Random(seed + 424242)
        self.scfg = EditSamplerCfg.from_dict(
            {"rounds": 1, "fill_steps": 6, "temperature": 0.7, **(cfg.sampler or {})}
        )
        self.corruptions: list[Corruption] = []
        self.mean_edit_mass = 0.0

    def ready(self, need: int = 1) -> bool:
        return len(self.corruptions) >= max(need, 1)

    # ---------- generation ----------

    def _clean_ref(self, text: str) -> list[int] | None:
        ids = self.b.encode(text)
        if len(ids) < 16:
            return None
        budget = self.cfg.max_len - 2
        if len(ids) > budget:
            s = self.rng.randint(0, len(ids) - budget)
            ids = ids[s : s + budget]
        head = [self.b.bos_id] if self.b.bos_id is not None else [self.b.eos_id]
        return head + ids + [self.b.eos_id]

    def _repair_rollout(self, editor_call, ref: list[int], device) -> list[int] | None:
        junk = make_junk_sampler(
            self.b.vocab_size, frozenset(self.b.protected | {self.b.mask_id}), echo_pool=ref
        )
        c = corrupt(ref, self.rng, self.ccfg, junk, protected=self.b.protected)
        if c.n_junk() + c.n_missing() == 0:
            return None
        out, _ = repair(editor_call, self.b, c.corrupted, self.scfg, device)
        return out

    def _fill_rollout(self, editor_call, ref: list[int], device) -> list[int] | None:
        t = self.rng.uniform(self.cfg.fill_t_min, self.cfg.fill_t_max)
        ids = [
            self.b.mask_id if (tok not in self.b.protected and self.rng.random() < t) else tok
            for tok in ref
        ]
        if self.b.mask_id not in ids:
            return None
        x = torch.tensor([ids], dtype=torch.long, device=device)
        x = _fill_masks(
            editor_call, x, self.b.mask_id, self.scfg.fill_steps,
            self.scfg.temperature, self.scfg.top_p,
        )
        return x[0].tolist()

    @torch.no_grad()
    def refresh(self, editor, code_stream: Iterator[dict], device) -> dict:
        was_training = editor.training
        editor.eval()
        editor_call = editor.editor_call()
        fresh: list[Corruption] = []
        masses: list[int] = []
        attempts = 0
        try:
            while len(fresh) < self.cfg.buffer_size and attempts < self.cfg.buffer_size * 4:
                attempts += 1
                try:
                    sample = next(code_stream)
                except StopIteration:
                    break
                ref = self._clean_ref(sample.get("text") or "")
                if ref is None:
                    continue
                if self.rng.random() < self.cfg.mode_mix:
                    hyp = self._repair_rollout(editor_call, ref, device)
                else:
                    hyp = self._fill_rollout(editor_call, ref, device)
                if hyp is None or hyp == ref:
                    continue
                c = align_interior(hyp, ref)
                if c is None:
                    continue
                mass = c.n_junk() + c.n_missing()
                if mass == 0:
                    continue
                fresh.append(c)
                masses.append(mass)
        finally:
            if was_training:
                editor.train()
        if fresh:  # keep serving the old buffer if generation came up dry
            self.corruptions = fresh
            self.mean_edit_mass = sum(masses) / len(masses)
        return {"rollin_pairs": len(fresh), "rollin_edit_mass": self.mean_edit_mass}

    # ---------- consumption ----------

    def batch(self, batch_size: int) -> dict | None:
        if not self.corruptions:
            return None
        chosen = [
            self.corruptions[self.rng.randrange(len(self.corruptions))]
            for _ in range(batch_size)
        ]
        return build_edit_views(chosen, self.b, self.insert_max, self.max_seq_len)
