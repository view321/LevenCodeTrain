"""Levenshtein repair loop: delete -> insert placeholders -> fill, iterated.

`editor_call` is any callable(input_ids [1, L]) -> dict with keys
  mlm_logits [1, L, V], del_logits [1, L], ins_logits [1, L-1, K+1].
Trained heads come from model/editor.py; tests use scripted mocks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from ..data.tokens import TokenizerBundle
from .block_sampler import pick_token


@dataclass
class EditSamplerCfg:
    rounds: int = 3
    delete_threshold: float = 0.5
    fill_steps: int = 8
    temperature: float = 0.0
    top_p: float = 0.9
    max_len: int = 2048
    # Subtracted from the count-0 logit before the insertion argmax; > 0 makes
    # the editor less conservative about inserting (counters class imbalance).
    ins_zero_penalty: float = 0.0

    @classmethod
    def from_dict(cls, d: dict) -> "EditSamplerCfg":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class EditTrace:
    rounds_used: int
    deleted: int
    inserted: int


@torch.no_grad()
def _fill_masks(
    editor_call: Callable,
    x: torch.Tensor,
    mask_id: int,
    steps: int,
    temperature: float,
    top_p: float,
) -> torch.Tensor:
    masked = (x[0] == mask_id).nonzero(as_tuple=False).squeeze(-1).tolist()
    for step in range(steps):
        if not masked:
            break
        logits = editor_call(x)["mlm_logits"]
        pos = torch.tensor(masked, dtype=torch.long, device=x.device)
        tok, conf = pick_token(logits[0, pos, :], temperature, top_p, None)
        k = min(math.ceil(len(masked) / (steps - step)), len(masked))
        commit = conf.argsort(descending=True)[:k]
        x[0, pos[commit]] = tok[commit].to(x.dtype)
        committed = set(pos[commit].tolist())
        masked = [p for p in masked if p not in committed]
    return x


@torch.no_grad()
def repair(
    editor_call: Callable[[torch.Tensor], dict],
    bundle: TokenizerBundle,
    ids: Sequence[int],
    cfg: EditSamplerCfg,
    device: torch.device | str = "cpu",
) -> tuple[list[int], EditTrace]:
    seq = list(ids)
    protected = bundle.protected
    total_del = total_ins = 0
    rounds_used = 0
    # A repair should stay near the input's length; without this, untrained or
    # miscalibrated insert heads balloon the sequence to cfg.max_len with noise.
    len_cap = min(cfg.max_len, int(len(seq) * 1.5) + 64)

    for _ in range(cfg.rounds):
        rounds_used += 1
        x = torch.tensor([seq], dtype=torch.long, device=device)

        # 1) delete pass
        out = editor_call(x)
        del_p = torch.sigmoid(out["del_logits"][0].float())
        keep = [
            i
            for i in range(len(seq))
            if seq[i] in protected or del_p[i].item() < cfg.delete_threshold
        ]
        n_del = len(seq) - len(keep)
        seq = [seq[i] for i in keep]
        total_del += n_del

        # 2) insertion pass on the post-deletion sequence
        x = torch.tensor([seq], dtype=torch.long, device=device)
        out = editor_call(x)
        n_ins = 0
        if len(seq) >= 2:
            ins_logits = out["ins_logits"][0].float()
            if cfg.ins_zero_penalty:
                ins_logits = ins_logits.clone()
                ins_logits[:, 0] -= cfg.ins_zero_penalty
            counts = ins_logits.argmax(dim=-1).tolist()  # per gap after token i
            new_seq: list[int] = []
            for i, tok in enumerate(seq):
                new_seq.append(tok)
                if i < len(counts) and counts[i] > 0 and len(new_seq) < len_cap:
                    n = int(counts[i])
                    new_seq.extend([bundle.mask_id] * n)
                    n_ins += n
            seq = new_seq[:len_cap]
        total_ins += n_ins

        # 3) fill pass
        if bundle.mask_id in seq:
            x = torch.tensor([seq], dtype=torch.long, device=device)
            x = _fill_masks(
                editor_call, x, bundle.mask_id, cfg.fill_steps, cfg.temperature, cfg.top_p
            )
            seq = x[0].tolist()

        if n_del == 0 and n_ins == 0:
            break

    return seq, EditTrace(rounds_used=rounds_used, deleted=total_del, inserted=total_ins)
