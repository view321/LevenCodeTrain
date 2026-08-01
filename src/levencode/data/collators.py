"""Collators turning normalized samples into training batches.

DiffusionSFTCollator — block-pattern masked-diffusion SFT (LLaDA-style loss
restricted to a contiguous block of the answer): clean context, one block
masked at rate t ~ U(t_min, 1), labels only on masked positions. Chat samples
mask inside the assistant answer (plus an EOS-pad tail so termination is
learned); raw code samples mask a block anywhere (true infill training).

EditCollator — builds the three Levenshtein-editor training views from one
corruption: DEL (find junk), INS (predict missing-token counts per gap on the
kept sequence), FILL (predict missing tokens at inserted placeholder masks).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from .corruption import Corruption, CorruptionCfg, corrupt, make_junk_sampler
from .tokens import TokenizerBundle

IGNORE = -100


def _pad_batch(rows: list[list[int]], pad_value: int) -> torch.Tensor:
    width = max(len(r) for r in rows)
    out = torch.full((len(rows), width), pad_value, dtype=torch.long)
    for i, r in enumerate(rows):
        out[i, : len(r)] = torch.tensor(r, dtype=torch.long)
    return out


def _attention_from(rows: list[list[int]], width: int | None = None) -> torch.Tensor:
    width = width or max(len(r) for r in rows)
    att = torch.zeros((len(rows), width), dtype=torch.long)
    for i, r in enumerate(rows):
        att[i, : len(r)] = 1
    return att


@dataclass
class SFTExample:
    seq: list[int]
    block_start: int
    block_end: int


class DiffusionSFTCollator:
    def __init__(self, bundle: TokenizerBundle, diffusion_cfg: dict, max_seq_len: int, seed: int = 0):
        self.b = bundle
        self.cfg = diffusion_cfg
        self.max_len = max_seq_len
        self.rng = random.Random(seed)

    # ---------- sample preparation ----------

    def _prep_chat(self, messages: list[dict]) -> SFTExample | None:
        try:
            prefix, answer = self.b.chat_pair_ids(messages)
        except Exception:
            return None
        if not answer:
            return None
        eos_pad = int(self.cfg.get("eos_pad", 16))
        seq = prefix + answer + [self.b.eos_id] * eos_pad
        region_start, region_end = len(prefix), len(seq)

        if len(seq) > self.max_len:
            overflow = len(seq) - self.max_len
            if overflow < region_start:  # trim context head, keep the answer
                seq = seq[overflow:]
                region_start -= overflow
                region_end -= overflow
            else:  # answer alone exceeds max_len: keep its head
                seq = seq[region_start : region_start + self.max_len]
                region_start, region_end = 0, len(seq)
        if region_end - region_start < 2:
            return None
        return self._choose_block(seq, region_start, region_end, allow_full=True)

    def _prep_text(self, text: str) -> SFTExample | None:
        ids = self.b.encode(text)
        if len(ids) < 32:
            return None
        head = [self.b.bos_id] if self.b.bos_id is not None else []
        budget = self.max_len - len(head) - 1
        if len(ids) > budget:  # random window into long files
            start = self.rng.randint(0, len(ids) - budget)
            ids = ids[start : start + budget]
        seq = head + ids + [self.b.eos_id]
        return self._choose_block(seq, len(head), len(seq), allow_full=False)

    def _choose_block(self, seq: list[int], rs: int, re: int, allow_full: bool) -> SFTExample:
        if allow_full and self.rng.random() < float(self.cfg.get("full_answer_prob", 0.3)):
            return SFTExample(seq, rs, min(re, rs + 256))
        bmin = int(self.cfg.get("block_size_min", 16))
        bmax = int(self.cfg.get("block_size_max", 128))
        b = self.rng.randint(bmin, bmax)
        b = min(b, re - rs)
        s = self.rng.randint(rs, max(rs, re - b))
        return SFTExample(seq, s, min(s + b, re))

    # ---------- batching ----------

    def __call__(self, samples: list[dict]) -> dict | None:
        rows, labels, ts, block_lens = [], [], [], []
        t_min = float(self.cfg.get("t_min", 0.05))
        for sample in samples:
            ex = (
                self._prep_chat(sample["messages"])
                if "messages" in sample
                else self._prep_text(sample["text"])
            )
            if ex is None:
                continue
            t = t_min + (1.0 - t_min) * self.rng.random()
            ids = list(ex.seq)
            lab = [IGNORE] * len(ids)
            masked_any = False
            for pos in range(ex.block_start, ex.block_end):
                if self.rng.random() < t:
                    lab[pos] = ids[pos]
                    ids[pos] = self.b.mask_id
                    masked_any = True
            if not masked_any:
                pos = self.rng.randint(ex.block_start, ex.block_end - 1)
                lab[pos] = ex.seq[pos]
                ids[pos] = self.b.mask_id
            rows.append(ids)
            labels.append(lab)
            ts.append(t)
            block_lens.append(ex.block_end - ex.block_start)
        if not rows:
            return None
        return {
            "input_ids": _pad_batch(rows, self.b.pad_id),
            "attention_mask": _attention_from(rows),
            "labels": _pad_batch(labels, IGNORE),
            "t": torch.tensor(ts, dtype=torch.float32),
            "block_len": torch.tensor(block_lens, dtype=torch.long),
        }


class EditCollator:
    def __init__(
        self,
        bundle: TokenizerBundle,
        corruption_cfg: CorruptionCfg,
        insert_max: int,
        max_seq_len: int,
        seed: int = 0,
    ):
        self.b = bundle
        self.ccfg = corruption_cfg
        self.k = insert_max
        self.max_len = max_seq_len
        self.rng = random.Random(seed)
        self.forbidden = frozenset(bundle.protected | {bundle.mask_id})

    def _corrupt_one(self, text: str) -> Corruption | None:
        ids = self.b.encode(text)
        if len(ids) < 16:
            return None
        cap = int(self.max_len * 0.85)
        if len(ids) > cap:
            start = self.rng.randint(0, len(ids) - cap)
            ids = ids[start : start + cap]
        head = [self.b.bos_id] if self.b.bos_id is not None else [self.b.eos_id]
        seq = head + ids + [self.b.eos_id]
        junk = make_junk_sampler(self.b.vocab_size, self.forbidden, echo_pool=ids)
        for _ in range(3):
            c = corrupt(seq, self.rng, self.ccfg, junk, protected=self.b.protected)
            if c.n_junk() + c.n_missing() > 0:
                return c
        return c  # identity corruption: all-zero labels are still valid supervision

    def __call__(self, samples: list[dict]) -> dict | None:
        corruptions = []
        for sample in samples:
            text = sample.get("text")
            if not text:
                continue
            c = self._corrupt_one(text)
            if c is not None:
                corruptions.append(c)
        return build_edit_views(corruptions, self.b, self.k, self.max_len)


def build_edit_views(
    corruptions: list[Corruption],
    bundle: TokenizerBundle,
    insert_max: int,
    max_len: int,
) -> dict | None:
    """The three training views from Corruption objects — shared between the
    synthetic-corruption collator and alignment-derived roll-in batches."""
    del_rows, del_labs = [], []
    ins_rows, ins_labs = [], []
    fill_rows, fill_labs = [], []
    for c in corruptions:
        del_rows.append(list(c.corrupted))
        del_labs.append(c.delete_labels())

        kept = c.kept_sequence()
        gaps = c.gap_counts()
        ins_rows.append(kept)
        ins_labs.append([min(g, insert_max) for g in gaps] + [IGNORE])  # no gap after last token

        fill_ids: list[int] = []
        fill_lab: list[int] = []
        missing = c.gap_missing_tokens()
        for i, tok in enumerate(kept):
            fill_ids.append(tok)
            fill_lab.append(IGNORE)
            if i < len(missing):
                for m in missing[i]:
                    fill_ids.append(bundle.mask_id)
                    fill_lab.append(m)
        fill_rows.append(fill_ids[:max_len])
        fill_labs.append(fill_lab[:max_len])

    if not del_rows:
        return None

    def view(rows: list[list[int]], labs: list[list[int]]) -> dict:
        return {
            "input_ids": _pad_batch(rows, bundle.pad_id),
            "attention_mask": _attention_from(rows),
            "labels": _pad_batch(labs, IGNORE),
        }

    return {
        "del": view(del_rows, del_labs),
        "ins": view(ins_rows, ins_labs),
        "fill": view(fill_rows, fill_labs),
    }
