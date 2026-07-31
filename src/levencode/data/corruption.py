"""Levenshtein corruption engine.

Corrupts a clean token sequence with delete / insert / substitute / span-delete
operations while tracking provenance, so the *inverse* edit operations (the
training labels for the delete head, insertion-count head, and fill head) are
known by construction — no O(n*m) alignment needed.

Provenance representation: for every token in the corrupted sequence, `origin`
holds the index of the clean token it came from, or -1 if it is inserted junk.
Kept origins are strictly increasing, which is what makes label derivation and
exact reconstruction trivial.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Sequence

JunkSampler = Callable[[random.Random], int]


@dataclass
class CorruptionCfg:
    p_delete: float = 0.10
    p_insert: float = 0.06
    p_substitute: float = 0.05
    max_insert_run: int = 3
    p_span_delete: float = 0.02
    max_span: int = 8

    @classmethod
    def from_dict(cls, d: dict) -> "CorruptionCfg":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class Corruption:
    clean: list[int]
    corrupted: list[int]
    origin: list[int]  # per corrupted token: clean index, or -1 for junk

    def delete_labels(self) -> list[int]:
        """1 for junk tokens the delete head should remove, 0 for kept."""
        return [1 if o < 0 else 0 for o in self.origin]

    def kept_sequence(self) -> list[int]:
        return [t for t, o in zip(self.corrupted, self.origin) if o >= 0]

    def kept_origins(self) -> list[int]:
        return [o for o in self.origin if o >= 0]

    def gap_counts(self) -> list[int]:
        """For the kept-only sequence: number of missing clean tokens in the gap
        after each kept token (length = len(kept) - 1, interior gaps only)."""
        idx = self.kept_origins()
        return [idx[i + 1] - idx[i] - 1 for i in range(len(idx) - 1)]

    def gap_missing_tokens(self) -> list[list[int]]:
        idx = self.kept_origins()
        return [self.clean[idx[i] + 1 : idx[i + 1]] for i in range(len(idx) - 1)]

    def n_junk(self) -> int:
        return sum(1 for o in self.origin if o < 0)

    def n_missing(self) -> int:
        return sum(self.gap_counts())


def corrupt(
    clean: Sequence[int],
    rng: random.Random,
    cfg: CorruptionCfg,
    junk_sampler: JunkSampler,
    protected: frozenset[int] = frozenset(),
) -> Corruption:
    """Apply random edits. Tokens in `protected` (BOS/EOS/specials) are never
    deleted or substituted, and junk is never inserted before position 0's
    protected prefix run — guaranteeing boundary anchors survive."""
    clean = list(clean)
    corrupted: list[int] = []
    origin: list[int] = []
    n = len(clean)

    def maybe_insert() -> None:
        if rng.random() < cfg.p_insert:
            for _ in range(rng.randint(1, cfg.max_insert_run)):
                corrupted.append(junk_sampler(rng))
                origin.append(-1)

    i = 0
    while i < n:
        tok = clean[i]
        if tok in protected:
            corrupted.append(tok)
            origin.append(i)
            i += 1
            continue
        maybe_insert()
        r = rng.random()
        if r < cfg.p_span_delete and i + 2 <= n:
            span = rng.randint(2, cfg.max_span)
            j = i
            while j < min(i + span, n) and clean[j] not in protected:
                j += 1
            i = j
            continue
        r -= cfg.p_span_delete
        if r < cfg.p_delete:
            i += 1
            continue
        r -= cfg.p_delete
        if r < cfg.p_substitute:
            corrupted.append(junk_sampler(rng))
            origin.append(-1)
            i += 1
            continue
        corrupted.append(tok)
        origin.append(i)
        i += 1

    return Corruption(clean=clean, corrupted=corrupted, origin=origin)


def reconstruct(c: Corruption) -> list[int]:
    """Invert the corruption using only the labels the model is trained to
    predict. Used as a property test: must equal `c.clean` exactly."""
    kept = c.kept_sequence()
    idx = c.kept_origins()
    if not kept:
        return list(c.clean)
    out: list[int] = list(c.clean[: idx[0]])  # missing prefix (empty if BOS kept)
    gaps = c.gap_missing_tokens()
    for i, tok in enumerate(kept):
        out.append(tok)
        if i < len(gaps):
            out.extend(gaps[i])
    out.extend(c.clean[idx[-1] + 1 :])  # missing suffix (empty if EOS kept)
    return out


def make_junk_sampler(
    vocab_size: int,
    forbidden: frozenset[int],
    echo_pool: Sequence[int] | None = None,
    p_echo: float = 0.3,
) -> JunkSampler:
    """Junk = uniform random vocab token, or (with p_echo) a token copied from
    the clean sequence — closer to realistic confusions than pure noise."""
    pool = [t for t in (echo_pool or []) if t not in forbidden]

    def sample(rng: random.Random) -> int:
        if pool and rng.random() < p_echo:
            return rng.choice(pool)
        while True:
            t = rng.randrange(vocab_size)
            if t not in forbidden:
                return t

    return sample
