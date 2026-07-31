"""Levenshtein distance over arbitrary sequences (token id lists or strings)."""

from __future__ import annotations

from typing import Sequence


def levenshtein(a: Sequence, b: Sequence, cap: int | None = None) -> int:
    """Two-row DP edit distance. If `cap` is given and both length difference
    already exceeds it, returns cap+1 early (distance is at least that)."""
    if a == b:
        return 0
    if cap is not None and abs(len(a) - len(b)) > cap:
        return cap + 1
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        if cap is not None and min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def lev_reduction(corrupted: Sequence, output: Sequence, reference: Sequence) -> float:
    """1.0 = output equals reference; 0.0 = no closer than the corrupted input;
    negative = output drifted further from the reference than the corruption."""
    d_before = levenshtein(corrupted, reference)
    if d_before == 0:
        return 1.0 if list(output) == list(reference) else 0.0
    d_after = levenshtein(output, reference)
    return (d_before - d_after) / d_before
