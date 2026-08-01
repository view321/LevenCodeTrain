"""Levenshtein alignment with backtrace, expressed as a `Corruption`.

For roll-in training the model's own output (hypothesis) plays the role of
"corrupted" and the reference plays "clean" — but unlike synthetic corruption,
provenance is unknown and must be recovered by alignment. The backtrace emits
exactly the Corruption structure the edit collator already consumes, so all
label derivation and its tested invariants are reused unchanged.

DP cost: match 0, substitute/delete/insert 1. Rows are fully vectorized via
the running-minimum identity dp[i,j] = j + min_{k<=j}(c[k] - k), where c holds
the vertical/diagonal candidates — so a 500x500 alignment is a few numpy ops
per row instead of 250k Python iterations.
"""

from __future__ import annotations

import numpy as np

from .corruption import Corruption


def _dp_matrix(hyp: np.ndarray, ref: np.ndarray) -> np.ndarray:
    n, m = len(hyp), len(ref)
    dp = np.empty((n + 1, m + 1), dtype=np.int32)
    ar = np.arange(m + 1, dtype=np.int32)
    dp[0] = ar
    for i in range(1, n + 1):
        prev = dp[i - 1]
        c = np.empty(m + 1, dtype=np.int32)
        c[0] = prev[0] + 1
        if m:
            sub_cost = (ref != hyp[i - 1]).astype(np.int32)
            c[1:] = np.minimum(prev[1:] + 1, prev[:-1] + sub_cost)
        dp[i] = np.minimum.accumulate(c - ar) + ar
    return dp


def align_to_corruption(hyp: list[int], ref: list[int]) -> Corruption:
    """Optimal-alignment provenance: origin[i] = matched ref index for hyp[i],
    or -1 (junk). Substitutions become junk + a missing ref token, matching the
    synthetic corruption engine's semantics exactly."""
    h = np.asarray(hyp, dtype=np.int64)
    r = np.asarray(ref, dtype=np.int64)
    dp = _dp_matrix(h, r)
    origin = [0] * len(hyp)
    i, j = len(hyp), len(ref)
    while i > 0 or j > 0:
        v = dp[i, j]
        if i > 0 and j > 0 and hyp[i - 1] == ref[j - 1] and v == dp[i - 1, j - 1]:
            origin[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and v == dp[i - 1, j - 1] + 1:  # substitution
            origin[i - 1] = -1
            i, j = i - 1, j - 1
        elif i > 0 and v == dp[i - 1, j] + 1:  # hyp token is junk
            origin[i - 1] = -1
            i -= 1
        else:  # ref token missing from hyp
            j -= 1
    return Corruption(clean=list(ref), corrupted=list(hyp), origin=origin)


def align_interior(
    hyp: list[int], ref: list[int], n_head: int = 1, n_tail: int = 1
) -> Corruption | None:
    """Align with the boundary anchors (BOS/EOS) force-matched, guaranteeing
    the kept sequence spans the reference — the invariant the edit views rely
    on (all missing tokens live in interior gaps). Returns None if the anchors
    disagree between hypothesis and reference."""
    if len(hyp) < n_head + n_tail or len(ref) < n_head + n_tail:
        return None
    if hyp[:n_head] != ref[:n_head]:
        return None
    if n_tail and hyp[len(hyp) - n_tail :] != ref[len(ref) - n_tail :]:
        return None
    core = align_to_corruption(
        hyp[n_head : len(hyp) - n_tail], ref[n_head : len(ref) - n_tail]
    )
    origin = (
        list(range(n_head))
        + [o + n_head if o >= 0 else -1 for o in core.origin]
        + [len(ref) - n_tail + k for k in range(n_tail)]
    )
    return Corruption(clean=list(ref), corrupted=list(hyp), origin=origin)
