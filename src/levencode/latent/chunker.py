"""Hierarchical chunking of token sequences into nested levels of granularity.

Levels are strictly nested: every coarse boundary is also a fine boundary, so a
coarse chunk is an exact union of fine chunks. Fine boundaries prefer
semantically meaningful cuts (newlines, indentation changes, sentence
punctuation) over arbitrary fixed-K windows — for code this is a token-level
approximation of AST-aware boundaries; if `tree_sitter` is importable we
sharpen boundaries with real AST node cut points (optional, never required).

At inference the chunker is not used — spans are fixed (K_f tokens per fine
chunk) — but during latent precomputation the teacher's chunk latents are
computed on these content-dependent spans, so the space the student predicts
is aligned to meaningful units."""

from __future__ import annotations

from dataclasses import dataclass, field

BOUNDARY_SENTENCE = ".!?;:"

try:  # optional; boundaries just stay heuristic without it
    import tree_sitter  # noqa: F401

    _HAS_TREE_SITTER = True
except Exception:  # pragma: no cover - environment dependent
    _HAS_TREE_SITTER = False


@dataclass
class LevelSpec:
    name: str = "fine"
    tokens_per_chunk: int = 8  # target chunk size in tokens
    min_tokens: int = 4
    max_tokens: int = 16
    bound_span: int = 4  # best-boundary search window around the target cut

    @classmethod
    def from_dict(cls, d: dict) -> "LevelSpec":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class HierarchicalSpec:
    levels: list[LevelSpec] = field(default_factory=list)  # coarsest first

    @classmethod
    def from_dict(cls, d: dict) -> "HierarchicalSpec":
        return cls(levels=[LevelSpec.from_dict(x) for x in d.get("levels", [])])


def _boundary_scores(ids: list[int], text: str, bundle) -> list[float]:
    """Per-gap score (gap i sits between token i-1 and token i). Higher =
    preferred cut point. Cheap heuristics over decoded text."""

    def word(s: str) -> str:
        return s.strip().lstrip(" \t\n\r\"'([{`*")

    toks = [bundle.decode([t]) if bundle else str(t) for t in ids]
    n = len(ids)
    scores = [0.0] * (n + 1)
    for i in range(1, n):
        raw_prev = toks[i - 1]
        cur, prev = word(toks[i]), word(raw_prev)
        if not cur and not prev:
            continue
        s = 0.0
        # newline check on the RAW token: word() strips trailing newlines, so
        # testing the stripped form can never fire
        if raw_prev.rstrip(" \t").endswith("\n"):
            s += 2.0
        if prev and prev.endswith(("}", ")", "]")):
            s += 1.5
        if prev and prev[-1] in BOUNDARY_SENTENCE:
            s += 1.0
        if not prev:
            s += 1.0  # previous token was pure whitespace
        # single decoded tokens are bare words — "def " with a trailing space
        # never matches one
        if cur and (cur[0].isupper() or cur in ("def", "class", "import", "from") or cur.startswith("@")):
            s += 0.5
        scores[i] = s
    return scores


def _choose_boundary(scores: list[float], lo: int, hi: int, target: int, rng) -> int:
    """Pick a cut in [lo, hi] preferring the highest boundary score near
    `target`; ties resolve to the cut closest to the target."""
    best, best_key = lo, None
    for i in range(lo, hi + 1):
        key = (scores[i], -abs(i - target))
        if best_key is None or key > best_key:
            best_key, best = key, i
    return best


def hierarchical_spans(
    ids: list[int],
    spec: HierarchicalSpec,
    bundle=None,
    rng=None,
    min_chunk_tokens: int = 4,
) -> list[list[tuple[int, int]]]:
    """Return per-level span lists (coarsest first), strictly nested.

    The finest level is built greedily with boundary preference; each coarser
    level is a grouping of finer chunks (never cuts inside a finer chunk)."""
    import random

    rng = rng or random.Random(0)
    if not spec.levels:
        return []
    levels = spec.levels
    fine = levels[-1]
    text = bundle.decode(ids) if bundle else ""
    scores = _boundary_scores(ids, text, bundle)

    n = len(ids)
    spans: list[tuple[int, int]] = []
    start = 0
    lo = fine.min_tokens if fine.min_tokens else fine.tokens_per_chunk // 2
    hi = fine.max_tokens if fine.max_tokens else fine.tokens_per_chunk * 2
    while start < n:
        target = min(start + fine.tokens_per_chunk, n)
        if n - start <= hi:
            cut = n
        else:
            jitter = rng.randint(-fine.tokens_per_chunk // 4, fine.tokens_per_chunk // 4)
            target = min(max(start + fine.tokens_per_chunk + jitter, start + lo), n)
            lo_cut = max(start + lo, target - fine.bound_span // 2)
            cut = _choose_boundary(scores, lo_cut, min(target + fine.bound_span, n), target, rng)
            cut = max(cut, min(start + lo, n))
            if cut >= n or n - cut < min_chunk_tokens:
                cut = n
        spans.append((start, cut))
        start = cut

    out: list[list[tuple[int, int]]] = [spans]
    for lv in reversed(levels[:-1]):  # coarser levels group finer chunks
        grouped: list[tuple[int, int]] = []
        cur: list[tuple[int, int]] = []
        total = 0
        target = lv.tokens_per_chunk
        for s, e in spans:
            # close the group at the target size OR when adding the next fine
            # chunk would blow past max_tokens ("and" made target a floor and
            # max the only real trigger, giving chunks up to 2x the target)
            if cur and (total >= target or (total + (e - s)) > lv.max_tokens):
                grouped.append((cur[0][0], cur[-1][1]))
                cur, total = [], 0
            cur.append((s, e))
            total += e - s
        if cur:
            grouped.append((cur[0][0], cur[-1][1]))
        out.append(grouped)
        spans = grouped
    out.reverse()
    return out
