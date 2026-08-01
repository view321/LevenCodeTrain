import random

from levencode.data.alignment import align_interior, align_to_corruption
from levencode.data.corruption import reconstruct
from levencode.util.lev import levenshtein

BOS, EOS = 1, 2


def rand_seq(rng, n):
    return [rng.randint(10, 30) for _ in range(n)]  # small vocab -> many matches


def test_identical_sequences_fully_matched():
    seq = [5, 6, 7, 8]
    c = align_to_corruption(seq, seq)
    assert c.n_junk() == 0 and c.n_missing() == 0
    assert c.kept_origins() == [0, 1, 2, 3]


def test_known_cases():
    # extra hyp token -> junk
    c = align_to_corruption([10, 99, 11], [10, 11])
    assert c.delete_labels() == [0, 1, 0]
    assert c.n_missing() == 0

    # missing ref token -> interior gap of 1
    c = align_to_corruption([10, 11], [10, 55, 11])
    assert c.n_junk() == 0
    assert c.gap_counts() == [1]
    assert c.gap_missing_tokens() == [[55]]

    # substitution -> junk + missing at the same spot
    c = align_to_corruption([10, 99, 11], [10, 55, 11])
    assert c.delete_labels() == [0, 1, 0]
    assert c.gap_missing_tokens() == [[55]]


def test_reconstruction_property():
    """Inverting alignment-derived labels must reproduce the reference exactly
    — same soundness property as the synthetic corruption engine."""
    rng = random.Random(0)
    for trial in range(200):
        ref = [BOS] + rand_seq(rng, rng.randint(0, 80)) + [EOS]
        # derive a hypothesis by mangling the reference
        hyp = [t for t in ref[1:-1] if rng.random() > 0.15]
        hyp = [t if rng.random() > 0.1 else rng.randint(10, 30) for t in hyp]
        for _ in range(rng.randint(0, 6)):
            hyp.insert(rng.randint(0, len(hyp)), rng.randint(10, 30))
        hyp = [BOS] + hyp + [EOS]
        c = align_interior(hyp, ref)
        assert c is not None
        assert reconstruct(c) == ref, f"trial {trial}"
        assert c.kept_sequence()[0] == BOS and c.kept_sequence()[-1] == EOS


def test_alignment_distance_is_optimal():
    """junk + missing double-counts substitutions, but the DP distance itself
    must equal the true Levenshtein distance."""
    from levencode.data.alignment import _dp_matrix
    import numpy as np

    rng = random.Random(1)
    for _ in range(100):
        a = rand_seq(rng, rng.randint(0, 40))
        b = rand_seq(rng, rng.randint(0, 40))
        dp = _dp_matrix(np.asarray(a), np.asarray(b))
        assert dp[len(a), len(b)] == levenshtein(a, b)


def test_align_interior_rejects_bad_anchors():
    assert align_interior([BOS, 5, EOS], [BOS, 5, 9], n_head=1, n_tail=1) is None
    assert align_interior([7, 5, EOS], [BOS, 5, EOS], n_head=1, n_tail=1) is None
    assert align_interior([BOS], [BOS, EOS], n_head=1, n_tail=1) is None


def test_empty_interior():
    c = align_interior([BOS, EOS], [BOS, 5, 6, EOS])
    assert c is not None
    assert c.n_missing() == 2
    assert reconstruct(c) == [BOS, 5, 6, EOS]
