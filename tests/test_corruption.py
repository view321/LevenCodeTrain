import random

from levencode.data.corruption import (
    Corruption,
    CorruptionCfg,
    corrupt,
    make_junk_sampler,
    reconstruct,
)

BOS, EOS = 1, 2
PROTECTED = frozenset({BOS, EOS})


def make_clean(rng: random.Random, n: int) -> list[int]:
    return [BOS] + [rng.randint(10, 999) for _ in range(n)] + [EOS]


def junk(rng: random.Random) -> int:
    return rng.randint(10, 999)


def test_reconstruction_property():
    """Inverting a corruption using only the derived labels must recover the
    clean sequence exactly — this is what makes edit supervision sound."""
    rng = random.Random(0)
    for trial in range(200):
        cfg = CorruptionCfg(
            p_delete=rng.uniform(0, 0.3),
            p_insert=rng.uniform(0, 0.2),
            p_substitute=rng.uniform(0, 0.2),
            max_insert_run=rng.randint(1, 4),
            p_span_delete=rng.uniform(0, 0.05),
            max_span=rng.randint(2, 8),
        )
        clean = make_clean(rng, rng.randint(5, 120))
        c = corrupt(clean, rng, cfg, junk, protected=PROTECTED)
        assert reconstruct(c) == clean, f"trial {trial}"


def test_protected_tokens_survive():
    rng = random.Random(1)
    cfg = CorruptionCfg(p_delete=0.9, p_substitute=0.09, p_insert=0.3)
    for _ in range(50):
        clean = make_clean(rng, 40)
        c = corrupt(clean, rng, cfg, junk, protected=PROTECTED)
        kept = c.kept_sequence()
        assert kept[0] == BOS and kept[-1] == EOS
        origins = c.kept_origins()
        assert origins[0] == 0 and origins[-1] == len(clean) - 1


def test_labels_coherent():
    rng = random.Random(2)
    cfg = CorruptionCfg()
    clean = make_clean(rng, 200)
    c = corrupt(clean, rng, cfg, junk, protected=PROTECTED)
    labels = c.delete_labels()
    assert len(labels) == len(c.corrupted)
    assert sum(labels) == c.n_junk()
    assert sum(c.gap_counts()) == c.n_missing()
    assert all(g >= 0 for g in c.gap_counts())
    # kept token values must match the clean tokens at their origins
    for tok, o in zip(c.corrupted, c.origin):
        if o >= 0:
            assert clean[o] == tok


def test_determinism():
    cfg = CorruptionCfg()
    clean = make_clean(random.Random(3), 100)
    a = corrupt(clean, random.Random(42), cfg, junk, protected=PROTECTED)
    b = corrupt(clean, random.Random(42), cfg, junk, protected=PROTECTED)
    assert a.corrupted == b.corrupted and a.origin == b.origin


def test_rates_roughly_match():
    rng = random.Random(4)
    cfg = CorruptionCfg(p_delete=0.5, p_insert=0.0, p_substitute=0.0, p_span_delete=0.0)
    clean = make_clean(rng, 4000)
    c = corrupt(clean, rng, cfg, junk, protected=PROTECTED)
    frac_missing = c.n_missing() / 4000
    assert 0.4 < frac_missing < 0.6


def test_junk_sampler_avoids_forbidden():
    rng = random.Random(5)
    forbidden = frozenset(range(0, 500))
    sampler = make_junk_sampler(1000, forbidden, echo_pool=[600, 601], p_echo=0.5)
    for _ in range(500):
        t = sampler(rng)
        assert t not in forbidden
