from levencode.util.lev import lev_reduction, levenshtein


def test_known_distances():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("", "abc") == 3
    assert levenshtein("abc", "abc") == 0
    assert levenshtein([1, 2, 3], [1, 3]) == 1
    assert levenshtein([1, 2, 3], [4, 5, 6]) == 3


def test_cap_early_exit():
    assert levenshtein("a" * 100, "b" * 5, cap=3) == 4  # length gap alone exceeds cap
    assert levenshtein("abcdef", "abcdxx", cap=10) == 2


def test_lev_reduction():
    ref = [1, 2, 3, 4, 5]
    corrupted = [1, 9, 3, 4]  # distance 2 from ref
    assert lev_reduction(corrupted, ref, ref) == 1.0            # perfect repair
    assert lev_reduction(corrupted, corrupted, ref) == 0.0       # no-op
    assert lev_reduction(corrupted, [9, 9, 9, 9, 9], ref) < 0    # made it worse
    assert lev_reduction(ref, ref, ref) == 1.0                   # already clean
