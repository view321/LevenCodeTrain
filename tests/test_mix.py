import pytest

from levencode.data.mix import (
    NORMALIZERS,
    WeightedMixer,
    build_edit_stream,
    build_mixture,
    extract_code,
)


def fake_factory(items):
    def factory(entry, seed, buf):
        def make(epoch):
            return iter(list(items))

        return make

    return factory


def entry(name, kind, weight):
    return {"name": name, "dataset": "fake", "subset": None, "split": "train", "weight": weight, "kind": kind}


CHAT_EX = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello there"}]}
CODE_EX = {"content": "def f():\n    return 1\n" * 20}


def test_normalizers():
    assert NORMALIZERS["chat"](CHAT_EX)["messages"][-1]["role"] == "assistant"
    assert NORMALIZERS["chat"]({"messages": [{"role": "user", "content": "x"}]}) is None
    m = NORMALIZERS["magicoder"]({"problem": "p", "solution": "s"})
    assert m["messages"][0]["content"] == "p"
    assert NORMALIZERS["metamath"]({"query": "q", "response": "r"})["messages"][1]["content"] == "r"
    assert NORMALIZERS["code_raw"](CODE_EX)["text"].startswith("def f")
    assert NORMALIZERS["code_raw"]({"content": "tiny"}) is None


def test_weighted_mixer_respects_zero_weight():
    entries = [entry("a", "chat", 1.0), entry("b", "code_raw", 0.0)]
    factories = {
        "a": (lambda epoch: iter([CHAT_EX] * 50)),
        "b": (lambda epoch: iter([CODE_EX] * 50)),
    }
    mixer = WeightedMixer(entries, factories, seed=0)
    it = iter(mixer)
    for _ in range(20):
        assert next(it)["_source"] == "a"


def test_mixer_restarts_exhausted_streams():
    entries = [entry("a", "chat", 1.0)]
    calls = {"n": 0}

    def factory(epoch):
        calls["n"] += 1
        return iter([CHAT_EX] * 3)

    mixer = WeightedMixer(entries, {"a": factory}, seed=0)
    it = iter(mixer)
    for _ in range(10):
        next(it)
    assert calls["n"] >= 3  # restarted at least twice


def test_mixer_survives_transient_stream_errors():
    entries = [entry("a", "chat", 1.0)]
    calls = {"n": 0}

    def factory(epoch):
        calls["n"] += 1
        if calls["n"] == 2:  # second epoch dies mid-stream, like a network reset
            def boom():
                yield CHAT_EX
                raise ConnectionError("stream reset")

            return boom()
        return iter([CHAT_EX] * 3)

    mixer = WeightedMixer(entries, {"a": factory}, seed=0)
    mixer.RETRY_DELAYS_S = (0.0,)  # no sleeping in tests
    it = iter(mixer)
    out = [next(it) for _ in range(12)]
    assert all(o["_source"] == "a" for o in out)
    assert calls["n"] >= 4  # original + broken epoch + rebuilds after the error


def test_mixer_raises_after_persistent_failure():
    entries = [entry("a", "chat", 1.0)]

    def factory(epoch):
        raise ConnectionError("no network")

    mixer = WeightedMixer(entries, {"a": factory}, seed=0)
    mixer.RETRY_DELAYS_S = ()
    mixer.COOLDOWN_S = 0.0
    mixer.MAX_CONSECUTIVE_MISSES = 3
    with pytest.raises(RuntimeError, match="failing repeatedly"):
        next(iter(mixer))


def test_build_mixture_with_fake_streams():
    data_cfg = {
        "shuffle_buffer": 10,
        "mix": [entry("chat", "chat", 0.5), entry("code", "code_raw", 0.5)],
    }
    streams = {"chat": [CHAT_EX] * 10, "code": [CODE_EX] * 10}

    def sf(e, seed, buf):
        return lambda epoch: iter(list(streams[e["name"]]))

    it = build_mixture(data_cfg, seed=0, stream_factory=sf)
    seen = {next(it)["_source"] for _ in range(30)}
    assert seen == {"chat", "code"}


def test_extract_code():
    md = "Here you go:\n```python\ndef f():\n    return 1\n```\nshort: ```x\ny=2\n```"
    assert extract_code(md).startswith("def f")
    assert extract_code("no fences here") == "no fences here"


def test_edit_stream_extracts_solutions():
    sol = "Sure!\n```python\n" + "def g(x):\n    return x * 2\n" * 10 + "```"
    data_cfg = {
        "shuffle_buffer": 10,
        "mix": [
            entry("mc", "magicoder", 0.5),
            entry("code", "code_raw", 0.5),
            entry("chat", "chat", 0.0),
        ],
    }
    streams = {
        "mc": [{"problem": "p", "solution": sol}] * 10,
        "code": [CODE_EX] * 10,
    }

    def sf(e, seed, buf):
        return lambda epoch: iter(list(streams[e["name"]]))

    it = build_edit_stream(data_cfg, seed=0, stream_factory=sf)
    outs = [next(it) for _ in range(20)]
    assert all("text" in o for o in outs)
    mc = [o for o in outs if o["_source"] == "mc"]
    assert mc and all(o["text"].startswith("def g") for o in mc)
