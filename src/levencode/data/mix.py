"""Weighted streaming mixture over HF datasets, normalized to a common shape:
{"messages": [...]} for instruct data or {"text": "..."} for raw code.

The mixer works over plain Python iterators so tests can inject fakes and no
datasets-library interleave quirks leak into training."""

from __future__ import annotations

import random
import re
import time
from typing import Any, Callable, Iterator

Normalizer = Callable[[dict], dict | None]

_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)


def _norm_chat(ex: dict) -> dict | None:
    msgs = ex.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 2:
        return None
    if msgs[-1].get("role") != "assistant" or not msgs[-1].get("content"):
        return None
    return {"messages": msgs}


def _norm_magicoder(ex: dict) -> dict | None:
    p, s = ex.get("problem"), ex.get("solution")
    if not p or not s:
        return None
    return {"messages": [{"role": "user", "content": p}, {"role": "assistant", "content": s}]}


def _norm_metamath(ex: dict) -> dict | None:
    q, r = ex.get("query"), ex.get("response")
    if not q or not r:
        return None
    return {"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": r}]}


def _norm_code_raw(ex: dict) -> dict | None:
    txt = ex.get("content") or ex.get("text") or ""
    if not (200 <= len(txt) <= 50_000):
        return None
    return {"text": txt}


def _norm_text_raw(ex: dict) -> dict | None:
    """Plain-text pretrain documents (fineweb-edu, finemath, ...)."""
    txt = ex.get("text") or ex.get("content") or ""
    if len(txt) < 200:
        return None
    return {"text": txt[:200_000]}


NORMALIZERS: dict[str, Normalizer] = {
    "chat": _norm_chat,
    "magicoder": _norm_magicoder,
    "metamath": _norm_metamath,
    "code_raw": _norm_code_raw,
    "text_raw": _norm_text_raw,
}


def extract_code(text: str) -> str:
    """Largest fenced code block if present, else the raw text."""
    blocks = _FENCE_RE.findall(text or "")
    if blocks:
        return max(blocks, key=len)
    return text or ""


def hf_stream_factory(entry: dict, seed: int, shuffle_buffer: int) -> Callable[[int], Iterator[dict]]:
    def make(epoch: int) -> Iterator[dict]:
        from datasets import load_dataset

        ds = load_dataset(
            entry["dataset"],
            entry.get("subset") or None,
            split=entry.get("split", "train"),
            streaming=True,
        )
        ds = ds.shuffle(seed=seed + epoch, buffer_size=shuffle_buffer)
        return iter(ds)

    return make


class WeightedMixer:
    """Samples a stream per step according to weights; exhausted streams restart
    with a bumped epoch seed, so the mixture never runs dry.

    Streaming HF datasets fail transiently (network hiccups, hub 5xx); a
    multi-hour run must not die for that. A failing stream is rebuilt with
    backoff, then benched for COOLDOWN_S while the rest of the mixture carries
    on; only when every draw keeps failing does the mixer raise."""

    RETRY_DELAYS_S = (1.0, 3.0, 10.0, 30.0)  # in-call backoff before benching
    COOLDOWN_S = 120.0                       # how long a failed stream sits out
    MAX_CONSECUTIVE_MISSES = 20              # then the mixture raises

    def __init__(self, entries: list[dict], factories: dict[str, Callable[[int], Iterator[dict]]], seed: int):
        self.entries = entries
        self.factories = factories
        self.rng = random.Random(seed)
        self.names = [e["name"] for e in entries]
        self.weights = [float(e["weight"]) for e in entries]
        self.kinds = {e["name"]: e["kind"] for e in entries}
        self.iters: dict[str, Iterator[dict]] = {}
        self.epochs: dict[str, int] = {n: 0 for n in self.names}

    def _rebuild(self, name: str) -> None:
        # Bump the epoch even on error rebuilds: the reshuffle avoids
        # re-walking the identical stream prefix after a mid-stream failure.
        self.epochs[name] += 1
        self.iters.pop(name, None)

    def _next_from(self, name: str) -> dict | None:
        exhausted = 0
        errors = 0
        while True:
            try:
                if name not in self.iters:
                    self.iters[name] = self.factories[name](self.epochs[name])
                return next(self.iters[name])
            except StopIteration:  # normal exhaustion: new epoch, reshuffled
                exhausted += 1
                self._rebuild(name)
                if exhausted > 2:  # restarts immediately empty: dead stream
                    return None
            except Exception as e:
                self._rebuild(name)
                if errors >= len(self.RETRY_DELAYS_S):
                    print(
                        f"[mix] stream {name!r} still failing after {errors} retries "
                        f"({type(e).__name__}: {e}); benching it for {self.COOLDOWN_S:.0f}s",
                        flush=True,
                    )
                    return None
                delay = self.RETRY_DELAYS_S[errors]
                errors += 1
                print(
                    f"[mix] stream {name!r} error ({type(e).__name__}: {e}); "
                    f"rebuilding in {delay:.0f}s",
                    flush=True,
                )
                time.sleep(delay)

    def __iter__(self) -> Iterator[dict]:
        cooldown: dict[str, float] = {}
        misses = 0
        while True:
            now = time.monotonic()
            avail = [n for n in self.names if cooldown.get(n, 0.0) <= now]
            avail_w = [self.weights[self.names.index(n)] for n in avail]
            if not avail or (sum(avail_w) <= 0 and cooldown):
                time.sleep(max(min(cooldown.values()) - now, 1.0))
                continue
            name = self.rng.choices(avail, weights=avail_w, k=1)[0]
            ex = self._next_from(name)
            if ex is None:
                cooldown[name] = time.monotonic() + self.COOLDOWN_S
                misses += 1
                if misses >= self.MAX_CONSECUTIVE_MISSES:
                    raise RuntimeError(
                        f"data mixture: streams failing repeatedly (last: {name!r}) — "
                        "check network and dataset availability"
                    )
                continue
            misses = 0
            cooldown.pop(name, None)
            norm = NORMALIZERS[self.kinds[name]](ex)
            if norm is None:
                continue
            norm["_source"] = name
            yield norm


def build_mixture(data_cfg: dict, seed: int, stream_factory=hf_stream_factory) -> Iterator[dict]:
    entries = data_cfg["mix"]
    buf = int(data_cfg.get("shuffle_buffer", 10_000))
    factories = {e["name"]: stream_factory(e, seed, buf) for e in entries}
    return iter(WeightedMixer(entries, factories, seed))


def build_edit_stream(data_cfg: dict, seed: int, stream_factory=hf_stream_factory) -> Iterator[dict]:
    """Code-only stream for edit-head training, normalized to {"text": code}."""
    code_kinds = {"code_raw", "magicoder"}
    entries = [e for e in data_cfg["mix"] if e["kind"] in code_kinds]
    if not entries:
        raise ValueError("data.mix has no code entries for the edit stream")
    buf = int(data_cfg.get("shuffle_buffer", 10_000))
    factories = {e["name"]: stream_factory(e, seed + 7919, buf) for e in entries}
    mixer = WeightedMixer(entries, factories, seed + 7919)

    def gen() -> Iterator[dict]:
        for sample in mixer:
            if "text" in sample:
                yield {"text": sample["text"], "_source": sample["_source"]}
            else:
                code = extract_code(sample["messages"][-1]["content"])
                if len(code) >= 80:
                    yield {"text": code, "_source": sample["_source"]}

    return gen()
