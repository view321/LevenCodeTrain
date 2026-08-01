"""Loom data: packed pretrain stream (fineweb-edu + code + math) and the SFT
chat stream, both built on levencode's WeightedMixer (streaming HF datasets
with retry/cooldown) and the shared LFM2 TokenizerBundle."""

from __future__ import annotations

from typing import Iterator

import torch

from levencode.data.mix import build_mixture


def doc_texts(stream: Iterator[dict]) -> Iterator[str]:
    """Normalize mixture samples to plain text (chat data flattens to turns)."""
    for ex in stream:
        if "text" in ex:
            yield ex["text"]
        elif "messages" in ex:
            yield "\n".join(str(m.get("content", "")) for m in ex["messages"])


def pack_documents(texts: Iterator[str], bundle, seq_len: int) -> Iterator[dict]:
    """Concatenate tokenized docs (EOS-separated) and emit fixed seq_len rows —
    every position supervised, no padding."""
    buf: list[int] = []
    for t in texts:
        buf.extend(bundle.encode(t))
        buf.append(bundle.eos_id)
        while len(buf) >= seq_len:
            yield {"input_ids": buf[:seq_len]}
            buf = buf[seq_len:]


def pretrain_stream(data_cfg: dict, bundle, seq_len: int, seed: int) -> Iterator[dict]:
    mix = build_mixture(
        {"mix": data_cfg["pretrain_mix"], "shuffle_buffer": data_cfg.get("shuffle_buffer", 10_000)},
        seed,
    )
    return pack_documents(doc_texts(mix), bundle, seq_len)


def sft_stream(data_cfg: dict, bundle, seq_len: int, seed: int) -> Iterator[dict]:
    """Chat SFT rows: loss only on answer tokens (+ EOS); prompt masked -100."""
    mix = build_mixture(
        {"mix": data_cfg["sft_mix"], "shuffle_buffer": data_cfg.get("shuffle_buffer", 10_000)},
        seed,
    )
    for ex in mix:
        msgs = ex.get("messages")
        if not msgs:
            continue
        try:
            prefix, answer = bundle.chat_pair_ids(msgs)
        except Exception:
            continue
        if not answer:
            continue
        ids = prefix + answer + [bundle.eos_id]
        if len(ids) > seq_len:
            continue
        labels = [-100] * len(prefix) + answer + [bundle.eos_id]
        yield {"input_ids": ids, "labels": labels}


def collate_pretrain(rows: list[dict]) -> dict:
    x = torch.tensor([r["input_ids"] for r in rows], dtype=torch.long)
    return {"input_ids": x, "labels": x.clone()}


def collate_sft(rows: list[dict], pad_id: int) -> dict:
    L = max(len(r["input_ids"]) for r in rows)
    ids = torch.full((len(rows), L), pad_id, dtype=torch.long)
    labels = torch.full((len(rows), L), -100, dtype=torch.long)
    for i, r in enumerate(rows):
        n = len(r["input_ids"])
        ids[i, :n] = torch.tensor(r["input_ids"], dtype=torch.long)
        labels[i, :n] = torch.tensor(r["labels"], dtype=torch.long)
    return {"input_ids": ids, "labels": labels}
