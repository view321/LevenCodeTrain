"""Loom data/optimizer/trainer tests — all CPU, no network (fake bundle +
injected row streams)."""

from __future__ import annotations

import json

import pytest
import torch

from loom.data import collate_pretrain, collate_sft, pack_documents
from loom.muon import Muon, zeropower_via_newtonschulz5
from loom.train import LoomTrainer


class FakeBundle:
    vocab_size = 128
    eos_id = 1
    pad_id = 0
    bos_id = 2
    mask_id = 3
    stop_ids = (1,)

    def encode(self, text):
        return [10 + (ord(c) % 90) for c in text]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids)

    def chat_pair_ids(self, messages):
        prefix = self.encode(str(messages[:-1]))[:8]
        answer = self.encode(str(messages[-1]["content"]))[:8]
        return prefix, answer


# ---------- data ----------

def test_pack_documents_exact_rows_and_eos():
    b = FakeBundle()
    docs = ["abcdefg", "hij", "klmnopqrstuv"]
    rows = list(pack_documents(iter(docs), b, seq_len=8))
    total_tokens = sum(len(b.encode(d)) + 1 for d in docs)
    assert len(rows) == total_tokens // 8
    for r in rows:
        assert len(r["input_ids"]) == 8
    flat = [t for r in rows for t in r["input_ids"]]
    assert b.eos_id in flat  # doc separators survive packing


def test_collate_pretrain_labels_are_inputs():
    rows = [{"input_ids": [5, 6, 7]}, {"input_ids": [8, 9, 10]}]
    batch = collate_pretrain(rows)
    assert batch["input_ids"].shape == (2, 3)
    assert torch.equal(batch["input_ids"], batch["labels"])


def test_collate_sft_pads_and_masks():
    rows = [
        {"input_ids": [4, 5, 6, 7], "labels": [-100, -100, 6, 7]},
        {"input_ids": [4, 5], "labels": [-100, 5]},
    ]
    batch = collate_sft(rows, pad_id=0)
    assert batch["input_ids"].shape == (2, 4)
    assert batch["input_ids"][1, 2] == 0 and batch["labels"][1, 2] == -100
    assert batch["labels"][0, 2] == 6


# ---------- muon ----------

def test_newtonschulz_orthogonalizes():
    torch.manual_seed(0)
    g = torch.randn(16, 8)
    o = zeropower_via_newtonschulz5(g)
    s = torch.linalg.svdvals(o.float())
    # bf16 NS is approximate; singular values must cluster near 1
    assert s.min() > 0.5 and s.max() < 1.5


def test_muon_fits_regression():
    torch.manual_seed(1)
    w_true = torch.randn(8, 8)
    x = torch.randn(256, 8)
    y = x @ w_true.T
    lin = torch.nn.Linear(8, 8, bias=False)
    opt = Muon(lin.parameters(), lr=0.05, momentum=0.9)
    first = None
    for _ in range(100):
        loss = (lin(x) - y).pow(2).mean()
        if first is None:
            first = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.2


def test_muon_rejects_1d():
    p = torch.nn.Parameter(torch.randn(8))
    opt = Muon([p], lr=0.01)
    p.grad = torch.randn(8)
    with pytest.raises(ValueError):
        opt.step()


# ---------- trainer ----------

def _tiny_cfg(tmp_path, stage: str) -> dict:
    return {
        "stage": stage,
        "run": {"experiment": "loomtest", "device": "cpu", "seed": 0, "dir": str(tmp_path)},
        "model": {"max_seq_len": 16},
        "loom": {
            "d_model": 32, "n_heads": 4, "n_kv_heads": 2, "prelude_layers": 1,
            "core_layers": 2, "coda_layers": 1, "n_loops": 2, "d_ff_dense": 64,
            "n_experts": 4, "top_k": 2, "d_ff_expert": 32, "max_seq_len": 64,
            "segment_len": 4, "concept_layers": 2, "concept_heads": 2, "concept_ff": 64,
        },
        "train": {
            "micro_batch_size": 2, "grad_accum": 1, "total_steps": 3,
            "warmup_steps": 1, "log_every": 1, "sample_every": 0, "ckpt_every": 0,
            "lr_muon": 0.01, "lr_adamw": 1e-3,
        },
        "concept": {"ema_momentum": 0.99, "loss_weight": 1.0},
        "bench": {"at_stage_end": False},
    }


def _fake_rows(seq_len=16, vocab=128):
    g = torch.Generator().manual_seed(0)
    while True:
        yield {"input_ids": torch.randint(10, vocab, (seq_len,), generator=g).tolist()}


def test_trainer_pretrain_smoke_writes_webui_state(tmp_path):
    cfg = _tiny_cfg(tmp_path, "pretrain")
    tr = LoomTrainer(cfg, bundle=FakeBundle(), train_iter=_fake_rows())
    assert tr.opt_muon is not None  # the split actually engages Muon
    tr.train()
    run_dir = tmp_path / "loomtest" / "pretrain"
    state = json.loads((run_dir / "state.json").read_text())
    assert state["step"] == 3 and state["status"] == "completed"
    lines = (run_dir / "metrics.jsonl").read_text().strip().splitlines()
    assert len(lines) == 3
    row = json.loads(lines[-1])
    assert "loss" in row and "ce" in row and row["loss"] == row["loss"]  # finite
    assert (run_dir / "ckpt" / "final" / "model.pt").exists()


def test_trainer_concept_smoke_has_ema_and_concept_loss(tmp_path):
    cfg = _tiny_cfg(tmp_path, "concept")
    tr = LoomTrainer(cfg, bundle=FakeBundle(), train_iter=_fake_rows())
    assert tr.ema is not None
    tr.train()
    run_dir = tmp_path / "loomtest" / "concept"
    row = json.loads((run_dir / "metrics.jsonl").read_text().strip().splitlines()[-1])
    assert "concept" in row and row["concept"] == row["concept"]
    # EMA drifted toward the online model but is not identical
    p_ema = next(iter(tr.ema.parameters()))
    p_on = next(iter(tr.model.parameters()))
    assert p_ema.shape == p_on.shape


def test_trainer_sft_smoke(tmp_path):
    cfg = _tiny_cfg(tmp_path, "sft")

    def rows():
        while True:
            yield {"input_ids": [4, 5, 6, 7, 8], "labels": [-100, -100, 6, 7, 8]}

    tr = LoomTrainer(cfg, bundle=FakeBundle(), train_iter=rows())
    tr.train()
    state = json.loads((tmp_path / "loomtest" / "sft" / "state.json").read_text())
    assert state["status"] == "completed"
