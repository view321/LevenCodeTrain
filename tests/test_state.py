import json

from levencode.train.state import RunDir
from levencode.util.io import read_jsonl


def test_run_lifecycle(tmp_path):
    rd = RunDir(tmp_path / "run1")
    rd.start("sft", total_steps=100, config={"lr": 1e-4})
    rd.progress(10, {"loss": 2.5, "ce": 2.4}, lr=1e-4, tok_per_sec=1000.0)
    rd.progress(20, {"loss": 2.0, "ce": 1.9}, lr=9e-5, tok_per_sec=1100.0)
    rd.finish()

    state = json.loads((tmp_path / "run1" / "state.json").read_text())
    assert state["status"] == "completed"
    assert state["stage"] == "sft"
    assert state["step"] == 20
    assert state["pct"] == 20.0
    assert state["eta_s"] is not None
    assert state["last"]["loss"] == 2.0

    rows = read_jsonl(tmp_path / "run1" / "metrics.jsonl")
    assert len(rows) == 2
    assert rows[0]["step"] == 10 and rows[1]["tok_per_sec"] == 1100.0

    cfg = json.loads((tmp_path / "run1" / "config.json").read_text())
    assert cfg["lr"] == 1e-4


def test_bench_and_samples(tmp_path):
    rd = RunDir(tmp_path / "run2")
    rd.save_bench("sft", {"gsm8k_em": 0.05, "mbpp_pass1": 0.02})
    bench = json.loads((tmp_path / "run2" / "bench" / "sft.json").read_text())
    assert bench["gsm8k_em"] == 0.05

    for step in range(12):
        rd.save_samples("sft", step, [{"prompt": "p", "output": f"o{step}"}], keep_last=5)
    gallery = json.loads((tmp_path / "run2" / "samples" / "sft.json").read_text())
    assert len(gallery) == 5
    assert gallery[-1]["step"] == 11


def test_torn_jsonl_line_tolerated(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text('{"step": 1}\n{"step": 2}\n{"ste', encoding="utf-8")
    rows = read_jsonl(p)
    assert [r["step"] for r in rows] == [1, 2]
