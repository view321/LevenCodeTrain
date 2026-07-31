import json

from fastapi.testclient import TestClient

from levencode.train.state import RunDir
from levencode.webui.server import create_app


def make_fixture_runs(root):
    rd = RunDir(root / "levencode" / "sft")
    rd.start("sft", total_steps=100)
    for step in range(10, 101, 10):
        rd.progress(step, {"loss": 3.0 - step / 50, "fill_loss": 2.9 - step / 50, "ce": 2.8 - step / 55},
                    lr=3e-5, tok_per_sec=18000 + step)
    rd.finish()
    rd.save_bench("sft", {"meta": {"stage": "sft"},
                          "gsm8k": {"gsm8k_em": 0.04, "n": 100, "seconds": 60.0},
                          "repair": {"repair_exact": 0.1, "repair_syntax_valid": 0.5,
                                     "repair_lev_reduction": 0.3, "n": 30, "seconds": 12.0}})
    rd.save_samples("sft", 50, [{"prompt": "hi", "output": "hello", "tok_per_sec": 120.0}])


def test_endpoints(tmp_path):
    make_fixture_runs(tmp_path)
    client = TestClient(create_app(tmp_path))

    exps = client.get("/api/experiments").json()["experiments"]
    assert len(exps) == 1
    assert exps[0]["experiment"] == "levencode"
    assert exps[0]["stages"][0]["stage"] == "sft"
    assert exps[0]["stages"][0]["state"]["status"] == "completed"
    assert exps[0]["stages"][0]["bench_names"] == ["sft"]

    rows = client.get("/api/run/levencode/sft/metrics").json()["rows"]
    assert len(rows) == 10
    assert rows[0]["step"] == 10 and "loss" in rows[0]

    st = client.get("/api/run/levencode/sft/state").json()
    assert st["total_steps"] == 100

    bench = client.get("/api/run/levencode/sft/bench").json()
    assert bench["sft"]["gsm8k"]["gsm8k_em"] == 0.04

    samples = client.get("/api/run/levencode/sft/samples").json()
    assert samples["sft"][0]["samples"][0]["output"] == "hello"

    page = client.get("/")
    assert page.status_code == 200
    assert "Levencode" in page.text

    js = client.get("/static/app.js")
    assert js.status_code == 200


def test_path_safety(tmp_path):
    make_fixture_runs(tmp_path)
    client = TestClient(create_app(tmp_path))
    assert client.get("/api/run/..%2F..%2Fetc/passwd/state").status_code in (400, 404)
    assert client.get("/api/run/levencode/nope/state").status_code == 404


def test_missing_runs_dir(tmp_path):
    client = TestClient(create_app(tmp_path / "does_not_exist"))
    assert client.get("/api/experiments").json() == {"experiments": []}
