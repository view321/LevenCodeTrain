"""Loom bench tests: BrierLM clamping, logprob scoring, offline task smoke."""

from __future__ import annotations

import torch

from loom import LoomConfig, LoomLM
from loom.bench import LoomBenchCtx, brier_ngram_stats, mean_logprob, run_loom_bench

from test_loom_train import FakeBundle


def test_brier_positive_for_sharp_correct_predictor():
    V = 50
    gold = [3, 7, 11, 19, 23, 29, 31, 37]
    logits = torch.zeros(len(gold), V)
    for i, t in enumerate(gold):
        logits[i, t] = 12.0
    stats = brier_ngram_stats(logits, gold, temperature=0.7, seed=0)
    assert stats["brier_1"] > 0.9
    assert stats["brierlm"] > 80.0


def test_brier_clamps_confidently_wrong_to_zero():
    V = 50
    gold = [3, 7, 11, 19]
    logits = torch.zeros(len(gold), V)
    for i, t in enumerate(gold):
        logits[i, (t + 1) % V] = 12.0  # deterministic and WRONG
    stats = brier_ngram_stats(logits, gold, temperature=0.7, seed=0)
    assert stats["brier_1"] < 0  # the honest signal survives per-component
    assert stats["brierlm"] == 0.0  # composite cannot sign-flip positive


def test_mean_logprob_prefers_model_argmax():
    torch.manual_seed(0)
    cfg = LoomConfig.tiny()
    model = LoomLM(cfg)
    model.eval()
    ctx = LoomBenchCtx(
        model=model, bundle=FakeBundle(), cfg={}, device=torch.device("cpu"),
        chat_format=False, use_concepts=False,
    )
    prefix = [5, 6, 7, 8]
    with torch.no_grad():
        nxt = int(model(torch.tensor([prefix]))["logits"][0, -1].argmax())
    lp_best = mean_logprob(ctx, prefix, [nxt])
    lp_other = mean_logprob(ctx, prefix, [(nxt + 1) % cfg.vocab_size])
    assert lp_best > lp_other


def test_run_loom_bench_offline_subset():
    torch.manual_seed(1)
    cfg = LoomConfig.tiny()
    model = LoomLM(cfg)
    model.eval()
    results = run_loom_bench(
        model, FakeBundle(), {"stage": "pretrain", "bench": {}}, torch.device("cpu"),
        only=["speed"],
    )
    assert "meta" in results and results["meta"]["use_concepts"] is False
    assert "gen_tok_per_sec" in results["speed"]
    assert results["speed"]["gen_tok_per_sec"] > 0
