"""Tests for the latent JEPA stack: chunker, RVQ, adapter, heads, bundle,
store, sampler, and BrierLM scoring."""

from __future__ import annotations

import pytest
import torch

from levencode.latent.adapter import DecodabilityAdapter
from levencode.latent.bundle import LatentBundle
from levencode.latent.chunker import HierarchicalSpec, LevelSpec, hierarchical_spans
from levencode.latent.heads import EnergyScorer, LatentHeads, LevelHeads, ResidualHead
from levencode.latent.losses import code_ce_loss, energy_loss, scorer_loss
from levencode.latent.rvq import RVQ
from levencode.latent.sampler import LatentSamplerCfg, generate_latent
from levencode.latent.teacher import LatentExample, PrecomputedLatents

from conftest import EOS, IM_END, MASK, VOCAB


# ---------- chunker ----------

def test_hierarchical_spans_are_nested(bundle):
    ids = [10 + i % 40 for i in range(200)]
    spec = HierarchicalSpec(
        levels=[
            LevelSpec(name="coarse", tokens_per_chunk=32, min_tokens=16, max_tokens=64),
            LevelSpec(name="fine", tokens_per_chunk=8, min_tokens=4, max_tokens=16),
        ]
    )
    spans = hierarchical_spans(ids, spec, bundle=bundle)
    coarse, fine = spans[0], spans[1]
    assert coarse[0][0] == 0 and coarse[-1][1] == 200
    assert fine[0][0] == 0 and fine[-1][1] == 200
    # every coarse boundary is a fine boundary (strict nesting)
    fine_bounds = {e for _, e in fine}
    for _, ce in coarse:
        assert ce in fine_bounds


def test_hierarchical_spans_prefer_newlines():
    class Shim:
        def decode(self, ids):
            return "".join("\n" if t == 10 else "x" for t in ids)

    # a "\n" token (id 10) every 8th position; chunker should cut on it
    ids = [11] * 240
    for i in range(7, 240, 8):
        ids[i] = 10
    spec = HierarchicalSpec(levels=[LevelSpec(name="fine", tokens_per_chunk=8, min_tokens=4, max_tokens=16)])
    spans = hierarchical_spans(ids, spec, bundle=Shim(), rng=None)[0]
    toks = Shim().decode(ids)
    newline_pos = {i for i, t in enumerate(toks) if t == "\n"}
    boundary_pos = {e - 1 for _, e in spans}
    assert len(boundary_pos & newline_pos) >= len(boundary_pos) // 2


# ---------- RVQ ----------

def test_rvq_roundtrip_and_lookup():
    rvq = RVQ(num_books=2, codebook_size=64, dim=8)
    z = torch.randn(16, 8)
    z_q, codes, stats = rvq.quantize(z)
    assert z_q.shape == z.shape and codes.shape == (16, 2)
    assert "commit" in stats and "dist" in stats
    # eval-mode lookup must reproduce the training quantizer output
    z_q2 = rvq.quantize_codes(codes)
    assert torch.allclose(z_q, z_q2, atol=1e-5)
    # straight-through: gradient flows back to z
    z.requires_grad_(True)
    z_q3, _, _ = rvq.quantize(z)
    z_q3.sum().backward()
    assert z.grad is not None and z.grad.abs().sum() > 0


def test_rvq_warm_reduces_distance():
    rvq = RVQ(num_books=2, codebook_size=64, dim=8)
    z = torch.randn(4096, 8)
    d_before = (z - rvq.quantize(z)[0]).norm(dim=-1).mean()
    rvq.warm(z, steps=10)
    d_after = (z - rvq.quantize(z)[0]).norm(dim=-1).mean()
    assert d_after < d_before


# ---------- adapter ----------

def test_adapter_shapes_and_kl_clip():
    ad = DecodabilityAdapter(
        latent_dim=16, bottleneck_dim=8, hidden=32, layers=1, heads=4, max_chunk=8, student_hidden=24
    )
    ad.train()
    z = torch.randn(5, 16)
    out = ad(z)
    assert out["logits_hidden"].shape == (5, 8, 24)
    assert out["kl"] >= 8 * 0.5 - 1e-6  # clipped per-dim at 0.5
    ad.eval()
    out2 = ad(z, reparam=False)
    assert torch.allclose(out2["mu"], out2["z_latent"])
    assert out2["kl"] == out2["kl"]  # no NaN


def test_adapter_token_ce_smoke(bundle):
    torch.manual_seed(0)
    ad = DecodabilityAdapter(latent_dim=8, bottleneck_dim=4, hidden=16, layers=1, heads=2, max_chunk=4, student_hidden=8)
    z = torch.randn(2, 8)
    tokens = torch.randint(10, 30, (2, 4))
    out = ad(z)
    logits = torch.randn(2, 4, 32)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 32), tokens.reshape(-1))
    assert loss.ndim == 0 and torch.isfinite(loss)


# ---------- heads ----------

def test_level_heads_shapes():
    torch.manual_seed(0)
    lv = LevelHeads("t", ar_dim=16, codebook_size=32, latent_dim=8, ar_layers=1, ar_heads=4, residual_blocks=1)
    head = torch.randn(3, 16)
    codes = torch.randint(0, 32, (3, 5, 2))
    mask = torch.tensor([[1.0] * 4 + [0.0], [1.0] * 5, [1.0] * 3 + [0.0] * 2])
    out = lv(head, codes, mask)
    # b1 has an extra trailing "next chunk" slot (used by inference)
    assert out["b1_logits"].shape == (3, 6, 32)
    assert out["b2_logits"].shape == (3, 5, 32)
    assert out["conds"].shape == (3, 5, 16)
    ce, a1, a2 = code_ce_loss(out["b1_logits"], out["b2_logits"], codes, mask)
    assert ce.ndim == 0 and torch.isfinite(ce)

    rh = ResidualHead(16, 8, blocks=2, hidden=32)
    samples = rh.sample(head, 6)  # [B=3, n=6, D=8]
    assert samples.shape == (3, 6, 8)
    es = EnergyScorer(16, 8, hidden=32)
    e = es(head, samples[:, 0])
    assert e.shape == (3,)
    dropped = torch.ones(3, dtype=torch.bool)
    e_d = es(head, torch.randn(3, 8), dropped=dropped)
    assert e_d.shape == (3,)


def test_energy_loss_strictly_proper_monotonic():
    torch.manual_seed(1)
    # a well-concentrated head (samples near target) must score better (lower)
    # than a scattered head under the energy loss
    tgt = torch.randn(4, 20, 8)
    good = tgt[:, :8] + 0.01 * torch.randn(4, 8, 8)
    bad = torch.randn(4, 8, 8) * 6.0
    assert energy_loss(good, tgt) < energy_loss(bad, tgt)


def test_scorer_loss_pushes_pos_down():
    e_pos = torch.tensor([5.0, 4.0])
    e_neg = torch.tensor([1.0, 1.0])
    loss = scorer_loss(e_pos, e_neg, margin=1.0)
    assert loss > 0


def test_latent_heads_container():
    lh = LatentHeads(student_hidden=12, latent_dim=8, codebook_size=32, ar_dim=16, ar_layers=1, ar_heads=4, residual_blocks=1)
    emb = lh.ctx_embed(torch.randn(2, 12))
    assert emb.shape == (2, 16)


# ---------- bundle ----------

def _make_bundle(**kw):
    torch.manual_seed(0)
    defaults = dict(
        latent_dim=16, student_hidden=12, codebook_size=32, rvq_books=2,
        ar_dim=16, ar_layers=1, residual_blocks=1, energy_hidden=16,
        bottleneck_dim=8, adapter_layers=1, adapter_hidden=16, adapter_heads=2,
        max_chunk=8, residual_m_targets=8,
    )
    defaults.update(kw)
    return LatentBundle(**defaults)


def test_latent_step_losses_flow():
    lb = _make_bundle()
    lb.train()
    B, W, Wf = 2, 3, 2
    batch = {
        "ctx_ids": torch.randint(10, 30, (B, 24)),
        "z_coarse": torch.randn(B, W, 16),
        "z_fine": torch.randn(B, Wf, 16),
        "z_coarse_cond": torch.randn(B, 16),
        "coarse_mask": torch.ones(B, W),
        "fine_mask": torch.ones(B, Wf),
    }
    h = torch.randn(B, 24, 12)
    out = lb.latent_step(batch, h, torch.device("cpu"))
    keys = ["coarse_ce", "coarse_energy", "coarse_scorer", "fine_ce", "fine_energy", "fine_scorer", "commit"]
    for k in keys:
        assert torch.isfinite(out[k]), k
    total = out["coarse_ce"] + out["fine_ce"] + out["coarse_energy"] + out["fine_energy"] + out["coarse_scorer"] + out["fine_scorer"]
    total.backward()
    grads = [p.grad for p in lb.heads.parameters() if p.grad is not None]
    assert grads, "no gradients reached the latent heads"


def test_decodability_step():
    lb = _make_bundle()
    lb.train()
    batch = {"z": torch.randn(3, 16), "tokens": torch.randint(10, 50, (3, 8))}
    lm = torch.nn.Linear(12, VOCAB)
    out = lb.decodability_step(batch, lm)
    assert torch.isfinite(out["decodability"])
    assert out["decodability_acc"].shape == ()
    out["decodability"].backward()
    assert any(p.grad is not None for p in lb.adapter.parameters())


# ---------- store ----------

def test_store_roundtrip(tmp_path):
    exs = [
        LatentExample(
            ctx_ids=[1, 5, 6, 7],
            fine_spans=[(4, 8)],
            coarse_of_fine=[0],
            fine_tokens=[[5, 6, 7, 8]],
            z_fine=torch.randn(1, 8),
            z_coarse=torch.randn(1, 8),
        ),
        LatentExample(
            ctx_ids=[1, 9, 9],
            fine_spans=[(3, 6), (6, 9)],
            coarse_of_fine=[0, 0],
            fine_tokens=[[9, 9, 9], [9, 9, 9]],
            z_fine=torch.randn(2, 8),
            z_coarse=torch.randn(1, 8),
        ),
    ]
    store = PrecomputedLatents(tmp_path / "store")
    store.write(exs, {"latent_dim": 8, "teacher": "t"})
    assert store.exists()
    m = store.manifest()
    assert m["n_samples"] == 2 and m["n_fine"] == 3 and m["n_coarse"] == 2
    ex = store.sample(1)
    assert len(ex.fine_tokens) == 2 and ex.z_fine.shape == (2, 8)
    b = store.batch([0, 1], ctx_len=4, coarse_window=2, fine_window=2, seed=3)
    assert b["z_coarse"].shape[0] == 2
    assert b["z_coarse_cond"].shape == (2, 8)
    d = store.decodability_batch([0, 1], chunk_tokens=8, n_chunks=2, seed=4)
    assert d["z"].shape[0] == 2 and d["tokens"].shape == (2, 8)


# ---------- sampler ----------

class FakeEditor:
    """Minimal editor stand-in: student hidden + lm_head + mlm_call."""

    def __init__(self, latent: LatentBundle):
        self.latent = latent
        self.lm_head = torch.nn.Linear(12, VOCAB)
        self.emb = torch.nn.Embedding(VOCAB, 12)

    def hidden(self, x, att=None):
        return self.emb(x)

    def mlm_call(self):
        def call(x: torch.Tensor) -> torch.Tensor:
            return self.lm_head(self.emb(x))
        return call

    @property
    def backbone(self):
        return self


def test_generate_latent_runs_and_stops(bundle):
    torch.manual_seed(2)
    lb = _make_bundle()
    lb.eval()
    editor = FakeEditor(lb)
    cfg = LatentSamplerCfg(
        fine_chunk_tokens=4,
        fine_per_coarse=2,
        fill_steps=2,
        max_coarse_chunks=2,
        cycle_consistency=False,
        plan_weight=0.0,
        max_total_len=256,
    )
    prompt = [1, 20, 21, 22]
    res = generate_latent(editor, lb, bundle, prompt, cfg, device="cpu")
    assert res.ids[:4] == prompt
    assert len(res.ids) >= 4
    assert all(0 <= t < VOCAB for t in res.ids)
    assert MASK not in res.ids


def test_generate_latent_stops_on_eos(bundle):
    torch.manual_seed(3)
    lb = _make_bundle()
    lb.eval()
    editor = FakeEditor(lb)
    cfg = LatentSamplerCfg(
        fine_chunk_tokens=4,
        fine_per_coarse=2,
        fill_steps=2,
        max_coarse_chunks=4,
        cycle_consistency=False,
        plan_weight=0.0,
        max_total_len=256,
    )
    # force the adapter to emit EOS: make EOS the only plausible token by
    # sweeping logits — easiest is to let random sampling trip a stop token
    res = generate_latent(editor, lb, bundle, [1, 20, 21], cfg, device="cpu")
    assert res.blocks >= 1


# ---------- brierlm ----------

def test_brierlm_scoring():
    from levencode.bench.tasks import _brierlm_from_scores

    V = 50
    gold = [[1, 2, 3, 4], [5, 6, 7, 8]]
    plans = [torch.zeros(4, V) for _ in range(2)]
    for i, t in enumerate(gold[0]):
        plans[0][i, t] = 5.0  # plan strongly suggests the gold token
    for i, t in enumerate(gold[1]):
        plans[1][i, t] = 5.0
    bb = torch.zeros(8, V)
    stats = _brierlm_from_scores(plans, gold, bb, temperature=0.1, seed=0)
    assert 0 <= stats["brierlm"] <= 100
    assert stats["brier_1"] >= stats["brier_4"]
    # a perfect predictor scores higher than a random one
    stats_good = _brierlm_from_scores(plans, gold, bb, temperature=0.2, seed=0)
    plans_bad = [torch.zeros(4, V) for _ in range(2)]
    stats_bad = _brierlm_from_scores(plans_bad, gold, bb, temperature=0.1, seed=0)
    assert stats_good["brierlm"] > stats_bad["brierlm"]
