"""LoomLM architecture tests: shapes, causality, loop KV-cache equivalence,
MoE routing, zero-init concept no-op, optimizer partition, checkpoint IO."""

from __future__ import annotations

import pytest
import torch

from loom import LoomConfig, LoomLM, param_report
from loom.layers import MoELayer


@pytest.fixture()
def cfg() -> LoomConfig:
    return LoomConfig.tiny()


@pytest.fixture()
def model(cfg) -> LoomLM:
    torch.manual_seed(0)
    m = LoomLM(cfg)
    m.eval()
    return m


def test_forward_shapes_and_loss(model, cfg):
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model(ids, labels=ids)
    assert out["logits"].shape == (2, 16, cfg.vocab_size)
    assert out["hidden"].shape == (2, 16, cfg.d_model)
    for k in ("loss", "ce", "aux_lb", "aux_z"):
        assert torch.isfinite(out[k]), k
    out["loss"].backward()
    expert_grads = [p.grad for p in model.core[0].moe.experts.parameters() if p.grad is not None]
    assert expert_grads and any(g.abs().sum() > 0 for g in expert_grads)
    assert model.adapter.weight.grad is not None


def test_causality(model, cfg):
    torch.manual_seed(1)
    ids1 = torch.randint(0, cfg.vocab_size, (1, 12))
    ids2 = ids1.clone()
    ids2[0, -1] = (ids2[0, -1] + 1) % cfg.vocab_size
    l1 = model(ids1)["logits"]
    l2 = model(ids2)["logits"]
    assert torch.allclose(l1[:, :-1], l2[:, :-1], atol=1e-5)
    assert not torch.allclose(l1[:, -1], l2[:, -1], atol=1e-5)


def test_kv_cache_matches_full_forward(model, cfg):
    """The loop-cache bug magnet: every (loop, layer) pair owns a slot and the
    slot order must be identical between prefill and decode."""
    torch.manual_seed(2)
    ids = torch.randint(0, cfg.vocab_size, (1, 10))
    full = model(ids)["logits"]
    past = None
    step_logits = []
    for t in range(ids.shape[1]):
        out = model(ids[:, t : t + 1], past=past, use_cache=True)
        past = out["past"]
        step_logits.append(out["logits"][:, 0])
    inc = torch.stack(step_logits, dim=1)
    assert len(past) == model.n_cache_slots
    assert torch.allclose(full, inc, atol=1e-4), (full - inc).abs().max()


def test_kv_cache_chunked_prefill(model, cfg):
    """Prefix prefill + multi-token chunk (the offset-mask attention path)."""
    torch.manual_seed(3)
    ids = torch.randint(0, cfg.vocab_size, (1, 12))
    full = model(ids)["logits"]
    out1 = model(ids[:, :5], use_cache=True)
    out2 = model(ids[:, 5:], past=out1["past"], use_cache=True)
    chunked = torch.cat([out1["logits"], out2["logits"]], dim=1)
    assert torch.allclose(full, chunked, atol=1e-4)


def test_zero_init_concepts_are_noop(model, cfg):
    """Phase-1/phase-2 contract: with the modulator at zero-init, passing any
    concept tensor must not change the logits at all."""
    torch.manual_seed(4)
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    S = 16 // cfg.segment_len
    concepts = torch.randn(2, S, cfg.d_model)
    base = model(ids)["logits"]
    guided = model(ids, concepts=concepts)["logits"]
    assert torch.allclose(base, guided, atol=1e-6)


def test_nonzero_modulator_engages(model, cfg):
    torch.manual_seed(5)
    with torch.no_grad():
        model.modulator.proj.weight.add_(torch.randn_like(model.modulator.proj.weight) * 0.1)
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    concepts = torch.randn(1, 2, cfg.d_model)
    assert not torch.allclose(model(ids)["logits"], model(ids, concepts=concepts)["logits"], atol=1e-5)


def test_moe_layer_routing():
    torch.manual_seed(6)
    moe = MoELayer(d_model=16, n_experts=4, top_k=2, d_ff_expert=8, shared=True)
    x = torch.randn(3, 7, 16)
    out, aux = moe(x)
    assert out.shape == x.shape
    assert torch.isfinite(aux["lb"]) and torch.isfinite(aux["z"])
    # load-balance loss is E * sum f_e P_e; for a uniform router it approaches
    # 1.0 — a collapsed router (all mass on one expert) approaches E
    assert 0.5 < aux["lb"].item() < 4.0


def test_param_groups_partition(model):
    groups = model.param_groups()
    muon_ids = {id(p) for p in groups["muon"]}
    adamw_ids = {id(p) for p in groups["adamw"]}
    assert not (muon_ids & adamw_ids)
    all_ids = {id(p) for p in model.parameters() if p.requires_grad}
    assert muon_ids | adamw_ids == all_ids
    assert all(p.ndim >= 2 for p in groups["muon"])
    # tied embedding must land in adamw, not muon
    assert id(model.embed.weight) in adamw_ids
    assert all(id(model.core[0].moe.router.weight) != i for i in muon_ids)


def test_tied_embeddings(model):
    assert model.lm_head.weight is model.embed.weight


def test_generate_and_checkpoint_roundtrip(model, cfg, tmp_path):
    torch.manual_seed(7)
    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    out = model.generate(ids, max_new_tokens=6, temperature=0.0)
    assert out.shape[1] == 10
    model.save(tmp_path / "ck")
    re = LoomLM.load(tmp_path / "ck")
    re.eval()
    assert torch.allclose(model(ids)["logits"], re(ids)["logits"], atol=1e-6)


def test_grad_checkpoint_matches(model, cfg):
    torch.manual_seed(8)
    model.train()
    ids = torch.randint(0, cfg.vocab_size, (2, 12))
    out_a = model(ids, labels=ids)
    out_b = model(ids, labels=ids, grad_checkpoint=True)
    assert torch.allclose(out_a["loss"], out_b["loss"], atol=1e-5)


def test_autocast_bf16_forward_backward(model, cfg):
    """Regression: the 5090 run died on index_add_ dtype mismatch — experts
    emit bf16 under autocast while the fp32 residual stream owns the buffer."""
    torch.manual_seed(9)
    model.train()
    ids = torch.randint(0, cfg.vocab_size, (2, 12))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(ids, labels=ids)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()


def test_param_report_keys(model):
    rep = param_report(model)
    assert rep["total"] > rep["active_compute_per_token(excl. head)"] > 0
    assert rep["effective_depth"] == model.cfg.effective_depth
