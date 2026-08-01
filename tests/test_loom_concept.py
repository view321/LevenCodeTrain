"""Concept-level tests: segment pooling, causal predictor, shift helper,
loss masking, and guided generation smoke."""

from __future__ import annotations

import torch

from loom import LoomConfig, LoomLM
from loom.concept import ConceptPredictor, concept_loss, pool_segments, shift_concepts


def test_pool_segments_shapes_and_tail():
    h = torch.ones(2, 10, 8)
    pooled = pool_segments(h, segment_len=4)  # segments of 4, 4, 2 tokens
    assert pooled.shape == (2, 3, 8)
    # all-ones input: every segment mean is 1 pre-normalization, so the tail
    # segment (2 real tokens) must equal the full ones after tail-aware pooling
    assert torch.allclose(pooled[:, -1], pooled[:, 0], atol=1e-5)


def test_shift_concepts_is_causal_previous_segment():
    c = torch.arange(6, dtype=torch.float32).reshape(1, 3, 2)
    s = shift_concepts(c)
    assert torch.allclose(s[:, 0], torch.zeros(1, 2))
    assert torch.allclose(s[:, 1], c[:, 0])
    assert torch.allclose(s[:, 2], c[:, 1])


def test_concept_predictor_is_causal():
    torch.manual_seed(0)
    cfg = LoomConfig.tiny()
    pred = ConceptPredictor(cfg)
    pred.eval()
    c1 = torch.randn(1, 5, cfg.d_model)
    c2 = c1.clone()
    c2[0, 3] += 1.0  # perturb concept 3
    p1, p2 = pred(c1), pred(c2)
    # preds[:, j] is the guess for concept j from concepts < j: perturbing c_3
    # may only change predictions for concepts 4+
    assert torch.allclose(p1[:, :4], p2[:, :4], atol=1e-5)
    assert not torch.allclose(p1[:, 4:], p2[:, 4:], atol=1e-5)


def test_concept_predictor_shapes():
    cfg = LoomConfig.tiny()
    pred = ConceptPredictor(cfg)
    out = pred(torch.randn(2, 4, cfg.d_model))
    assert out.shape == (2, 5, cfg.d_model)  # S guesses + 1 next-segment plan


def test_concept_loss_masking():
    pred = torch.randn(2, 4, 8)
    tgt = torch.randn(2, 4, 8)
    full = concept_loss(pred, tgt)
    assert torch.isfinite(full) and full > 0
    none = concept_loss(pred, tgt, mask=torch.zeros(2, 4))
    assert none == 0.0
    partial = concept_loss(pred, tgt, mask=torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]]))
    assert torch.isfinite(partial)


def test_guided_generation_smoke():
    torch.manual_seed(1)
    cfg = LoomConfig.tiny()
    model = LoomLM(cfg)
    model.eval()
    # engage the modulator so guidance actually flows
    with torch.no_grad():
        model.modulator.proj.weight.add_(torch.randn_like(model.modulator.proj.weight) * 0.05)
    ids = torch.randint(0, cfg.vocab_size, (1, cfg.segment_len))
    out = model.generate(ids, max_new_tokens=cfg.segment_len + 2, temperature=0.0, use_concepts=True)
    assert out.shape[1] == ids.shape[1] + cfg.segment_len + 2


def test_pooled_concepts_helper():
    cfg = LoomConfig.tiny()
    model = LoomLM(cfg)
    model.eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 10))
    pooled = model.pooled_concepts(ids)
    assert pooled.shape == (2, (10 + cfg.segment_len - 1) // cfg.segment_len, cfg.d_model)
    # RMS-normalized targets: per-vector RMS ~ 1
    rms = pooled.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)
