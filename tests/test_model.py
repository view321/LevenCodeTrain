"""Model-surgery tests on a tiny randomly-initialized instance of the REAL
architecture (custom modeling code from the HF cache, shrunk dims). These need
the repo's Python/config files cached locally (pytest -m network first time)."""

import pytest
import torch

pytestmark = pytest.mark.network

from levencode.model.backbone import hidden_and_logits, tiny_backbone
from levencode.model.editor import LevencodeEditor
from levencode.model.jepa import JepaModule, ema_momentum


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    return tiny_backbone()


def test_tiny_backbone_forward(tiny):
    x = torch.randint(10, 1000, (2, 12))
    out = tiny(input_ids=x)
    assert out.logits.shape == (2, 12, tiny.config.vocab_size)
    h, logits = hidden_and_logits(tiny, x)
    assert h.shape == (2, 12, tiny.config.hidden_size)
    assert torch.allclose(logits, out.logits, atol=1e-4)


def test_attention_mask_padding_invariance(tiny):
    """Bidirectional pad masking: adding pad tokens must not change the
    hidden states of real positions."""
    tiny.eval()
    x = torch.randint(10, 1000, (1, 8))
    with torch.no_grad():
        h_plain, _ = hidden_and_logits(tiny, x)
        x_pad = torch.cat([x, torch.zeros(1, 4, dtype=torch.long)], dim=1)
        att = torch.cat([torch.ones(1, 8, dtype=torch.long), torch.zeros(1, 4, dtype=torch.long)], dim=1)
        h_pad, _ = hidden_and_logits(tiny, x_pad, att)
    assert torch.allclose(h_plain, h_pad[:, :8, :], atol=1e-4)


def test_editor_forward_and_shapes(tiny):
    editor = LevencodeEditor(tiny, insert_max=4)
    x = torch.randint(10, 1000, (2, 10))
    out = editor(x)
    assert out["mlm_logits"].shape == (2, 10, tiny.config.vocab_size)
    assert out["del_logits"].shape == (2, 10)
    assert out["ins_logits"].shape == (2, 9, 5)


def test_editor_backward(tiny):
    editor = LevencodeEditor(tiny, insert_max=4)
    x = torch.randint(10, 1000, (2, 10))
    out = editor(x)
    loss = out["mlm_logits"].float().mean() + out["del_logits"].float().mean() + out["ins_logits"].float().mean()
    loss.backward()
    grads = [p.grad for p in editor.heads.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    emb_grad = editor.backbone.get_input_embeddings().weight.grad
    assert emb_grad is not None


def test_editor_save_load_roundtrip(tiny, tmp_path):
    editor = LevencodeEditor(tiny, insert_max=4)
    x = torch.randint(10, 1000, (1, 8))
    editor.eval()
    with torch.no_grad():
        before = editor(x)
    editor.save(tmp_path / "ckpt")
    assert (tmp_path / "ckpt" / "backbone" / "modeling_lfm2_bidirectional.py").exists()

    loaded = LevencodeEditor.load(tmp_path / "ckpt")
    loaded.eval()
    with torch.no_grad():
        after = loaded(x)
    for k in ("mlm_logits", "del_logits", "ins_logits"):
        assert torch.allclose(before[k].float(), after[k].float(), atol=1e-4), k


def test_jepa_module(tiny):
    jepa = JepaModule(tiny.lfm2, tiny.config.hidden_size, predictor_layers=1, predictor_heads=4)
    x = torch.randint(10, 1000, (2, 8))
    with torch.no_grad():
        tgt = jepa.targets(x)
    assert tgt.shape == (2, 8, tiny.config.hidden_size)
    h, _ = hidden_and_logits(tiny, x)
    pred = jepa.predict(h)
    assert pred.shape == tgt.shape
    assert pred.requires_grad

    # EMA pulls the target toward the online encoder
    with torch.no_grad():
        for p in tiny.lfm2.parameters():
            p.add_(1.0)
    p_t0 = next(jepa.target.parameters()).clone()
    jepa.ema_update(tiny.lfm2, momentum=0.5)
    p_t1 = next(jepa.target.parameters())
    assert not torch.allclose(p_t0, p_t1)
    assert not any(p.requires_grad for p in jepa.target.parameters())


def test_ema_momentum_schedule():
    assert ema_momentum(0, 100, 0.99, 1.0) == 0.99
    assert ema_momentum(100, 100, 0.99, 1.0) == 1.0
    assert 0.99 < ema_momentum(50, 100, 0.99, 1.0) < 1.0
