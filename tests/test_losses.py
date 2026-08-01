import torch

from levencode.train.losses import (
    IGNORE,
    delete_loss,
    diffusion_fill_loss,
    diffusion_fill_loss_sparse,
    insert_loss,
    jepa_loss,
    masked_ce_loss,
    masked_ce_loss_sparse,
)


def make_batch():
    B, L, V = 2, 6, 11
    logits = torch.randn(B, L, V)
    labels = torch.full((B, L), IGNORE)
    labels[0, 2] = 3
    labels[0, 3] = 4
    labels[1, 1] = 5
    t = torch.tensor([0.5, 1.0])
    block_len = torch.tensor([4, 2])
    return logits, labels, t, block_len


def test_diffusion_loss_inverse_t_weighting():
    logits, labels, _, block_len = make_batch()
    l_half, _ = diffusion_fill_loss(logits, labels, torch.tensor([0.5, 0.5]), block_len)
    l_one, _ = diffusion_fill_loss(logits, labels, torch.tensor([1.0, 1.0]), block_len)
    assert torch.isclose(l_half, 2 * l_one, rtol=1e-5)


def test_diffusion_loss_ignores_unlabeled():
    logits, labels, t, block_len = make_batch()
    l1, _ = diffusion_fill_loss(logits, labels, t, block_len)
    logits2 = logits.clone()
    logits2[0, 0, :] = 99.0  # position with IGNORE label
    l2, _ = diffusion_fill_loss(logits2, labels, t, block_len)
    assert torch.isclose(l1, l2)


def test_diffusion_loss_decreases_with_correct_logits():
    logits, labels, t, block_len = make_batch()
    good = logits.clone()
    for b in range(labels.shape[0]):
        for pos in range(labels.shape[1]):
            if labels[b, pos] != IGNORE:
                good[b, pos, labels[b, pos]] = 20.0
    l_rand, _ = diffusion_fill_loss(logits, labels, t, block_len)
    l_good, _ = diffusion_fill_loss(good, labels, t, block_len)
    assert l_good < l_rand


def test_masked_ce_acc():
    logits = torch.zeros(1, 3, 5)
    logits[0, 0, 2] = 10.0
    logits[0, 1, 3] = 10.0
    labels = torch.tensor([[2, 4, IGNORE]])
    loss, acc = masked_ce_loss(logits, labels)
    assert 0.4 < acc.item() < 0.6  # 1 of 2 supervised correct
    assert loss.item() > 0


def test_delete_loss_masks_pads():
    logits = torch.tensor([[8.0, -8.0, 0.0]])
    labels = torch.tensor([[1, 0, IGNORE]])
    loss, acc = delete_loss(logits, labels)
    assert acc.item() == 1.0
    assert loss.item() < 0.01
    # all-IGNORE batch: zero loss, no NaN
    loss0, _ = delete_loss(logits, torch.full((1, 3), IGNORE))
    assert loss0.item() == 0.0


def test_delete_loss_pos_weight():
    """Upweighting positives must raise the penalty for a missed junk token."""
    logits = torch.tensor([[-6.0, -6.0, -6.0]])  # predicts "keep" everywhere
    labels = torch.tensor([[1, 0, 0]])           # first token is junk (missed)
    base, _ = delete_loss(logits, labels)
    weighted, _ = delete_loss(logits, labels, pos_weight=4.0)
    assert weighted.item() > base.item() * 2


def test_insert_loss_zero_weight():
    B, G, C = 1, 3, 5
    logits = torch.zeros(B, G, C)
    logits[..., 3] = 2.0  # head leans toward count-3, so per-class CE differs
    labels_nonzero = torch.tensor([[1, 2, 3]])
    a, _ = insert_loss(logits, labels_nonzero)
    b, _ = insert_loss(logits, labels_nonzero, zero_weight=0.5)
    assert torch.isclose(a, b)  # weight on class 0 is irrelevant if no 0 labels

    labels_mixed = torch.tensor([[0, 0, 3]])
    full, _ = insert_loss(logits, labels_mixed)
    down, _ = insert_loss(logits, labels_mixed, zero_weight=0.25)
    # downweighting the (badly-predicted) zero class must lower the weighted mean
    assert torch.isfinite(down) and down.item() < full.item()


def test_insert_loss_slices_labels():
    B, L, K1 = 1, 5, 4
    ins_logits = torch.zeros(B, L - 1, K1)
    ins_logits[0, 0, 2] = 10.0
    labels = torch.tensor([[2, 3, 0, IGNORE, IGNORE]])  # collator width L, logits width L-1
    loss, acc = insert_loss(ins_logits, labels)
    assert labels.shape[1] == L
    assert 0.5 < acc.item() < 1.0  # gaps 0 and 2 right, gap 1 wrong
    assert loss.item() > 0


def test_sparse_losses_match_dense():
    """The memory-efficient gathered-position losses must equal the dense ones
    exactly — they are what the trainer actually uses."""
    logits, labels, t, block_len = make_batch()
    dense, dense_ce = diffusion_fill_loss(logits, labels, t, block_len)

    b_idx, pos = (labels != IGNORE).nonzero(as_tuple=True)
    logits_sel = logits[b_idx, pos]
    labels_sel = labels[b_idx, pos]
    sparse, sparse_ce = diffusion_fill_loss_sparse(logits_sel, labels_sel, b_idx, t, block_len)
    assert torch.isclose(dense, sparse, rtol=1e-5)
    assert torch.isclose(dense_ce, sparse_ce, rtol=1e-5)

    d_ml, d_acc = masked_ce_loss(logits, labels)
    s_ml, s_acc = masked_ce_loss_sparse(logits_sel, labels_sel)
    assert torch.isclose(d_ml, s_ml, rtol=1e-5)
    assert torch.isclose(d_acc, s_acc, rtol=1e-5)


def test_jepa_loss():
    B, L, H = 2, 4, 8
    pred = torch.randn(B, L, H)
    target = torch.randn(B, L, H)
    pos = torch.zeros(B, L, dtype=torch.bool)
    assert jepa_loss(pred, target, pos).item() == 0.0
    pos[0, 1] = True
    assert jepa_loss(pred, target, pos).item() > 0.0
    # scale-invariance of the layer-normed target: doubling target changes little
    l1 = jepa_loss(pred, target, pos).item()
    l2 = jepa_loss(pred, target * 2, pos).item()
    assert abs(l1 - l2) < 1e-5
