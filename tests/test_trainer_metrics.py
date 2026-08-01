"""MetricAcc: per-key mean semantics that fix the edit-stage divisor bug."""

from levencode.train.trainer import MetricAcc


def test_metric_acc_divides_each_key_by_its_own_count():
    acc = MetricAcc()
    # Simulate one log window of an edit-stage step mix (grad_accum=8,
    # retain_sft_frac=0.5): `ce` only comes from the 4 SFT micro-batches,
    # `del_loss` only from the 4 edit micro-batches, `loss` from every step.
    for v in (1.0, 1.2, 0.8, 1.0):
        acc.add("ce", v)
    for v in (0.5, 0.7, 0.6, 0.6):
        acc.add("del_loss", v)
    for v in (2.0, 2.2):
        acc.add("loss", v)
    means = acc.means()
    assert means["ce"] == 1.0        # 4.0 / 4 adds — not / 8 micro-batches
    assert means["del_loss"] == 0.6
    assert means["loss"] == 2.1


def test_metric_acc_conditional_keys():
    acc = MetricAcc()
    acc.add("del_recall", 1.0)  # only added when a batch contains junk
    means = acc.means()
    assert means["del_recall"] == 1.0
    assert MetricAcc().means() == {}
