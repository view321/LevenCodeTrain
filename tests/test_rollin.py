import itertools

import torch

from levencode.data.alignment import align_interior
from levencode.data.collators import IGNORE, build_edit_views
from levencode.data.corruption import CorruptionCfg
from levencode.train.rollin import RollinBuffer, RollinCfg

from conftest import BOS, EOS, MASK, VOCAB

C_TOK = 102


class ScriptedCall:
    """Deletes nothing, inserts nothing, fills every mask with C_TOK — so both
    rollout modes deterministically produce hypotheses that differ from the
    reference (repair-mode keeps the synthetic junk; fill-mode rewrites the
    masked positions)."""

    def __call__(self, x: torch.Tensor) -> dict:
        L = x.shape[1]
        mlm = torch.zeros(1, L, VOCAB)
        mlm[:, :, C_TOK] = 10.0
        return {
            "mlm_logits": mlm,
            "del_logits": torch.full((1, L), -8.0),
            "ins_logits": torch.zeros(1, max(L - 1, 0), 9),
        }


class MockEditor:
    def __init__(self):
        self.training = True
        self._call = ScriptedCall()

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def editor_call(self):
        return self._call


def code_stream():
    text = " ".join(f"word{i}" for i in range(60))
    return itertools.cycle([{"text": text}])


def make_buffer(bundle, **overrides):
    cfg = RollinCfg.from_dict({"enabled": True, "buffer_size": 6, "max_len": 128, **overrides})
    return RollinBuffer(
        cfg, bundle, CorruptionCfg(), insert_max=8, max_seq_len=256, seed=0
    )


def test_cfg_from_dict():
    cfg = RollinCfg.from_dict({"enabled": True, "frac": 0.5, "unknown_key": 1})
    assert cfg.enabled and cfg.frac == 0.5
    assert RollinCfg.from_dict({}).enabled is False
    assert RollinCfg.from_dict(None).refresh_every == 200


def test_refresh_produces_pairs(bundle):
    buf = make_buffer(bundle)
    editor = MockEditor()
    stats = buf.refresh(editor, code_stream(), torch.device("cpu"))
    assert stats["rollin_pairs"] > 0
    assert stats["rollin_edit_mass"] > 0
    assert buf.ready()
    assert editor.training  # train mode restored after generation


def test_refresh_keeps_old_buffer_on_dry_stream(bundle):
    buf = make_buffer(bundle)
    buf.refresh(MockEditor(), code_stream(), torch.device("cpu"))
    old = list(buf.corruptions)
    stats = buf.refresh(MockEditor(), iter([]), torch.device("cpu"))
    assert stats["rollin_pairs"] == 0
    assert buf.corruptions == old  # stale buffer beats empty buffer


def test_batch_views_are_coherent(bundle):
    buf = make_buffer(bundle)
    buf.refresh(MockEditor(), code_stream(), torch.device("cpu"))
    batch = buf.batch(4)
    assert batch is not None
    for key in ("del", "ins", "fill"):
        v = batch[key]
        assert v["input_ids"].shape == v["labels"].shape
    assert set(batch["del"]["labels"].unique().tolist()) <= {0, 1, IGNORE}
    fmask = batch["fill"]["input_ids"] == MASK
    supervised = batch["fill"]["labels"] != IGNORE
    assert torch.equal(fmask & (batch["fill"]["attention_mask"] == 1), supervised)


def test_alignment_views_reconstruct_reference(bundle):
    """End-to-end label soundness for the roll-in path: alignment-derived FILL
    views must reconstruct the reference exactly, like the synthetic path."""
    ref = [BOS] + [200, 201, 202, 203, 204, 205] + [EOS]
    hyp = [BOS] + [200, 999, 202, 205] + [EOS]  # junk 999; missing 201/203/204... substituted 201
    c = align_interior(hyp, ref)
    views = build_edit_views([c], bundle, insert_max=8, max_len=256)
    f = views["fill"]
    att = f["attention_mask"][0].bool()
    merged = torch.where(f["labels"][0] != IGNORE, f["labels"][0], f["input_ids"][0])[att].tolist()
    assert merged == ref
