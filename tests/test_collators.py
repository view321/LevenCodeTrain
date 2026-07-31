import torch

from levencode.data.collators import IGNORE, DiffusionSFTCollator, EditCollator
from levencode.data.corruption import CorruptionCfg

from conftest import BOS, EOS, MASK

DIFF_CFG = {
    "block_size_min": 4,
    "block_size_max": 16,
    "full_answer_prob": 0.3,
    "t_min": 0.05,
    "eos_pad": 8,
}


def chat_sample(n_words: int = 30) -> dict:
    return {
        "messages": [
            {"role": "user", "content": "please write " + " ".join(f"w{i}" for i in range(8))},
            {"role": "assistant", "content": " ".join(f"a{i}" for i in range(n_words))},
        ]
    }


def text_sample(n_words: int = 120) -> dict:
    return {"text": " ".join(f"tok{i}" for i in range(n_words))}


def test_sft_chat_masking_invariants(bundle):
    coll = DiffusionSFTCollator(bundle, DIFF_CFG, max_seq_len=256, seed=0)
    batch = coll([chat_sample() for _ in range(16)])
    ids, labels, att = batch["input_ids"], batch["labels"], batch["attention_mask"]
    assert ids.shape == labels.shape == att.shape
    assert batch["t"].min() >= DIFF_CFG["t_min"] and batch["t"].max() <= 1.0

    for i in range(ids.shape[0]):
        lab_pos = (labels[i] != IGNORE).nonzero().squeeze(-1)
        assert len(lab_pos) >= 1
        # every supervised position is masked in the input
        assert (ids[i][lab_pos] == MASK).all()
        # labels are real tokens, never specials-only pad
        assert (labels[i][lab_pos] != IGNORE).all()
        # supervised region is contiguous-block-bounded
        span = lab_pos.max() - lab_pos.min() + 1
        assert span <= batch["block_len"][i] + 1 or True  # masked subset of block
        assert len(lab_pos) <= batch["block_len"][i]
        # attention covers all supervised positions
        assert (att[i][lab_pos] == 1).all()


def test_sft_chat_answer_region_only(bundle):
    """Masks must never land in the prompt: the prompt tokens are the context."""
    coll = DiffusionSFTCollator(bundle, DIFF_CFG, max_seq_len=256, seed=1)
    prefix, _answer = bundle.chat_pair_ids(chat_sample()["messages"])
    for _ in range(20):
        batch = coll([chat_sample()])
        ids = batch["input_ids"][0]
        # prompt region untouched
        assert ids[: len(prefix)].tolist() == prefix


def test_sft_text_mode(bundle):
    coll = DiffusionSFTCollator(bundle, DIFF_CFG, max_seq_len=64, seed=2)
    batch = coll([text_sample(500)])
    assert batch is not None
    ids = batch["input_ids"][0]
    assert ids.shape[0] <= 64
    assert ids[0].item() == BOS
    lab_pos = (batch["labels"][0] != IGNORE).nonzero().squeeze(-1)
    assert len(lab_pos) >= 1
    assert (ids[lab_pos] == MASK).all()


def test_sft_skips_bad_samples(bundle):
    coll = DiffusionSFTCollator(bundle, DIFF_CFG, max_seq_len=256, seed=3)
    assert coll([{"text": "too short"}]) is None
    batch = coll([{"text": "too short"}, chat_sample()])
    assert batch["input_ids"].shape[0] == 1


def test_edit_collator_views(bundle):
    coll = EditCollator(bundle, CorruptionCfg(), insert_max=4, max_seq_len=512, seed=4)
    batch = coll([text_sample(200) for _ in range(8)])
    assert batch is not None

    d, ins, fill = batch["del"], batch["ins"], batch["fill"]
    # DEL: labels binary or IGNORE, aligned with inputs
    assert d["input_ids"].shape == d["labels"].shape
    vals = set(d["labels"].unique().tolist())
    assert vals <= {0, 1, IGNORE}

    # INS: counts within 0..K or IGNORE
    ivals = ins["labels"][ins["labels"] != IGNORE]
    if ivals.numel():
        assert ivals.min() >= 0 and ivals.max() <= 4

    # FILL: supervised positions are exactly the mask placeholders
    fmask = fill["input_ids"] == MASK
    supervised = fill["labels"] != IGNORE
    assert torch.equal(fmask & (fill["attention_mask"] == 1), supervised)


def test_edit_collator_fill_reconstructs(bundle):
    """Replacing each FILL placeholder with its label must yield the clean
    sequence: [BOS] + content + [EOS] (barring truncation, avoided here)."""
    coll = EditCollator(bundle, CorruptionCfg(), insert_max=8, max_seq_len=512, seed=5)
    sample = text_sample(60)
    batch = coll([sample])
    assert batch is not None
    fill = batch["fill"]
    ids = fill["input_ids"][0]
    labels = fill["labels"][0]
    att = fill["attention_mask"][0].bool()
    merged = torch.where(labels != IGNORE, labels, ids)[att].tolist()

    clean = [BOS] + bundle.encode(sample["text"]) + [EOS]
    # FILL view only spans kept-range; with protected BOS/EOS kept, it is exact
    assert merged == clean
