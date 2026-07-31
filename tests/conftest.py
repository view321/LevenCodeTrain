from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from levencode.data.tokens import TokenizerBundle  # noqa: E402

PAD, BOS, EOS, MASK, IM_END = 0, 1, 2, 3, 4
ROLE_USER, ROLE_ASSISTANT, ROLE_SYSTEM = 5, 6, 7
VOCAB = 1000


class FakeTok:
    """Deterministic word-level tokenizer with a ChatML-shaped template, small
    enough to reason about in tests. Ids 10..999 are content tokens."""

    mask_token_id = MASK
    eos_token_id = EOS
    pad_token_id = PAD
    bos_token_id = BOS
    unk_token_id = None
    all_special_ids = [PAD, BOS, EOS, MASK, IM_END, ROLE_USER, ROLE_ASSISTANT, ROLE_SYSTEM]

    def __len__(self) -> int:
        return VOCAB

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [10 + (hash(w) % (VOCAB - 10)) for w in text.split()]

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        specials = set(self.all_special_ids)
        return " ".join(
            f"t{i}" for i in ids if not (skip_special_tokens and i in specials)
        )

    def convert_tokens_to_ids(self, token: str):
        return {"<|im_end|>": IM_END}.get(token)

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        marker = {"user": "QQQ", "assistant": "AAA", "system": "SSS"}
        parts = []
        for m in messages:
            parts.append(f"{marker[m['role']]} {m['content']} END{marker[m['role']]}")
        if add_generation_prompt:
            parts.append("AAA")
        text = " ".join(parts)
        if tokenize:
            return [BOS] + self.encode(text)
        return text


@pytest.fixture
def bundle() -> TokenizerBundle:
    tok = FakeTok()
    return TokenizerBundle(
        tok=tok,
        vocab_size=VOCAB,
        mask_id=MASK,
        pad_id=PAD,
        bos_id=BOS,
        eos_id=EOS,
        stop_ids=(EOS, IM_END),
        protected=frozenset(FakeTok.all_special_ids),
        answer_suffix=" ENDAAA",
    )
