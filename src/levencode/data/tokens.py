"""TokenizerBundle: the minimal tokenizer surface the pipeline depends on.

Everything downstream (collators, samplers, bench) talks to this bundle, not
to a transformers tokenizer directly — so tests can substitute a tiny fake."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class TokenizerBundle:
    tok: Any
    vocab_size: int
    mask_id: int
    pad_id: int
    bos_id: int | None
    eos_id: int
    stop_ids: tuple[int, ...]  # ids that terminate generation (eos, <|im_end|>, ...)
    protected: frozenset = field(default_factory=frozenset)
    # Template-specific text that closes an assistant answer (e.g. "\n[/Answer]"
    # for LFM2.5's marker-style template). Appended to answers during SFT and
    # used as a text-level stop before the finetune teaches a stop *token*.
    answer_suffix: str = ""

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        return self.tok.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def chat_prompt_ids(self, messages: list[dict]) -> list[int]:
        """Prompt ids for generation: rendered template (through the opening
        answer marker) + BOS prepended. Messages must NOT include the answer."""
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        head = [self.bos_id] if self.bos_id is not None else []
        return head + self.encode(text)

    def chat_pair_ids(self, messages: list[dict]) -> tuple[list[int], list[int]]:
        """(prefix_ids, answer_ids) for SFT. The answer is encoded separately
        from the prompt so the span boundary is exact by construction; the
        answer includes the closing marker text but not the EOS padding."""
        prefix = self.chat_prompt_ids(messages[:-1])
        answer_text = str(messages[-1].get("content", "")).strip() + self.answer_suffix
        return prefix, self.encode(answer_text)


def bundle_from_tokenizer(tok: Any, answer_suffix: str = "\n[/Answer]") -> TokenizerBundle:
    mask_id = getattr(tok, "mask_token_id", None)
    if mask_id is None:
        raise ValueError(
            "tokenizer has no mask token — the diffusion backbone requires one; "
            f"special tokens present: {getattr(tok, 'special_tokens_map', {})}"
        )
    eos_id = tok.eos_token_id
    if eos_id is None:
        raise ValueError("tokenizer has no eos token")
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos_id
    bos_id = getattr(tok, "bos_token_id", None)

    stop_ids = {eos_id}
    for t in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>"):
        tid = tok.convert_tokens_to_ids(t)
        if tid is not None and tid >= 0 and tid != getattr(tok, "unk_token_id", None):
            stop_ids.add(tid)

    protected = {mask_id, pad_id, eos_id}
    if bos_id is not None:
        protected.add(bos_id)
    protected |= stop_ids
    all_special = getattr(tok, "all_special_ids", None) or []
    protected |= set(all_special)

    return TokenizerBundle(
        tok=tok,
        vocab_size=len(tok) if hasattr(tok, "__len__") else tok.vocab_size,
        mask_id=mask_id,
        pad_id=pad_id,
        bos_id=bos_id,
        eos_id=eos_id,
        stop_ids=tuple(sorted(stop_ids)),
        protected=frozenset(protected),
        answer_suffix=answer_suffix,
    )
