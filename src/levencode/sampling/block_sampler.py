"""Canvas-by-canvas (block) generation for masked-diffusion models.

Append a block of masks after the committed context, iteratively fill the most
confident positions, then open the next block — until a stop token appears.
`model_call` is any callable(input_ids [1, L]) -> logits [1, L, V], keeping the
sampler independent of the concrete model class (and mockable in tests)."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from ..data.tokens import TokenizerBundle


@dataclass
class BlockSamplerCfg:
    block_size: int = 64
    steps_per_block: int = 16
    max_blocks: int = 8
    temperature: float = 0.0
    top_p: float = 0.9
    # Text-level stops (e.g. "[/Answer]") — needed because the pretrained
    # template closes answers with marker *text*; stage-1 SFT additionally
    # teaches the <|im_end|> stop token.
    stop_texts: tuple = ()

    @classmethod
    def from_dict(cls, d: dict) -> "BlockSamplerCfg":
        out = cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})
        out.stop_texts = tuple(out.stop_texts or ())
        return out


@dataclass
class GenResult:
    ids: list[int]          # full sequence including prompt
    new_ids: list[int]      # generated part only (stop token excluded)
    blocks: int
    forwards: int
    seconds: float
    text: str = ""          # decoded generation, truncated at the first stop text

    @property
    def tokens_per_sec(self) -> float:
        return len(self.new_ids) / max(self.seconds, 1e-9)


def pick_token(
    logits: torch.Tensor, temperature: float, top_p: float, generator: torch.Generator | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-position token choice + its probability (confidence). logits [N, V]."""
    if temperature <= 0.0:
        probs = torch.softmax(logits.float(), dim=-1)
        conf, tok = probs.max(dim=-1)
        return tok, conf
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    if 0.0 < top_p < 1.0:
        sorted_p, sorted_idx = probs.sort(dim=-1, descending=True)
        cum = sorted_p.cumsum(dim=-1)
        cut = cum - sorted_p >= top_p  # positions strictly beyond the nucleus
        sorted_p = sorted_p.masked_fill(cut, 0.0)
        sorted_p = sorted_p / sorted_p.sum(dim=-1, keepdim=True)
        choice = torch.multinomial(sorted_p, 1, generator=generator).squeeze(-1)
        tok = sorted_idx.gather(-1, choice.unsqueeze(-1)).squeeze(-1)
    else:
        tok = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
    conf = probs.gather(-1, tok.unsqueeze(-1)).squeeze(-1)
    return tok, conf


@torch.no_grad()
def generate(
    model_call: Callable[[torch.Tensor], torch.Tensor],
    bundle: TokenizerBundle,
    prompt_ids: Sequence[int],
    cfg: BlockSamplerCfg,
    device: torch.device | str = "cpu",
    seed: int | None = None,
    max_total_len: int = 4096,
) -> GenResult:
    gen = None
    if seed is not None and cfg.temperature > 0:
        gen = torch.Generator(device="cpu").manual_seed(seed)

    ids = list(prompt_ids)
    prompt_len = len(ids)
    stop_set = set(bundle.stop_ids)
    forwards = 0
    t0 = time.perf_counter()
    blocks_done = 0

    for _ in range(cfg.max_blocks):
        if len(ids) + cfg.block_size > max_total_len:
            break
        base = len(ids)
        canvas = ids + [bundle.mask_id] * cfg.block_size
        x = torch.tensor([canvas], dtype=torch.long, device=device)
        masked = list(range(base, base + cfg.block_size))

        for step in range(cfg.steps_per_block):
            if not masked:
                break
            logits = model_call(x)
            forwards += 1
            pos = torch.tensor(masked, dtype=torch.long, device=x.device)
            tok, conf = pick_token(
                logits[0, pos, :], cfg.temperature, cfg.top_p, gen
            )
            k = math.ceil(len(masked) / (cfg.steps_per_block - step))
            k = min(k, len(masked))
            commit = conf.argsort(descending=True)[:k]
            x[0, pos[commit]] = tok[commit].to(x.dtype)
            committed = set(pos[commit].tolist())
            masked = [p for p in masked if p not in committed]

        block = x[0, base:].tolist()
        blocks_done += 1
        stop_at = next((i for i, t in enumerate(block) if t in stop_set), None)
        if stop_at is not None:
            ids = x[0, :base].tolist() + block[: stop_at + 1]
            break
        ids = x[0].tolist()
        if cfg.stop_texts:
            txt_so_far = bundle.decode(ids[prompt_len:])
            if any(st in txt_so_far for st in cfg.stop_texts):
                break

    seconds = time.perf_counter() - t0
    new_ids = [t for t in ids[prompt_len:] if t not in stop_set]
    text = bundle.decode(new_ids)
    for st in cfg.stop_texts:
        cut = text.find(st)
        if cut >= 0:
            text = text[:cut]
    return GenResult(
        ids=ids, new_ids=new_ids, blocks=blocks_done, forwards=forwards, seconds=seconds, text=text
    )
