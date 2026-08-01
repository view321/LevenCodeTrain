"""Latent-guided generation: think in chunk latents, speak in tokens.

Pipeline per coarse chunk (the mixture of JEPAs at work):
1. Encode the committed context with the student encoder -> pooled ctx embed.
2. AR-sample the coarse *plan codes* (discrete anchor — cheap temperature,
   beam/lookahead searchable), then draw residual candidates from the energy
   head and pick with the classifier-free guided score
   (1+w)*E(cond, z) - w*E(null, z)  ->  z_coarse = z_q + r.
3. Per fine chunk (fixed K_f tokens at inference): fine codes conditioned on
   the quantized coarse latent, CFG-scored residual -> z_fine.
4. Decode the fine latent through the decodability adapter -> per-position
   plan logits; the block-diffusion filler then fills the 8 mask positions
   with backbone logits *plus* the plan prior (context-aware decoder — the
   thing LCM/CALM lacked).
5. Cycle-consistency rejection: re-encode the committed chunk with the frozen
   teacher; if the roundtrip drifted from the predicted latent, resample the
   chunk (bounded retries) instead of propagating the error.

The teacher is loaded lazily and only when `cycle_consistency` is on; it is
never needed for training."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from ..data.tokens import TokenizerBundle
from ..sampling.block_sampler import GenResult, pick_token
from .bundle import LatentBundle
from .teacher import TEACHER_REPO


@dataclass
class LatentSamplerCfg:
    fine_chunk_tokens: int = 8
    fine_per_coarse: int = 4
    plan_temperature: float = 0.8
    plan_top_p: float = 0.9
    residual_candidates: int = 8
    cfg_weight: float = 1.0
    plan_weight: float = 1.0
    fill_steps: int = 8
    temperature: float = 0.0
    top_p: float = 0.9
    max_coarse_chunks: int = 6
    code_history: int = 4  # AR prior context in coarse chunks; training never
    # sees windows longer than latent.coarse_window, so cap the history there
    # instead of conditioning on untrained sequence lengths
    ctx_len: int = 192  # student-context pooling window — keep = latent.ctx_len:
    # training pools the last ctx_len tokens (store.batch), so pooling the full
    # committed sequence here would shift the ctx-embed distribution
    cycle_consistency: bool = True
    cycle_threshold: float = 0.75
    cycle_retries: int = 2
    teacher_repo: str = TEACHER_REPO
    max_total_len: int = 4096
    stop_texts: tuple = ()

    @classmethod
    def from_dict(cls, d: dict) -> "LatentSamplerCfg":
        out = cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})
        out.stop_texts = tuple(out.stop_texts or ())
        return out


class _CycleTeacher:
    """Lazy singleton for cycle-consistency re-encoding (inference only)."""

    _model = None
    _tok = None
    _lock = threading.Lock()

    @classmethod
    def get(cls, repo_id: str, device):
        with cls._lock:
            if cls._model is None:
                from .teacher import load_teacher

                cls._model, cls._tok = load_teacher(repo_id, device=device, dtype=torch.bfloat16)
                cls._model.eval()
            return cls._model, cls._tok

    @classmethod
    def clear(cls) -> None:
        cls._model = cls._tok = None


@torch.no_grad()
def _ctx_embed(latent: LatentBundle, editor, ids: Sequence[int], device, ctx_len: int = 192) -> torch.Tensor:
    x = torch.tensor([list(ids)[-ctx_len:]], dtype=torch.long, device=device)
    h = editor.hidden(x, None)
    return latent.heads.ctx_embed(h.mean(dim=1))


@torch.no_grad()
def _plan_logits(latent: LatentBundle, editor, z: torch.Tensor, k: int, device) -> torch.Tensor:
    """Per-position plan logits for the first k positions of a chunk."""
    out = latent.adapter(z.unsqueeze(0), reparam=False)
    logits = editor.backbone.lm_head(out["logits_hidden"])  # [1, max_chunk, V]
    return logits[0, :k].float()


@torch.no_grad()
def _fill_plan_block(
    mlm_call: Callable[[torch.Tensor], torch.Tensor],
    plan_logits: torch.Tensor,
    ids: list[int],
    base: int,
    k: int,
    cfg: LatentSamplerCfg,
    device,
    plan_weight: float,
) -> tuple[list[int], int]:
    """Iterative unmask of k mask positions with backbone + plan prior mixing.
    plan_weight is passed explicitly (retries halve it) so the shared cfg is
    never mutated mid-generation."""
    x = torch.tensor([ids], dtype=torch.long, device=device)
    masked = list(range(base, base + k))
    forwards = 0
    for step in range(cfg.fill_steps):
        if not masked:
            break
        logits = mlm_call(x)
        forwards += 1
        pos = torch.tensor(masked, dtype=torch.long, device=x.device)
        if plan_weight != 0.0:
            logits = logits.clone()
            mix = (plan_weight * plan_logits[pos - base]).to(logits.dtype)
            logits[0, pos] = logits[0, pos] + mix
        tok, conf = pick_token(logits[0, pos], cfg.temperature, cfg.top_p, None)
        kk = math.ceil(len(masked) / (cfg.fill_steps - step))
        kk = min(kk, len(masked))
        commit = conf.argsort(descending=True)[:kk]
        x[0, pos[commit]] = tok[commit].to(x.dtype)
        committed = set(pos[commit].tolist())
        masked = [p for p in masked if p not in committed]
    return x[0].tolist(), forwards


@torch.no_grad()
def _cycle_score(
    cfg: LatentSamplerCfg, device, ids: list[int], span: tuple[int, int], z_pred: torch.Tensor
) -> float:
    """Cosine between the re-encoded committed chunk and the predicted latent."""
    if not cfg.cycle_consistency:
        return 1.0
    try:
        model, _ = _CycleTeacher.get(cfg.teacher_repo, device)
    except Exception:
        return 1.0
    x = torch.tensor([ids], dtype=torch.long, device=device)
    h = model(x, use_cache=False, output_hidden_states=True).hidden_states[-1][0].float()
    vec = torch.nn.functional.normalize(h[span[0] : span[1]].mean(0), dim=-1)
    # z_pred = z_q + residual candidate is only approximately unit-norm; a raw
    # dot product would conflate norm drift with directional agreement
    zp = torch.nn.functional.normalize(z_pred.float(), dim=-1)
    return float((vec * zp).sum())


@torch.no_grad()
def generate_latent(
    editor,
    latent: LatentBundle,
    bundle: TokenizerBundle,
    prompt_ids: Sequence[int],
    cfg: LatentSamplerCfg,
    device: torch.device | str = "cpu",
) -> GenResult:
    ids = list(prompt_ids)
    prompt_len = len(ids)
    stop_set = set(bundle.stop_ids)
    forwards = 0
    t0 = time.perf_counter()
    coarse_done = 0
    K_f = cfg.fine_chunk_tokens
    K_c = K_f * cfg.fine_per_coarse
    prev_coarse = torch.zeros(1, 0, 2, dtype=torch.long, device=device)
    window_ctx: torch.Tensor | None = None
    cycle_cosines: list[float] = []

    for _ in range(cfg.max_coarse_chunks):
        if len(ids) + K_c > cfg.max_total_len:
            break
        # training pins the context at the code window's start (store.batch:
        # ctx = text before chunk c0, codes c0..c0+W-1), so the coarse ctx
        # embed is refreshed only when a new window opens — advancing it every
        # chunk would overlap the ctx text with the chunks in the code history,
        # a conditioning pattern the prior never trained on
        if prev_coarse.shape[1] == 0 or window_ctx is None:
            window_ctx = _ctx_embed(latent, editor, ids, device, cfg.ctx_len)
        z_c, codes_c = latent.predict_coarse_latent(window_ctx, prev_coarse, cfg.__dict__)
        chunk_start = len(ids)
        # fine ctx is pinned at THIS coarse chunk's start for the whole fine
        # window (training: ctx = text before the fine chunks' own coarse
        # chunk), not recomputed per fine chunk
        fine_ctx = window_ctx if prev_coarse.shape[1] == 0 else _ctx_embed(latent, editor, ids, device, cfg.ctx_len)

        best_try, best_cos = None, -1.0
        for attempt in range(cfg.cycle_retries + 1):
            plan_w = cfg.plan_weight * (0.5**attempt)
            chunk_ids = list(ids)
            prev_fine = torch.zeros(1, 0, 2, dtype=torch.long, device=device)
            stopped = False
            for j in range(cfg.fine_per_coarse):
                if len(chunk_ids) + K_f > cfg.max_total_len:
                    break
                z_f, codes_f = latent.predict_fine_latent(fine_ctx, codes_c, prev_fine, cfg.__dict__)
                plan = _plan_logits(latent, editor, z_f, K_f, device)
                base = len(chunk_ids)
                chunk_ids = chunk_ids + [bundle.mask_id] * K_f
                chunk_ids, fwd = _fill_plan_block(
                    editor.mlm_call(), plan, chunk_ids, base, K_f, cfg, device, plan_w
                )
                forwards += fwd
                prev_fine = torch.cat([prev_fine, codes_f.unsqueeze(0)], dim=1)
                if any(t in stop_set for t in chunk_ids[base:]):
                    stopped = True
                    break
            cos = _cycle_score(cfg, device, chunk_ids, (chunk_start, len(chunk_ids)), z_c)
            cycle_cosines.append(cos)
            if cos >= best_cos:
                best_cos, best_try = cos, chunk_ids
            if cos >= cfg.cycle_threshold or stopped:
                break
        ids = best_try or ids
        coarse_done += 1

        if any(t in stop_set for t in ids[chunk_start:]):
            break
        if cfg.stop_texts:
            txt = bundle.decode(ids[prompt_len:])
            if any(st in txt for st in cfg.stop_texts):
                break
        prev_coarse = torch.cat([prev_coarse, codes_c.unsqueeze(0)], dim=1)
        if prev_coarse.shape[1] >= cfg.code_history:
            prev_coarse = prev_coarse[:, :0]  # window full: next chunk opens a fresh one

    seconds = time.perf_counter() - t0
    new_ids = [t for t in ids[prompt_len:] if t not in stop_set]
    text = bundle.decode(new_ids)
    for st in cfg.stop_texts:
        cut = text.find(st)
        if cut >= 0:
            text = text[:cut]
    res = GenResult(
        ids=ids, new_ids=new_ids, blocks=coarse_done, forwards=forwards, seconds=seconds, text=text
    )
    res.cycle_cosines = cycle_cosines  # type: ignore[attr-defined]
    return res
