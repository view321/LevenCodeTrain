"""LoomLM: prelude -> looped MoE core (input-injected, concept-modulated) -> coda.

Loop recipe (Huginn-style recurrent depth, fixed R):
    e   = prelude(embed(ids))            # token embedding enriched once
    s_0 = e
    u_r = adapter([s_r ; e]) + loop_emb_r * rms(u)      # depth tag, scale-free
    s_{r+1} = core_r( u_r modulated by FiLM(c) )        # core_r: per-loop gains
    logits  = lm_head(norm(coda(s_R)))

- The adapter re-injects e every loop so the state cannot drift away from the
  token evidence.
- Depth conditioning is scale-relative everywhere. The first run's fixed-scale
  `loop_emb` measured vestigial (RMS 0.05 against an adapter output at RMS
  48-103, ratio 0.001, dCE +0.0001 at 2.6B tokens): its gradient is
  proportional to its own effect, so it can never bootstrap. The live channels
  are per-loop RMSNorm gain deltas and a per-loop router logit bias, both
  scale-free and zero-init (see `LoomConfig.per_loop_cond`), plus `loop_emb`
  now expressed in units of the stream's own RMS. `cond_report()` surfaces all
  three in the live metrics so a silently-inert pathway is visible immediately.
- Concept FiLM applies to the loop-state entry point (zero-init in
  ConceptModulator), touching all R loops — plans steer computation, not logits.
- KV caches: the core runs R times per token with different states, so every
  (loop, layer) pair owns a cache slot; cache order is fixed by construction.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .concept import ConceptModulator, ConceptPredictor, pool_segments
from .config import LoomConfig
from .layers import DenseBlock, MoEBlock, RMSNorm, build_rope_cache


def _ce_sum(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int) -> torch.Tensor:
    """Summed CE over one chunk; the fp32 cast lives and dies inside here."""
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        labels.reshape(-1),
        ignore_index=ignore_index,
        reduction="sum",
    )


def shifted_cross_entropy(
    logits: torch.Tensor,   # [B, T, V]
    labels: torch.Tensor,   # [B, T]
    ignore_index: int = -100,
    chunks: int = 1,
) -> torch.Tensor:
    """Next-token CE (logits[:, :-1] predicts labels[:, 1:]) computed in
    row-chunks to keep the fp32 logit copy small.

    Chunking ALONE saves nothing — every chunk's cast and log_softmax would
    survive until backward — so each chunk is wrapped in `checkpoint` and
    recomputed. Peak becomes one chunk's worth instead of the whole batch's.
    Summing then dividing by the valid-token count is exactly `reduction=
    "mean"` with `ignore_index`, and accumulating in fp32 across chunks is if
    anything better conditioned than one big reduction.
    """
    B, V = logits.shape[0], logits.shape[-1]
    lg_all, lb_all = logits[:, :-1], labels[:, 1:]
    if chunks <= 1:
        return F.cross_entropy(
            lg_all.reshape(-1, V).float(), lb_all.reshape(-1), ignore_index=ignore_index
        )
    n_valid = (lb_all != ignore_index).sum().clamp_min(1)
    size = max((B + chunks - 1) // chunks, 1)
    total = None
    for i in range(0, B, size):
        lg, lb = lg_all[i : i + size], lb_all[i : i + size]
        if torch.is_grad_enabled() and lg.requires_grad:
            part = checkpoint(_ce_sum, lg, lb, ignore_index, use_reentrant=False)
        else:
            part = _ce_sum(lg, lb, ignore_index)
        total = part if total is None else total + part
    return total / n_valid


def _rms(x: torch.Tensor) -> torch.Tensor:
    """Per-token RMS of the residual stream, detached — the unit that
    conditioning terms are expressed in so they stay trainable as the stream
    norm grows through training and across loops."""
    return x.detach().pow(2).mean(-1, keepdim=True).sqrt()


class LoomLM(nn.Module):
    def __init__(self, cfg: LoomConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.embed = nn.Embedding(cfg.vocab_size, d)
        self.prelude = nn.ModuleList(
            DenseBlock(d, cfg.n_heads, cfg.n_kv_heads, cfg.d_ff_dense, cfg.norm_eps)
            for _ in range(cfg.prelude_layers)
        )
        self.core = nn.ModuleList(
            MoEBlock(d, cfg.n_heads, cfg.n_kv_heads, cfg.n_experts, cfg.top_k,
                     cfg.d_ff_expert, cfg.shared_expert, cfg.norm_eps,
                     n_loops=cfg.n_loops if cfg.per_loop_cond else 0)
            for _ in range(cfg.core_layers)
        )
        self.coda = nn.ModuleList(
            DenseBlock(d, cfg.n_heads, cfg.n_kv_heads, cfg.d_ff_dense, cfg.norm_eps)
            for _ in range(cfg.coda_layers)
        )
        self.adapter = nn.Linear(2 * d, d, bias=False)
        self.adapter._is_residual_out = True
        self.loop_emb = nn.Parameter(torch.zeros(cfg.n_loops, d))
        self.norm = RMSNorm(d, cfg.norm_eps)
        self.lm_head = nn.Linear(d, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        self.concept_predictor = ConceptPredictor(cfg)
        self.modulator = ConceptModulator(d)
        cos, sin = build_rope_cache(cfg.max_seq_len, cfg.head_dim, cfg.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)
        nn.init.zeros_(self.modulator.proj.weight)  # re-assert after apply()
        nn.init.normal_(self.loop_emb, std=cfg.init_std)

    def _init(self, m: nn.Module) -> None:
        std = self.cfg.init_std
        if isinstance(m, nn.Linear):
            s = std
            if getattr(m, "_is_residual_out", False):
                s = std / (2 * self.cfg.effective_depth) ** 0.5
            nn.init.normal_(m.weight, std=s)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=std)

    # ---------- forward ----------

    @property
    def n_cache_slots(self) -> int:
        c = self.cfg
        return c.prelude_layers + c.n_loops * c.core_layers + c.coda_layers

    def _concept_film(
        self, concepts: torch.Tensor, past_len: int, T: int, device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather each token's segment concept -> (gamma, beta) [B, T, D].
        `concepts` [B, S, D] must satisfy the causality contract (segment j's
        vector derived from segments < j only)."""
        pos = torch.arange(past_len, past_len + T, device=device)
        seg = (pos // self.cfg.segment_len).clamp_max(concepts.shape[1] - 1)
        per_tok = concepts.index_select(1, seg)  # [B, T, D]
        return self.modulator(per_tok)

    def forward(
        self,
        input_ids: torch.Tensor,  # [B, T]
        concepts: torch.Tensor | None = None,  # [B, S, D], causally valid
        labels: torch.Tensor | None = None,    # [B, T], -100 = ignore
        past: list | None = None,
        use_cache: bool = False,
        grad_checkpoint: bool = False,
    ) -> dict:
        cfg = self.cfg
        B, T = input_ids.shape
        past_len = past[0][0].shape[2] if past else 0
        cos = self.cos[past_len : past_len + T]
        sin = self.sin[past_len : past_len + T]
        ckpt = grad_checkpoint and self.training and not use_cache

        def run(block, x, kv, *extra):
            if ckpt:
                return checkpoint(block, x, cos, sin, kv, *extra, use_reentrant=False)
            return block(x, cos, sin, kv, *extra)

        new_past: list = []
        slot = 0

        def kv_in():
            return past[slot] if past else None

        h = self.embed(input_ids)
        for blk in self.prelude:
            h, kv = run(blk, h, kv_in())
            new_past.append(kv)
            slot += 1
        e = h

        film = None
        if concepts is not None:
            film = self._concept_film(concepts, past_len, T, input_ids.device)

        aux_lb = h.new_zeros(())
        aux_z = h.new_zeros(())
        film_stats: dict = {}
        n_moe = 0
        s = e
        for r in range(cfg.n_loops):
            # Both injections are scaled by the stream's own RMS. A fixed-scale
            # additive term is unusable here: the adapter output runs at RMS
            # ~50-100 once trained, so an init-scale vector sits 1000x below
            # the signal and its gradient is too small to ever bootstrap
            # (measured: loop_emb ratio 0.001, dCE +0.0001 at 2.6B tokens).
            u = self.adapter(torch.cat([s, e], dim=-1))
            u = u + self.loop_emb[r] * _rms(u)
            if film is not None:
                rms = _rms(u)
                if r == 0:  # FiLM health, sampled once per forward
                    with torch.no_grad():
                        film_stats = {
                            "beta_frac": float(film[1].abs().mean() / rms.mean().clamp_min(1e-9)),
                            "gamma_rms": float(film[0].pow(2).mean().sqrt()),
                        }
                u = u * (1 + film[0]) + film[1] * rms
            for blk in self.core:
                u, kv, aux = run(blk, u, kv_in(), r)
                new_past.append(kv)
                slot += 1
                aux_lb = aux_lb + aux["lb"]
                aux_z = aux_z + aux["z"]
                n_moe += 1
            s = u
        for blk in self.coda:
            s, kv = run(blk, s, kv_in())
            new_past.append(kv)
            slot += 1

        hidden = self.norm(s)
        logits = self.lm_head(hidden)
        aux_lb = aux_lb / max(n_moe, 1)
        aux_z = aux_z / max(n_moe, 1)

        out = {
            "logits": logits,
            "hidden": hidden,
            "aux_lb": aux_lb,
            "aux_z": aux_z,
            "past": new_past if use_cache else None,
            **film_stats,
        }
        if labels is not None:
            ce = shifted_cross_entropy(logits, labels, chunks=cfg.ce_chunks)
            out["ce"] = ce
            out["loss"] = ce + cfg.router_aux_weight * aux_lb + cfg.router_z_weight * aux_z
        return out

    # ---------- concept helpers ----------

    @torch.no_grad()
    def pooled_concepts(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Target concepts: pooled final hiddens of a plain (unguided) pass.
        Phase-2 training points this at an EMA copy of the model."""
        h = self.forward(input_ids)["hidden"]
        return pool_segments(h, self.cfg.segment_len)

    # ---------- conditioning health ----------

    @torch.no_grad()
    def cond_report(self) -> dict:
        """Is depth conditioning actually being used? Every channel here is
        zero (or near-zero) at init and only matters if it moves off zero, so
        these go in the live metrics — a silent no-op pathway is exactly the
        failure that made `loop_emb` vestigial in the first run."""
        out = {"loop_emb_frac": float(self.loop_emb.pow(2).mean(-1).sqrt().mean())}
        gains, biases = [], []
        for blk in self.core:
            for nrm in (blk.attn_norm, blk.ffn_norm):
                if nrm.cond is not None:
                    # gain delta relative to the base gain it perturbs
                    gains.append(
                        float(nrm.cond.pow(2).mean(-1).sqrt().mean())
                        / max(float(nrm.weight.pow(2).mean().sqrt()), 1e-9)
                    )
            if blk.moe.loop_bias is not None:
                biases.append(float(blk.moe.loop_bias.abs().mean()))
        if gains:
            out["cond_gain_frac"] = sum(gains) / len(gains)
        if biases:
            out["cond_router_bias"] = sum(biases) / len(biases)
        return out

    # ---------- optimizer partition ----------

    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        """Muon for the 2D trunk weights; AdamW for everything Muon's
        orthogonalized update would mistreat: embeddings (tied head included),
        norms/1D, the router (keep routing adaptation snappy), the zero-init
        modulator, and the loop embeddings."""
        # `cond` and `loop_bias` are 2D but are gain/bias stacks, not linear
        # maps — Muon's orthogonalized update would scramble them.
        adamw_name = ("embed", "lm_head", "router", "modulator", "loop_emb", "bos",
                      "cond", "loop_bias")
        muon, adamw = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim >= 2 and not any(k in name for k in adamw_name):
                muon.append(p)
            else:
                adamw.append(p)
        return {"muon": muon, "adamw": adamw}

    # ---------- generation (minimal; the training-time sampler) ----------

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,  # [1, T]
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.9,
        stop_ids: tuple[int, ...] = (),
        use_concepts: bool = False,
    ) -> torch.Tensor:
        ids = input_ids
        past = None
        concepts = None
        # never generate past the RoPE cache / trained context
        limit = max(self.cfg.max_seq_len - ids.shape[1], 0)
        for _ in range(min(max_new_tokens, limit)):
            if use_concepts:
                # re-plan at segment boundaries from the committed tokens
                if concepts is None or ids.shape[1] % self.cfg.segment_len == 0:
                    pooled = self.pooled_concepts(ids)
                    concepts = self.concept_predictor(pooled)[:, 1:]  # drop BOS slot
                    past = None  # FiLM changed: cached states are stale
            step = ids if past is None else ids[:, -1:]
            out = self.forward(step, concepts=concepts, past=past, use_cache=True)
            past = out["past"]
            logits = out["logits"][:, -1].float()
            if temperature <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                probs = logits.softmax(-1)
                sp, si = probs.sort(descending=True)
                keep = sp.cumsum(-1) - sp < top_p
                keep[..., 0] = True
                sp = sp * keep
                nxt = si.gather(-1, torch.multinomial(sp / sp.sum(-1, keepdim=True), 1))
            ids = torch.cat([ids, nxt], dim=1)
            if stop_ids and int(nxt) in stop_ids:
                break
        return ids

    # ---------- checkpoint IO ----------

    def save(self, ckpt_dir: str | Path) -> None:
        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "config.json").write_text(
            json.dumps(self.cfg.__dict__, indent=2), encoding="utf-8"
        )
        torch.save(self.state_dict(), ckpt_dir / "model.pt")

    @classmethod
    def load(cls, ckpt_dir: str | Path, device="cpu") -> "LoomLM":
        ckpt_dir = Path(ckpt_dir)
        cfg = LoomConfig.from_dict(json.loads((ckpt_dir / "config.json").read_text(encoding="utf-8")))
        model = cls(cfg)
        model.load_state_dict(torch.load(ckpt_dir / "model.pt", map_location="cpu", weights_only=True))
        return model.to(device)


def param_report(model: LoomLM) -> dict:
    """Total / active-per-token parameter accounting for sizing sanity."""
    cfg = model.cfg
    total = sum(p.numel() for p in model.parameters())
    embed = model.embed.weight.numel()
    core_total = sum(p.numel() for p in model.core.parameters())
    per_layer_expert = sum(p.numel() for p in model.core[0].moe.experts[0].parameters())
    shared = sum(p.numel() for p in model.core[0].moe.shared.parameters()) if cfg.shared_expert else 0
    attn = sum(p.numel() for p in model.core[0].attn.parameters())
    core_active_per_loop = cfg.core_layers * (attn + cfg.top_k * per_layer_expert + shared)
    dense = sum(p.numel() for b in list(model.prelude) + list(model.coda) for p in b.parameters())
    concept = sum(p.numel() for p in model.concept_predictor.parameters())
    active = dense + cfg.n_loops * core_active_per_loop + model.adapter.weight.numel() * cfg.n_loops
    return {
        "total": total,
        "embed_tied": embed,
        "dense_prelude_coda": dense,
        "core_total": core_total,
        "core_active_per_loop": core_active_per_loop,
        "concept_stack": concept,
        "active_compute_per_token(excl. head)": active,
        "active_incl_head": active + embed,
        "effective_depth": cfg.effective_depth,
    }
