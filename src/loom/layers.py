"""Loom building blocks: RMSNorm, RoPE, GQA attention, SwiGLU, MoE.

Design notes:
- Pre-norm everywhere; residual out-projections are marked with
  `_is_residual_out` so LoomLM can scale their init by 1/sqrt(2*depth).
- MoE layers return (out, aux) where aux carries the load-balance and
  router-z losses; the model averages them over (core_layers x n_loops).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Optionally carries `n_cond` per-condition gain deltas (adaLN-style).

    The delta is added to the gain, which multiplies an already unit-RMS
    vector — so a delta of 0.1 is 10% of the signal no matter how large the
    residual stream has grown. That is the whole point: additive conditioning
    at the loop entry point cannot compete with an RMS-100 stream, gain
    conditioning always can. Zero-init => exact no-op at step 0.
    """

    def __init__(self, dim: int, eps: float = 1e-5, n_cond: int = 0):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.cond = nn.Parameter(torch.zeros(n_cond, dim)) if n_cond else None
        self.eps = eps

    def forward(self, x: torch.Tensor, cond_idx: int | None = None) -> torch.Tensor:
        dt = x.dtype
        w = self.weight
        if self.cond is not None and cond_idx is not None:
            w = w + self.cond[cond_idx]
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * w.float()).to(dt)


def build_rope_cache(max_seq: int, head_dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin caches [max_seq, head_dim] (half-dims duplicated, NeoX layout)."""
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq).float()
    freqs = torch.outer(t, inv)  # [T, hd/2]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """q,k: [B, H, T, hd]; cos/sin: [T, hd] for these positions."""
    cos = cos[None, None, :, :].to(q.dtype)
    sin = sin[None, None, :, :].to(q.dtype)
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv = n_kv_heads
        self.hd = d_model // n_heads
        self.q_proj = nn.Linear(d_model, n_heads * self.hd, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.hd, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.hd, bias=False)
        self.o_proj = nn.Linear(n_heads * self.hd, d_model, bias=False)
        self.o_proj._is_residual_out = True

    def forward(
        self,
        x: torch.Tensor,  # [B, T, D]
        cos: torch.Tensor,  # [T, hd] rope for the CURRENT positions
        sin: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        new_kv = (k, v)
        rep = self.n_heads // self.n_kv
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        S = k.shape[2]
        if T == S:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=T > 1)
        elif T == 1:
            out = F.scaled_dot_product_attention(q, k, v)
        else:  # chunked decode with a prefix cache: offset causal mask
            pos_q = torch.arange(S - T, S, device=x.device)
            pos_k = torch.arange(S, device=x.device)
            mask = pos_k[None, :] <= pos_q[:, None]
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(out), new_kv


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
        self.down._is_residual_out = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MoELayer(nn.Module):
    """Top-k softmax routing over SwiGLU experts, plus an always-on shared
    expert (DeepSeek-style). Router stays small and trains under AdamW."""

    def __init__(self, d_model: int, n_experts: int, top_k: int, d_ff_expert: int, shared: bool,
                 n_loops: int = 0):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, n_experts, bias=False)
        # Per-loop logit bias: lets loop r shift its expert preferences without
        # the state having to encode "which loop am I" first. Acts in logit
        # space, so it is scale-free like the norm gains. Zero-init = no-op.
        self.loop_bias = nn.Parameter(torch.zeros(n_loops, n_experts)) if n_loops else None
        self.experts = nn.ModuleList(SwiGLU(d_model, d_ff_expert) for _ in range(n_experts))
        self.shared = SwiGLU(d_model, d_ff_expert) if shared else None

    def forward(self, x: torch.Tensor, loop_idx: int | None = None) -> tuple[torch.Tensor, dict]:
        B, T, D = x.shape
        flat = x.reshape(-1, D)
        logits = self.router(flat).float()  # [N, E]
        if self.loop_bias is not None and loop_idx is not None:
            logits = logits + self.loop_bias[loop_idx].float()
        probs = logits.softmax(-1)
        top_p, top_i = probs.topk(self.top_k, dim=-1)
        top_p = top_p / top_p.sum(-1, keepdim=True)  # renormalize the chosen k
        out = torch.zeros_like(flat)
        for e in range(self.n_experts):
            rows, slot = (top_i == e).nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue
            y = self.experts[e](flat[rows])
            # under autocast the experts emit bf16 while the residual stream
            # (and hence `out`) is fp32 — index_add_ demands matching dtypes
            out.index_add_(0, rows, (y * top_p[rows, slot].unsqueeze(-1)).to(out.dtype))
        if self.shared is not None:
            out = out + self.shared(flat)
        # Switch-style load balance: E * sum_e f_e * P_e  (f = routed fraction)
        with torch.no_grad():
            counts = torch.zeros(self.n_experts, device=x.device, dtype=torch.float)
            counts.scatter_add_(0, top_i.reshape(-1), torch.ones_like(top_i.reshape(-1), dtype=torch.float))
            f = counts / max(top_i.numel(), 1)
        lb = self.n_experts * (f * probs.mean(0)).sum()
        z = logits.logsumexp(-1).pow(2).mean()
        return out.reshape(B, T, D), {"lb": lb, "z": z}


class DenseBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, d_ff: int, eps: float):
        super().__init__()
        self.attn_norm = RMSNorm(d_model, eps)
        self.attn = Attention(d_model, n_heads, n_kv_heads)
        self.ffn_norm = RMSNorm(d_model, eps)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x, cos, sin, past_kv=None):
        a, kv = self.attn(self.attn_norm(x), cos, sin, past_kv)
        x = x + a
        x = x + self.ffn(self.ffn_norm(x))
        return x, kv


class MoEBlock(nn.Module):
    """The looped core block. `n_loops > 0` enables per-loop conditioning on
    both norms and the router; `loop_idx=None` at call time falls back to
    unconditioned behaviour."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, n_experts: int, top_k: int,
                 d_ff_expert: int, shared: bool, eps: float, n_loops: int = 0):
        super().__init__()
        self.attn_norm = RMSNorm(d_model, eps, n_cond=n_loops)
        self.attn = Attention(d_model, n_heads, n_kv_heads)
        self.ffn_norm = RMSNorm(d_model, eps, n_cond=n_loops)
        self.moe = MoELayer(d_model, n_experts, top_k, d_ff_expert, shared, n_loops=n_loops)

    def forward(self, x, cos, sin, past_kv=None, loop_idx=None):
        a, kv = self.attn(self.attn_norm(x, loop_idx), cos, sin, past_kv)
        x = x + a
        m, aux = self.moe(self.ffn_norm(x, loop_idx), loop_idx)
        x = x + m
        return x, kv, aux
