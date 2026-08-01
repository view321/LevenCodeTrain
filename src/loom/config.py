"""Loom configuration.

Loom = looped-MoE token decoder + concept-level guidance in its own latent
space. Sizing targets a single RTX 5090 (32GB) and a ~5B-token budget:

    active compute / token ~ prelude+coda (45M) + n_loops x core-active (48M)
                             + tied LM head (67M)  ~ 262M effective
    6 * 262M * 5e9 tokens  ~ 7.9e18 FLOPs  ~ 24-30h at realistic 5090 MFU

which is compute-per-token comparable to a 2.7B-A350M MoE baseline while
holding total parameters to ~285M (weight reuse via looping)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LoomConfig:
    # tokenizer (LFM2 shared vocab by default)
    vocab_size: int = 65536
    tie_embeddings: bool = True

    # trunk
    d_model: int = 1024
    n_heads: int = 16
    n_kv_heads: int = 4
    prelude_layers: int = 2
    core_layers: int = 4
    coda_layers: int = 2
    n_loops: int = 3
    d_ff_dense: int = 2816
    max_seq_len: int = 4096
    rope_theta: float = 100000.0
    norm_eps: float = 1e-5

    # MoE core
    n_experts: int = 8
    top_k: int = 2
    d_ff_expert: int = 1024
    shared_expert: bool = True
    router_aux_weight: float = 0.01
    router_z_weight: float = 1e-3

    # concept level (LCM-style planner in the model's own hidden space)
    segment_len: int = 32
    concept_layers: int = 4
    concept_heads: int = 8
    concept_ff: int = 2816

    init_std: float = 0.02

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def effective_depth(self) -> int:
        """Depth as seen by a token in one forward."""
        return self.prelude_layers + self.n_loops * self.core_layers + self.coda_layers

    @property
    def max_segments(self) -> int:
        return (self.max_seq_len + self.segment_len - 1) // self.segment_len

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must divide n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must divide n_kv_heads (GQA)")
        if self.top_k > self.n_experts:
            raise ValueError("top_k > n_experts")

    @classmethod
    def from_dict(cls, d: dict) -> "LoomConfig":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})

    @classmethod
    def tiny(cls) -> "LoomConfig":
        """CPU-test scale."""
        return cls(
            vocab_size=128, d_model=32, n_heads=4, n_kv_heads=2,
            prelude_layers=1, core_layers=2, coda_layers=1, n_loops=2,
            d_ff_dense=64, n_experts=4, top_k=2, d_ff_expert=32,
            max_seq_len=64, segment_len=4, concept_layers=2, concept_heads=2,
            concept_ff=64,
        )
