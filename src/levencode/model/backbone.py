"""Loading and probing the LFM2.5 bidirectional masked-diffusion backbone.

The HF repo ships custom code (modeling_lfm2_bidirectional.py) exposing
Lfm2BidirectionalForMaskedLM with `.lfm2` (base encoder) and `.lm_head`
(weight-tied to the embeddings). We always fetch hidden states from the base
module directly so the edit heads see the definitive post-norm activations."""

from __future__ import annotations

import torch

from ..data.tokens import TokenizerBundle, bundle_from_tokenizer

REPO_ID = "LiquidAI/LFM2.5-Encoder-350M-Diffusion"
ANSWER_SUFFIX = "\n[/Answer]"


def load_tokenizer_bundle(repo_id: str = REPO_ID) -> TokenizerBundle:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
    return bundle_from_tokenizer(tok, answer_suffix=ANSWER_SUFFIX)


def load_backbone(
    repo_id: str = REPO_ID,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
):
    from transformers import AutoModelForMaskedLM

    model = AutoModelForMaskedLM.from_pretrained(
        repo_id, trust_remote_code=True, dtype=dtype
    )
    return model.to(device)


def hidden_and_logits(
    backbone, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """(post-norm hidden states, MLM logits) in a single base-model pass."""
    base = backbone.lfm2(
        input_ids=input_ids, attention_mask=attention_mask, return_dict=True
    )
    h = base.last_hidden_state
    return h, backbone.lm_head(h)


def tiny_backbone(repo_id: str = REPO_ID):
    """Randomly initialized model with the real architecture but tiny dims —
    unit tests exercise the exact custom code path without the 1.4GB weights."""
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    cfg = AutoConfig.from_pretrained(repo_id, trust_remote_code=True)
    cfg.update(
        dict(
            hidden_size=64,
            block_dim=64,
            conv_dim=64,
            conv_dim_out=64,
            intermediate_size=128,
            num_hidden_layers=2,
            layer_types=["conv", "full_attention"],
            num_attention_heads=4,
            num_heads=4,
            num_key_value_heads=2,
        )
    )
    cls = get_class_from_dynamic_module(
        "modeling_lfm2_bidirectional.Lfm2BidirectionalForMaskedLM", repo_id
    )
    return cls(cfg)
