# Loom

A from-scratch LM combining three ideas, sized for a single RTX 5090 and a
~5B-token budget, to be compared against a 2.7B-A350M MoE baseline trained on
similar compute:

1. **Looped MoE core** (weight reuse): prelude (2 dense layers) → core
   (4 MoE layers applied R=3 times, Huginn-style input injection + scale-free
   per-loop conditioning) → coda (2 dense layers). Effective depth 16 from 8
   layers' worth of unique weights; ~294M total params, ~263M
   active-equivalent compute per token (incl. tied head).
2. **Token-level autoregression** with plain cross-entropy. The Levencode
   stage-5 postmortem's central lesson: keep the densest, best-conditioned
   training signal at the fine level; do not generate through latents.
3. **Concept guidance** (the LCM/JEPA part): mean-pooled final hiddens per
   32-token segment form concepts in the model's OWN space; a small causal
   ConceptPredictor plans the next concept; the plan conditions all core
   loops via zero-init FiLM. Guidance-only — a wrong plan degrades toward
   the unguided model instead of overriding it (contrast: stage-5 logit
   mixing, where a broken plan produced `]`).

## Why this shape (postmortem → design)

| Stage-5 failure | Loom answer |
|---|---|
| Frozen-teacher latent space needed a decodability adapter that had to learn to "speak" | Concepts pooled from the model's own hiddens — natively decodable, no adapter |
| Plan logits mixed at full magnitude overrode the filler | Zero-init FiLM on the loop state; guidance fades in only as it helps |
| 8-token canvases were OOD for the filler | No canvases: token AR all the way down; concepts change *computation*, not the output space |
| Plan prior conditioned on one pooled ctx vector, never beat chance | Predictor attends over the full concept sequence; phase-2 ablations gate whether it earns its keep |
| RVQ/energy/CFG machinery = many links that can silently fail | Regression-only concept loss (guidance tolerates mean-regression); each phase benches against the previous |

## Run 1 postmortem (2.6B tokens) → run 2 changes

Routing analysis on `step_30000` (`scripts/loom_routing.py`) showed the loops
run as a **staged pipeline, not repeated identical computation**: router
entropy falls monotonically with loop index (~2.95 → ~2.7 → ~1.1-1.9 bits of
3.0) in every core layer on every corpus, cross-loop usage JS exceeds
cross-domain JS (0.054 vs 0.032 bits, null 0.00002), and at core layer 3 loops
0 and 2 route independently (kappa +0.02). That happened *despite* depth
conditioning being broken, not because of it.

`scripts/loom_loopemb.py` found `loop_emb` vestigial: RMS 0.046-0.074 against
an adapter output at RMS 48.8-102.9 (ratio 0.001), zeroing it costs +0.0001
nats. It is not that the model didn't need a depth tag — the tag was never
operative, because a fixed-scale additive term cannot bootstrap against a
stream three orders of magnitude larger. Run 2 therefore makes every
conditioning channel scale-relative (`per_loop_cond`, `_rms`-scaled `loop_emb`
and FiLM beta) and logs `cond_gain_frac` / `cond_router_bias` /
`loop_emb_frac` / `beta_frac` so inertness is visible live rather than
inferred from a null bench.

## Training phases (planned)

1. **Pretrain** (~5B tokens): pure looped MoE — concepts off (zero-init makes
   them a structural no-op). Muon on 2D trunk weights, AdamW on
   embeddings/norms/router/modulator (`LoomLM.param_groups()`).
2. **Concept phase**: freeze-ish trunk, train predictor + modulator; targets
   are pooled hiddens from an EMA copy (`pooled_concepts`), loss
   `concept_loss` (smooth-L1 on RMS-normalized targets), teacher-forced
   concepts mixed with predicted ones on a schedule.
3. **SFT** on the instruct mix; bench vs the 2.7B-A350M baseline at matched
   token budget (5B pretrain + 0.5B SFT ≈ the baseline's 5.5B).

Ablation arms the architecture supports for free: n_loops (1 = plain MoE),
concepts off / `shift_concepts` (previous-segment conditioning, no predictor)
/ full predictor.

## Files

- `config.py` — `LoomConfig` (+ `tiny()` test preset); sizing math in docstring
- `layers.py` — RMSNorm, RoPE, GQA attention (KV-cache aware), SwiGLU, MoE
  with load-balance + z losses
- `model.py` — `LoomLM` (loop wiring, FiLM injection, per-(loop,layer) KV
  cache, Muon/AdamW param split, minimal sampler, checkpoint IO),
  `param_report`
- `concept.py` — segment pooling, `ConceptPredictor`, `ConceptModulator`,
  `concept_loss`, `shift_concepts`
- `data.py` — packed pretrain stream (fineweb-edu 55% / code 25% / math 20%)
  and the SFT chat stream, over levencode's retrying WeightedMixer
- `muon.py` — single-device Muon (Newton-Schulz orthogonalized momentum)
- `train.py` — `LoomTrainer` (three stages, Muon+AdamW, EMA concept targets,
  gallery + WebUI state via levencode's `RunDir`)
- `bench.py` — causal bench suite: WikiText-103 held-out CE/ppl, chat CE,
  ARC-Easy (mean-logprob MC), GSM8K EM, MBPP (sandboxed), speed, BrierLM
  (positive-clamped composite); same result shape as levencode's bench

## Running

```bash
python scripts/loom_train.py --config configs/loom_pretrain.yaml   # ~24-30h
python scripts/loom_train.py --config configs/loom_concept.yaml    # ~1.5h
python scripts/loom_train.py --config configs/loom_sft.yaml        # ~3h
python scripts/loom_bench.py --config configs/loom_sft.yaml \
    --ckpt runs/loom/sft/ckpt/final --name sft_guided
python scripts/loom_bench.py --config configs/loom_sft.yaml \
    --ckpt runs/loom/sft/ckpt/final --name sft_noconcepts \
    --set bench.use_concepts=false
```

Runs land in `runs/loom/<stage>/` — the existing WebUI
(`python scripts/serve.py --runs-dir runs --port 7860`) shows progress/ETA,
loss + component curves (`ce`, `concept`, `aux_lb`), the generation gallery,
and the bench table alongside the levencode experiments.
