# Levencode

A feasibility probe: can a small **block-diffusion** language model with
**Levenshtein edit operations** and a **JEPA auxiliary objective** be
competitive with a same-size autoregressive LLM on code — especially on
repair/infill — while decoding faster?

Backbone: [LiquidAI/LFM2.5-Encoder-350M-Diffusion](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M-Diffusion)
(354M bidirectional encoder, pretrained MLM + masked-diffusion instruct tune).
On top of it, this repo adds:

- **Block-canvas sampling** — generate 20–100-token canvases by iterative
  unmasking, then append the next canvas (DiffusionGemma-style), with text-level
  `[/Answer]` stop plus a trained `<|im_end|>` stop.
- **Levenshtein edit heads** — a per-token *delete* head and a per-gap
  *insertion-count* head; the pretrained MLM head fills inserted placeholders.
  Supervision comes free from a corruption engine that tracks provenance
  (no alignment DP needed) — see `src/levencode/data/corruption.py`.
- **JEPA auxiliary** — EMA target encoder + small predictor regressing clean-code
  latents at masked positions. Run as an ablation (stage 3 vs stage 2).
- **GRPO/RLVR (experimental)** — policy gradient over exact edit-action
  log-probs on the repair task with dense rewards (Levenshtein reduction +
  syntax validity + exact match). See `src/levencode/train/grpo.py`.

## Pipeline stages

| Stage | Config | Init from | What it does |
|---|---|---|---|
| 1 `sft` | `configs/stage1_sft.yaml` | HF repo | Block-pattern masked-diffusion SFT on the data mix; teaches canvas generation + EOS |
| 2 `edit` | `configs/stage2_edit.yaml` | stage 1 | Adds edit heads, trains on synthetic corruption (+ SFT retention) |
| 2b `edit_rollin` | `configs/stage2b_rollin.yaml` | stage 2 | **Roll-in**: edit labels from aligning the model's own outputs against references |
| 3 `jepa` | `configs/stage3_jepa.yaml` | stage 1 | Same as stage 2 **plus** JEPA loss — clean ablation vs stage 2 |
| 4 `grpo` | `configs/stage4_grpo.yaml` | stage 2 | Experimental RLVR on repair |
| 5 `latent` | `configs/stage5_latent.yaml` | stage 2 | **Multi-granularity latent JEPA** on frozen teacher latents (GRM-3.2-Turf, precomputed — never loaded in training): coarse plan at 1/32 token rate (RVQ code anchors + AR prior), fine level at 1/8 rate conditioned on the quantized coarse latent, continuous detail from an energy-trained residual head, CFG-guided sampling, plus a decodability adapter (variational + KL-clip + dual dropout) decoding latents to tokens through the tied LM head |

Roll-in (stage 2b) attacks the two gaps synthetic corruption can't: the
localization gap (heads must fix states the *model* visits, not just random
noise) and the draft-distribution mismatch (self-authored decoherence). A
buffer regenerates hypothesis/reference pairs with the current weights every
`rollin.refresh_every` steps — half imperfect self-repairs, half stochastic
self-fills — recovers edit labels via a Levenshtein alignment backtrace
(`data/alignment.py`), and serves them through the standard edit views. Watch
`rollin_edit_mass` in the metrics: it is the mean edit distance of the model's
own outputs from the reference, and should FALL as roll-in training works.

Stage 5 is the distillation idea from the prompt: the teacher's pooled hidden
states (L2-normalized, per-chunk at two granularities, CALM-style token-mask
dropout) are written once by `scripts/precompute_latents.py` into a memmap
store, and training consumes only that store — so a ~2h demo run fits on one
5090 while the student still inherits the teacher's semantic/context-aware
properties. Generation runs the latent pipeline first (AR plan codes → CFG
energy residuals → adapter plan logits) and then fills tokens with the
block-diffusion filler *plus* the plan prior; committed chunks are re-encoded
and rejected on cycle-consistency drift. Evaluation adds **BrierLM** (CALM
Sec. 4): a likelihood-free, sample-based n-gram metric on teacher-forced
plan logits that correlates with CE (-0.966) where mode-averaged metrics hide.

Data mix (streamed from HF, weights in `configs/base.yaml`): smoltalk (chat +
reasoning) 30%, Magicoder-OSS-Instruct (code) 25%, MetaMathQA (math) 20%,
codeparrot-clean (raw Python, feeds infill + edit training) 25%.

Each stage ends with an automatic **benchmark**: chat masked-CE, ARC-Easy
(likelihood MC), GSM8K EM, MBPP pass@1 (sandboxed execution), plus the
signature evals — self-located **repair** (exact / syntax-valid / Levenshtein
reduction), **infill**, **BrierLM** (stage-5 runs), and generation speed.
Results land in `runs/<experiment>/<stage>/bench/` and render in the WebUI.

## Setup on the training box (RTX 5090)

Requires Python 3.11–3.13 and a Blackwell-capable PyTorch (CUDA 12.8+).

```bash
git clone <this repo> Levencode && cd Levencode   # or copy the folder
python -m venv .venv
.venv/Scripts/activate            # Windows; on Linux: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e .[dev]
```

Sanity-check everything (CPU-ok, ~3 min; downloads the 1.4GB model on first run):

```bash
python scripts/smoke_test.py
python -m pytest tests/ -q
```

## Training

```bash
# stage by stage
python scripts/train.py --config configs/stage1_sft.yaml
python scripts/train.py --config configs/stage2_edit.yaml
python scripts/train.py --config configs/stage3_jepa.yaml

# stage 5: latent JEPA on frozen teacher latents
python scripts/precompute_latents.py --config configs/stage5_latent.yaml
python scripts/train.py --config configs/stage5_latent.yaml

# or the whole ladder (skips stages whose final checkpoint exists)
python scripts/run_all.py
```

The WebUI (start it in a second terminal, keep it running during training):

```bash
python scripts/serve.py --runs-dir runs --port 7860
```

Then open http://127.0.0.1:7860 — stage progress/ETA, loss + component curves,
throughput, the benchmark table across stages, and a generation gallery
refreshed every `sample_every` steps.

Benchmark a checkpoint (or the raw backbone) manually:

```bash
python scripts/run_bench.py --config configs/base.yaml --name backbone_baseline
python scripts/run_bench.py --config configs/stage2_edit.yaml --ckpt runs/levencode/edit/ckpt/final --name edit_final
```

### Expected wall-clock on the 5090 (defaults)

Defaults: micro-batch 8 × grad-accum 8 × 1024 tokens ≈ 65k tokens/step.

| Stage | Steps | ~Tokens | ~Time |
|---|---|---|---|
| sft | 3000 | ~200M | 3–4 h |
| edit | 2000 | ~130M (3 views/edit batch) | 3–4 h |
| jepa | 2000 | same + EMA/target passes | 4–5 h |
| latent | 1500 | ~100M + precompute pass | 1.5–2 h (+ ~1h precompute) |
| bench per stage | — | — | 20–40 min |

That fits the "couple of days" budget with room for a config iteration. Raise
`total_steps` if curves are still dropping. If VRAM allows (watch the WebUI
tok/s), raise `micro_batch_size` before anything else.

Memory notes: the trainer backwards each edit view separately and computes
LM-head logits only at supervised positions, so edit/jepa stages fit
comfortably in 32GB at the defaults. Edit micro-batches default to half of
`micro_batch_size` (three view forwards each) — override with
`train.edit_micro_batch_size`. If you still hit OOM, lower that first;
`scripts/train.py` also enables `expandable_segments` to curb fragmentation.

## Layout

```
configs/            base + stage configs (`_extends` inheritance)
src/levencode/
  data/             corruption engine, collators, HF streaming mix
  model/            backbone loader, edit heads, editor, JEPA
  sampling/         block-canvas sampler, edit (repair) sampler
  train/            trainer, losses, run-state writer, GRPO (experimental)
  bench/            tasks, sandbox, fixtures, orchestrator
  webui/            FastAPI server + static dashboard
scripts/            train / run_all / run_bench / serve / smoke_test
tests/              58 unit + integration tests (CPU, tiny real-arch model)
```

## Notes and caveats

- **Streaming data needs network** at train time (HF datasets). First run also
  downloads the model and benchmark datasets into the HF cache.
- **MBPP/repair execute model-generated code** in a subprocess (`-I`, timeout,
  temp cwd). That guards against accidents, not adversaries.
- **GRPO stage is experimental**: exact-logprob REINFORCE with group baseline,
  fill/delete/insert actions all on-policy. Run it only after stage 2 shows a
  reasonable repair signal.
- The pretrained model closes answers with the *text* marker `[/Answer]`
  (no stop token). Stage 1 teaches `<|im_end|>`; until then samplers rely on
  the text stop — both are wired in.
- Tests: `pytest -q` runs everything; `-m "not network"` skips the ones needing
  the HF cache (tiny-model architecture tests).
