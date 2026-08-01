"""Precompute teacher latents (frozen GRM-3.2-Turf) for the stage-5 demo run.

Streams the data mix, tokenizes with the shared LFM2 tokenizer, chunks each
sample into nested fine/coarse units (heuristic semantic boundaries; optional
tree-sitter), runs ONE teacher forward per sample with CALM-style input-token
masking (dropout_tokens), pools hidden states per chunk at both granularities,
L2-normalizes, and writes the memmap store that stage-5 training consumes.

Usage:
    python scripts/precompute_latents.py --config configs/stage5_latent.yaml \
        --samples 12000 --out runs/precompute/demo
    # small smoke run:
    python scripts/precompute_latents.py --config configs/stage5_latent.yaml \
        --samples 32 --out runs/precompute/demo_smoke

The teacher is only loaded here (and optionally at bench time for
cycle-consistency); training never touches it — that is what makes the
~2h demo run feasible on a single 5090."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from levencode.config import cfg_get, load_config
from levencode.data.mix import build_mixture, extract_code
from levencode.latent.chunker import HierarchicalSpec, LevelSpec, hierarchical_spans
from levencode.latent.teacher import PrecomputedLatents, TeacherExtractor, load_teacher
from levencode.model.backbone import load_tokenizer_bundle


def sample_text(sample: dict) -> str | None:
    if "text" in sample:
        return sample["text"]
    msgs = sample.get("messages")
    if msgs and msgs[-1].get("role") == "assistant":
        return extract_code(str(msgs[-1].get("content", ""))) or str(msgs[-1].get("content", ""))
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage5_latent.yaml")
    ap.add_argument("--samples", type=int, default=12000)
    ap.add_argument("--out", default=None, help="store dir (default: latent.store from config)")
    ap.add_argument("--max_len", type=int, default=512, help="tokens per sample (ctx + chunks)")
    ap.add_argument("--ctx_tokens", type=int, default=64, help="context prefix tokens kept per sample")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="bf16")
    args = ap.parse_args()

    cfg = load_config(args.config)
    teacher_repo = cfg_get(cfg, "latent.teacher")
    out_dir = args.out or cfg_get(cfg, "latent.store")
    levels_cfg = cfg_get(cfg, "latent.levels", None)
    spec = HierarchicalSpec(
        levels=[LevelSpec(name="coarse", tokens_per_chunk=32), LevelSpec(name="fine", tokens_per_chunk=8)]
    )

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    print(f"[precompute] loading teacher {teacher_repo} on {device}", flush=True)
    model, tok = load_teacher(teacher_repo, device=device, dtype=dtype)
    model.eval()
    extractor = TeacherExtractor(
        model, tok,
        dropout_tokens=float(cfg_get(cfg, "latent.dropout_tokens", 0.15)),
        pool=cfg_get(cfg, "latent.pool", "mean"),
    )

    print("[precompute] loading student tokenizer (LFM2 shared vocab)", flush=True)
    bundle = load_tokenizer_bundle(cfg_get(cfg, "model.repo_id"))
    extractor._check_vocab(bundle)

    stream = build_mixture(cfg["data"], int(cfg_get(cfg, "run.seed", 1337)))

    examples = []
    stats = {"fine_sizes": [], "coarse_sizes": [], "n_chunks_used": 0, "masked_tokens": 0}
    t0 = time.perf_counter()
    got = 0
    while got < args.samples:
        sample = next(stream, None)
        if sample is None:
            break
        text = sample_text(sample)
        if not text:
            continue
        ids = bundle.encode(text)
        if len(ids) < 64:
            continue
        ids = ids[: args.max_len]
        # context prefix: tokens before the first chunk (at least ctx_tokens)
        ctx_tokens = min(args.ctx_tokens, max(len(ids) // 4, 16))
        body = ids[ctx_tokens:]
        if not body:
            continue
        body_spans = hierarchical_spans(body, spec, bundle=bundle, rng=None)
        c_spans, f_spans = body_spans[0], body_spans[1]
        ctx_ids = ids[:ctx_tokens]
        f_off = ctx_tokens
        fine_spans_rel = [(s + f_off, e + f_off) for s, e in f_spans]
        c_spans_abs = [(s + f_off, e + f_off) for s, e in c_spans]
        coarse_of_fine = []
        for fs in f_spans:
            for ci, (cs, ce) in enumerate(c_spans):
                if fs[1] <= ce:
                    coarse_of_fine.append(ci)
                    break
        if not f_spans or not c_spans:
            continue

        masked = extractor.masked_ids(ids, bundle.mask_id)
        h = extractor.hiddens(masked, device)
        z_f = extractor.pooled_for_spans(ids, h, fine_spans_rel)
        z_c = extractor.pooled_for_spans(ids, h, c_spans_abs)
        fine_tokens = [ids[s:e] for s, e in fine_spans_rel]

        from levencode.latent.teacher import LatentExample

        examples.append(
            LatentExample(
                ctx_ids=ctx_ids,
                fine_spans=fine_spans_rel,
                coarse_of_fine=coarse_of_fine,
                fine_tokens=fine_tokens,
                z_fine=z_f,
                z_coarse=z_c,
            )
        )
        stats["fine_sizes"].extend(e - s for s, e in fine_spans_rel)
        stats["coarse_sizes"].extend(e - s for s, e in c_spans)
        stats["masked_tokens"] += sum(1 for t in masked if t == bundle.mask_id)
        got += 1
        if got % 500 == 0:
            dt = time.perf_counter() - t0
            print(
                f"[precompute] {got}/{args.samples} samples, {dt:.0f}s "
                f"({got / max(dt, 1e-9):.1f} samples/s)",
                flush=True,
            )

    if not examples:
        print("no samples produced — check data mix / network", file=sys.stderr)
        sys.exit(1)

    manifest = {
        "teacher": teacher_repo,
        "latent_dim": int(model.config.hidden_size),
        "levels": [
            {"name": "coarse", "tokens_per_chunk": 32},
            {"name": "fine", "tokens_per_chunk": 8},
        ],
        "pool": extractor.pool,
        "dropout_tokens": extractor.dropout_tokens,
        "n_samples": len(examples),
        "fine_chunk_mean": sum(stats["fine_sizes"]) / max(len(stats["fine_sizes"]), 1),
        "coarse_chunk_mean": sum(stats["coarse_sizes"]) / max(len(stats["coarse_sizes"]), 1),
        "masked_token_rate": stats["masked_tokens"] / max(sum(len(e.ctx_ids) + sum(len(t) for t in e.fine_tokens) for e in examples), 1),
    }
    store = PrecomputedLatents(out_dir)
    store.write(examples, manifest)
    print(
        f"[precompute] wrote {manifest['n_samples']} samples, "
        f"{manifest['n_fine']} fine / {manifest['n_coarse']} coarse chunks to {out_dir}",
        flush=True,
    )
    print(f"[precompute] fine chunk mean {manifest['fine_chunk_mean']:.1f} tokens, "
          f"coarse {manifest['coarse_chunk_mean']:.1f} tokens, "
          f"masked rate {manifest['masked_token_rate']:.3f}", flush=True)


if __name__ == "__main__":
    main()
