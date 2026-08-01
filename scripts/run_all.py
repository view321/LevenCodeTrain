"""Run the full pipeline: stage1 (sft) -> stage2 (edit) -> stage3 (jepa)
-> stage5 (latent JEPA on precomputed teacher latents).

Each stage benchmarks at its end; results land in runs/<experiment>/<stage>/
and show up in the WebUI. Stages whose final checkpoint already exists are
skipped unless --force. Stage 5 needs the latent store built first by
scripts/precompute_latents.py (the runner prints a reminder and skips it)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from levencode.config import cfg_get, load_config

STAGES = ["stage1_sft.yaml", "stage2_edit.yaml", "stage3_jepa.yaml", "stage5_latent.yaml"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs-dir", default="configs")
    ap.add_argument("--experiment", default=None)
    ap.add_argument("--force", action="store_true", help="re-run stages even if completed")
    args = ap.parse_args()

    for name in STAGES:
        cfg = load_config(Path(args.configs_dir) / name)
        if args.experiment:
            cfg.setdefault("run", {})["experiment"] = args.experiment
        exp = cfg_get(cfg, "run.experiment", "levencode")
        stage = cfg["stage"]
        final = Path(cfg_get(cfg, "run.runs_dir", "runs")) / exp / stage / "ckpt" / "final"
        if final.exists() and not args.force:
            print(f"[run_all] {stage}: final checkpoint exists, skipping")
            continue
        if stage == "latent" and not Path(cfg_get(cfg, "latent.store")).exists():
            print(
                "[run_all] latent: store not found — run first:\n"
                "  python scripts/precompute_latents.py --config configs/stage5_latent.yaml"
            )
            continue
        print(f"[run_all] === stage {stage} ===")
        from levencode.train.trainer import Trainer

        Trainer(cfg).train()


if __name__ == "__main__":
    main()
