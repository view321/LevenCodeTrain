"""Run the full pipeline: stage1 (sft) -> stage2 (edit) -> stage3 (jepa).

Each stage benchmarks at its end; results land in runs/<experiment>/<stage>/
and show up in the WebUI. Stages whose final checkpoint already exists are
skipped unless --force."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from levencode.config import cfg_get, load_config

STAGES = ["stage1_sft.yaml", "stage2_edit.yaml", "stage3_jepa.yaml"]


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
        print(f"[run_all] === stage {stage} ===")
        from levencode.train.trainer import Trainer

        Trainer(cfg).train()


if __name__ == "__main__":
    main()
