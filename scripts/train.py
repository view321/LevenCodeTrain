"""Train one stage:  python scripts/train.py --config configs/stage1_sft.yaml"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Reduce fragmentation-driven OOMs; must be set before CUDA initializes.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from levencode.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=None, help="override train.total_steps")
    ap.add_argument("--device", default=None, help="override run.device (cuda/cpu)")
    ap.add_argument("--experiment", default=None, help="override run.experiment")
    ap.add_argument("--no-bench", action="store_true", help="skip the stage-end benchmark")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.steps is not None:
        cfg["train"]["total_steps"] = args.steps
    if args.device:
        cfg.setdefault("run", {})["device"] = args.device
    if args.experiment:
        cfg.setdefault("run", {})["experiment"] = args.experiment
    if args.no_bench:
        cfg.setdefault("bench", {})["at_stage_end"] = False

    if cfg["stage"] == "grpo":
        from levencode.train.grpo import run_grpo

        run_grpo(cfg)
    else:
        from levencode.train.trainer import Trainer

        Trainer(cfg).train()


if __name__ == "__main__":
    main()
