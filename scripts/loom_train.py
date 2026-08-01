"""Train a Loom stage: python scripts/loom_train.py --config configs/loom_pretrain.yaml"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from levencode.config import apply_overrides, load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=None, help="override train.total_steps")
    ap.add_argument("--device", default=None, help="override run.device")
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides key=value")
    args = ap.parse_args()

    cfg = load_config(args.config)
    apply_overrides(cfg, args.set)
    if args.steps is not None:
        cfg.setdefault("train", {})["total_steps"] = args.steps
    if args.device is not None:
        cfg.setdefault("run", {})["device"] = args.device

    from loom.train import LoomTrainer  # noqa: E402  (after sys.path insert)

    LoomTrainer(cfg).train()


if __name__ == "__main__":
    main()
