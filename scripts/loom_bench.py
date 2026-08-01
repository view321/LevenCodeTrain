"""Bench a Loom checkpoint:
    python scripts/loom_bench.py --config configs/loom_sft.yaml \
        --ckpt runs/loom/sft/ckpt/final --name sft_final
Results land in runs/<experiment>/<stage>/bench/<name>.json (WebUI-visible)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from levencode.config import apply_overrides, cfg_get, load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True, help="checkpoint dir (config.json + model.pt)")
    ap.add_argument("--name", required=True, help="result name -> bench/<name>.json")
    ap.add_argument("--only", nargs="*", default=None, help="subset of tasks")
    ap.add_argument("--device", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides key=value")
    args = ap.parse_args()

    cfg = load_config(args.config)
    apply_overrides(cfg, args.set)
    if args.device is not None:
        cfg.setdefault("run", {})["device"] = args.device

    from levencode.model.backbone import load_tokenizer_bundle  # noqa: E402
    from levencode.train.state import RunDir  # noqa: E402
    from levencode.util import resolve_device  # noqa: E402
    from loom.bench import run_loom_bench  # noqa: E402
    from loom.model import LoomLM  # noqa: E402

    device = resolve_device(cfg_get(cfg, "run.device", "auto"))
    model = LoomLM.load(args.ckpt, device=device)
    model.eval()
    bundle = load_tokenizer_bundle(cfg_get(cfg, "model.tokenizer_repo"))
    results = run_loom_bench(model, bundle, cfg, device, only=args.only)

    run = RunDir(Path(cfg_get(cfg, "run.dir", "runs")) / cfg_get(cfg, "run.experiment", "loom") / cfg.get("stage", "bench"))
    run.save_bench(args.name, results)
    print(f"saved bench to {run.root / 'bench' / (args.name + '.json')}")


if __name__ == "__main__":
    main()
