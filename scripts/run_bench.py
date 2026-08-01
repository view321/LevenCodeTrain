"""Run the benchmark against a checkpoint (or the raw pretrained backbone).

  python scripts/run_bench.py --config configs/base.yaml --name baseline
  python scripts/run_bench.py --config configs/stage2_edit.yaml \
      --ckpt runs/levencode/edit/ckpt/final --name edit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from levencode.bench.benchmark import run_benchmark
from levencode.config import apply_overrides, cfg_get, load_config
from levencode.model.backbone import load_tokenizer_bundle
from levencode.model.editor import build_editor
from levencode.train.state import RunDir
from levencode.train.trainer import latent_kwargs_from_config
from levencode.util import resolve_device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None, help="checkpoint dir; defaults to the HF repo backbone")
    ap.add_argument("--name", required=True, help="result name, becomes bench/<name>.json")
    ap.add_argument("--only", nargs="*", default=None, help="subset of tasks to run")
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--set", nargs="*", default=[], metavar="KEY=VALUE",
        help="config overrides, e.g. --set bench.mbpp_n=257 bench.gsm8k_n=300",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    apply_overrides(cfg, args.set)
    if args.device:
        cfg.setdefault("run", {})["device"] = args.device
    device = resolve_device(cfg_get(cfg, "run.device", "auto"))
    bundle = load_tokenizer_bundle(cfg_get(cfg, "model.repo_id"))
    has_latent = bool(args.ckpt and (Path(args.ckpt) / "latent.pt").exists())
    editor = build_editor(
        args.ckpt or cfg_get(cfg, "model.repo_id"),
        insert_max=int(cfg_get(cfg, "model.insert_max", 8)),
        device=device,
        dtype=torch.float32,
        with_latent=has_latent,
        latent_kwargs=latent_kwargs_from_config(cfg),
    )
    results = run_benchmark(editor, bundle, cfg, device, only=args.only)

    runs_root = Path(cfg_get(cfg, "run.runs_dir", "runs")) / cfg_get(cfg, "run.experiment", "levencode")
    stage = cfg.get("stage", "bench")
    run = RunDir(runs_root / stage)
    run.save_bench(args.name, results)
    print(f"saved bench to {run.root / 'bench' / (args.name + '.json')}")
    for task, metrics in results.items():
        print(task, metrics)


if __name__ == "__main__":
    main()
