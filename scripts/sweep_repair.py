"""Sweep repair-sampler knobs on an existing checkpoint — no retraining.

The oracle-vs-self-located gap tells you localization is the bottleneck; this
finds how much of it inference knobs recover:

  python scripts/sweep_repair.py --config configs/stage2_edit.yaml \
      --ckpt runs/levencode/edit/ckpt/final

Grid defaults: delete_threshold x ins_zero_penalty x rounds = 3 x 3 x 2.
The oracle metric is sampler-independent and computed once up top.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from levencode.bench.tasks import BenchCtx, task_repair
from levencode.config import cfg_get, load_config
from levencode.model.backbone import load_tokenizer_bundle
from levencode.util import resolve_device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.4, 0.5])
    ap.add_argument("--penalties", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    ap.add_argument("--rounds", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--n", type=int, default=None, help="override bench.repair_n")
    ap.add_argument("--device", default=None)
    ap.add_argument("--tiny", action="store_true", help="random tiny model (debug only)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.device:
        cfg.setdefault("run", {})["device"] = args.device
    if args.n:
        cfg.setdefault("bench", {})["repair_n"] = args.n
    device = resolve_device(cfg_get(cfg, "run.device", "auto"))
    bundle = load_tokenizer_bundle(cfg_get(cfg, "model.repo_id"))

    if args.tiny:
        from levencode.model.backbone import tiny_backbone
        from levencode.model.editor import LevencodeEditor

        torch.manual_seed(0)
        editor = LevencodeEditor(tiny_backbone(), insert_max=int(cfg_get(cfg, "model.insert_max", 8)))
        editor = editor.to(device)
    else:
        from levencode.model.editor import build_editor

        editor = build_editor(
            args.ckpt or cfg_get(cfg, "model.repo_id"),
            insert_max=int(cfg_get(cfg, "model.insert_max", 8)),
            device=device,
            dtype=torch.float32,
        )
    editor.eval()

    # Oracle reference, once (sampler-independent).
    base_cfg = copy.deepcopy(cfg)
    base_res = task_repair(BenchCtx(editor=editor, bundle=bundle, cfg=base_cfg, device=device))
    print(
        f"\noracle (fill with true locations): exact={base_res.get('repair_oracle_exact', 0):.3f} "
        f"syntax={base_res.get('repair_oracle_syntax_valid', 0):.3f}   n={base_res.get('n')}\n"
    )

    rows = []
    for thr in args.thresholds:
        for pen in args.penalties:
            for rnd in args.rounds:
                c = copy.deepcopy(cfg)
                c.setdefault("edit_sampler", {}).update(
                    {"delete_threshold": thr, "ins_zero_penalty": pen, "rounds": rnd}
                )
                c.setdefault("bench", {})["repair_oracle"] = False
                res = task_repair(BenchCtx(editor=editor, bundle=bundle, cfg=c, device=device))
                rows.append((thr, pen, rnd, res))
                r = res
                print(
                    f"thr={thr:.2f} pen={pen:.2f} rounds={rnd}  ->  "
                    f"exact={r['repair_exact']:.3f} syntax={r['repair_syntax_valid']:.3f} "
                    f"lev_red={r['repair_lev_reduction']:.3f} len={r['repair_len_ratio']:.3f} "
                    f"noop={r['repair_noop_rate']:.2f} del={r['repair_mean_deleted']:.1f} "
                    f"ins={r['repair_mean_inserted']:.1f}",
                    flush=True,
                )

    rows.sort(key=lambda x: (x[3]["repair_exact"], x[3]["repair_lev_reduction"]), reverse=True)
    print("\n=== best by exact match, then lev-reduction ===")
    print(f"{'thr':>5} {'pen':>5} {'rnd':>4} | {'exact':>6} {'syntax':>7} {'lev_red':>8} {'len':>6}")
    for thr, pen, rnd, r in rows[:8]:
        print(
            f"{thr:>5.2f} {pen:>5.2f} {rnd:>4d} | {r['repair_exact']:>6.3f} "
            f"{r['repair_syntax_valid']:>7.3f} {r['repair_lev_reduction']:>8.3f} "
            f"{r['repair_len_ratio']:>6.3f}"
        )
    best = rows[0]
    print(
        f"\nbest: delete_threshold={best[0]}, ins_zero_penalty={best[1]}, rounds={best[2]}"
        f"  -> put these under edit_sampler: in your stage config"
    )


if __name__ == "__main__":
    main()
