"""Routing-by-loop analysis on a Loom checkpoint.

    python scripts/loom_routing.py --config configs/loom_pretrain.yaml \
        --ckpt runs/loom/pretrain/ckpt/step_30000 --name step30k

Does the shared MoE core route differently on different loop iterations?
Add sources to get the domain positive control (strongly recommended — a null
result on loops is only meaningful if the same measurement can detect
specialization that we know should be there):

    ... --source heldout fineweb_edu code math

Results land in runs/<experiment>/<stage>/bench/routing_<name>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from levencode.config import apply_overrides, cfg_get, load_config  # noqa: E402


def _rows_for_source(source, cfg, bundle, seq_len, n_rows, seed):
    """`heldout` = WikiText-103 test; anything else = that entry of data.pretrain_mix."""
    if source == "heldout":
        from loom.bench import LoomBenchCtx, _heldout_rows

        ctx = LoomBenchCtx(None, bundle, cfg, None, False, False)
        return _heldout_rows(ctx, n_rows, seq_len)

    from loom.data import pretrain_stream

    mix = [m for m in cfg_get(cfg, "data.pretrain_mix", []) if m.get("name") == source]
    if not mix:
        names = [m.get("name") for m in cfg_get(cfg, "data.pretrain_mix", [])]
        raise SystemExit(f"unknown --source {source!r}; try 'heldout' or one of {names}")
    one = dict(cfg.get("data", {}))
    one["pretrain_mix"] = [{**mix[0], "weight": 1.0}]
    rows, stream = [], pretrain_stream(one, bundle, seq_len, seed)
    for ex in stream:
        rows.append(ex["input_ids"])
        if len(rows) >= n_rows:
            break
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True, help="checkpoint dir (config.json + model.pt)")
    ap.add_argument("--name", required=True, help="result name -> bench/routing_<name>.json")
    ap.add_argument("--source", nargs="+", default=["heldout"],
                    help="heldout (WikiText-103 test) and/or pretrain_mix names")
    ap.add_argument("--rows", type=int, default=64, help="sequences per source")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=None, help="default: model.max_seq_len")
    ap.add_argument("--device", default=None)
    ap.add_argument("--zero-loop-emb", action="store_true",
                    help="ablate loop_emb before measuring: separates routing that follows "
                         "the evolving state from routing driven by the per-loop bias vector")
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides key=value")
    args = ap.parse_args()

    cfg = load_config(args.config)
    apply_overrides(cfg, args.set)
    if args.device is not None:
        cfg.setdefault("run", {})["device"] = args.device

    import torch  # noqa: E402

    from levencode.model.backbone import load_tokenizer_bundle  # noqa: E402
    from levencode.train.state import RunDir  # noqa: E402
    from levencode.util import resolve_device  # noqa: E402
    from loom.model import LoomLM  # noqa: E402
    from loom.routing import (  # noqa: E402
        RoutingStats, capture_router_logits, cross_source_report,
        format_cross_source, format_report,
    )

    device = resolve_device(cfg_get(cfg, "run.device", "auto"))
    model = LoomLM.load(args.ckpt, device=device)
    model.eval()  # hooks assume one call per (loop, layer): no grad checkpointing
    if args.zero_loop_emb:
        model.loop_emb.data.zero_()
        print("[ablation] loop_emb zeroed -- residual cross-loop routing is state-driven")
    mcfg = model.cfg
    bundle = load_tokenizer_bundle(cfg_get(cfg, "model.tokenizer_repo"))
    seq_len = args.seq_len or int(cfg_get(cfg, "model.max_seq_len", 1024))
    seed = int(cfg_get(cfg, "run.seed", 1337))

    reps: dict[str, dict] = {}
    for source in args.source:
        rows = _rows_for_source(source, cfg, bundle, seq_len, args.rows, seed)
        stats = RoutingStats(mcfg.n_loops, mcfg.core_layers, mcfg.n_experts, mcfg.top_k, seed=seed)
        with torch.no_grad():
            for i in range(0, len(rows), args.batch_size):
                x = torch.tensor(rows[i : i + args.batch_size], dtype=torch.long, device=device)
                with capture_router_logits(model) as store:
                    model(x)  # concepts=None: unguided pass, FiLM is a no-op in pretrain
                stats.update(store, input_ids=x)
        reps[source] = stats.report()
        print(f"\n===== source: {source} ({len(rows)} rows x {seq_len} tok) =====")
        print(format_report(reps[source]))
        for row in reps[source]["top_reroute_tokens"][:8]:
            row["token"] = bundle.decode([row["token_id"]])
        top = ", ".join(
            f"{r['token']!r}:{r['reroute_rate']:.2f}" for r in reps[source]["top_reroute_tokens"][:8]
        )
        if top:
            print(f"  most re-routed tokens (loop 0 -> {mcfg.n_loops - 1}): {top}")

    out = {
        "meta": {
            "ckpt": args.ckpt, "rows": args.rows, "seq_len": seq_len,
            "zero_loop_emb": args.zero_loop_emb,
        },
        "per_source": reps,
    }
    if len(reps) > 1:
        control = cross_source_report(reps)
        out["cross_source"] = control
        print(format_cross_source(control))

    run = RunDir(
        Path(cfg_get(cfg, "run.dir", "runs"))
        / cfg_get(cfg, "run.experiment", "loom")
        / cfg.get("stage", "pretrain")
    )
    dest = run.root / "bench" / f"routing_{args.name}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nsaved {dest}")


if __name__ == "__main__":
    main()
