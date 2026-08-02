"""Is loop_emb earning its keep?

Zeroing loop_emb left routing statistically unchanged (<1% shift on an effect
2400x above the noise floor). Two readings: the router ignores it, or it never
learned a useful magnitude anywhere in the model. This separates them.

Measures:
  1. RMS of each loop embedding against the adapter output it is added to
     (`u = adapter([s_r; e]) + loop_emb[r]`) -- is it even at a scale that
     could matter?
  2. Pairwise cosine between loop embeddings -- did the loops learn distinct
     directions, or collapse onto one?
  3. Held-out CE with loop_emb intact vs zeroed -- the functional test. If CE
     is unchanged, per-loop depth conditioning is vestigial and the staging
     seen in routing comes from state evolution alone.

    python scripts/loom_loopemb.py --config configs/loom_pretrain.yaml \
        --ckpt runs/loom/pretrain/ckpt/step_30000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from levencode.config import cfg_get, load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.device is not None:
        cfg.setdefault("run", {})["device"] = args.device

    import torch  # noqa: E402

    from levencode.model.backbone import load_tokenizer_bundle  # noqa: E402
    from levencode.util import resolve_device  # noqa: E402
    from loom.bench import LoomBenchCtx, _heldout_rows  # noqa: E402
    from loom.model import LoomLM  # noqa: E402

    device = resolve_device(cfg_get(cfg, "run.device", "auto"))
    model = LoomLM.load(args.ckpt, device=device)
    model.eval()
    bundle = load_tokenizer_bundle(cfg_get(cfg, "model.tokenizer_repo"))
    seq_len = int(cfg_get(cfg, "model.max_seq_len", 1024))
    rows = _heldout_rows(LoomBenchCtx(None, bundle, cfg, None, False, False), args.rows, seq_len)

    emb = model.loop_emb.detach().float()          # [R, D]
    R = emb.shape[0]

    # adapter output scale, one entry per loop
    adapter_rms: list[float] = []

    def hook(_m, _i, out):
        adapter_rms.append(float(out.detach().float().pow(2).mean().sqrt()))

    h = model.adapter.register_forward_hook(hook)

    def heldout_ce() -> float:
        tot, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(rows), args.batch_size):
                x = torch.tensor(rows[i : i + args.batch_size], dtype=torch.long, device=device)
                tot += float(model(x, labels=x)["ce"])
                n += 1
        return tot / max(n, 1)

    ce_on = heldout_ce()
    per_loop_adapter = [
        sum(adapter_rms[r::R]) / max(len(adapter_rms[r::R]), 1) for r in range(R)
    ]
    h.remove()

    saved = model.loop_emb.data.clone()
    model.loop_emb.data.zero_()
    ce_off = heldout_ce()
    model.loop_emb.data.copy_(saved)

    print(f"loop_emb: {R} x {emb.shape[1]}")
    print("  loop   RMS(loop_emb)   RMS(adapter out)   ratio")
    for r in range(R):
        e_rms = float(emb[r].pow(2).mean().sqrt())
        a_rms = per_loop_adapter[r]
        print(f"  {r:<6} {e_rms:>12.5f}   {a_rms:>16.5f}   {e_rms / max(a_rms, 1e-9):>6.3f}")

    print("\n  pairwise cosine between loop embeddings:")
    for r in range(R):
        for r2 in range(r + 1, R):
            cos = float(torch.nn.functional.cosine_similarity(emb[r], emb[r2], dim=0))
            print(f"    {r}-{r2}: {cos:+.4f}")

    d = ce_off - ce_on
    print(f"\n  held-out CE  intact={ce_on:.4f}  zeroed={ce_off:.4f}  delta={d:+.4f} nats"
          f"  ({100 * d / max(ce_on, 1e-9):+.2f}%)")
    ratios = [float(emb[r].pow(2).mean().sqrt()) / max(per_loop_adapter[r], 1e-9) for r in range(R)]
    drowned = max(ratios) < 0.01
    if abs(d) < 0.01 and drowned:
        verdict = (
            f"loop_emb is inert AND drowned (max ratio {max(ratios):.4f}): it sits ~"
            f"{1 / max(max(ratios), 1e-9):.0f}x below the signal it is added to, so its own "
            "gradient cannot bootstrap it. This says the depth tag was never operative -- NOT "
            "that depth conditioning is unnecessary. Any other fixed-scale additive injection "
            "at this site (e.g. FiLM beta) has the same problem."
        )
    elif abs(d) < 0.01:
        verdict = ("loop_emb is inert despite being at a workable scale -- the loops really do "
                   "differentiate from state alone")
    else:
        verdict = ("loop_emb matters functionally even though it barely moves routing -- "
                   "it is acting somewhere other than the router")
    print(f"  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
