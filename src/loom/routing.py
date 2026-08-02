"""Routing-by-loop analysis for Loom's looped MoE core.

The core runs the SAME MoE weights R times (`s_{r+1} = core(adapter([s_r; e])
+ loop_emb_r)`). Nothing forces the router to behave differently on different
loops, so whether it does is an empirical question — and the answer says what
recurrence is buying.

Two things get measured, because they dissociate:

  (a) DEPTH SPECIALIZATION — does the aggregate expert-usage distribution
      differ across loops? JS divergence between per-loop usage histograms,
      compared against a within-loop half-split noise floor. JS >> null means
      loop r systematically prefers a different expert mixture than loop r'.

  (b) PER-TOKEN RE-ROUTING — for the same token, does the chosen expert change
      between loops? Chance-adjusted top-1 agreement (kappa-style) plus top-k
      set Jaccard. kappa ~ 1 means every token routes identically on every
      loop; low kappa means the loop state moved enough to re-route.

Readings:
  kappa ~ 1, JS ~ null   loops re-run identical routing — the core is extra
                         compute inside a fixed subnetwork (iterative
                         refinement / ensembling), not a staged pipeline.
  kappa low, JS ~ null   tokens re-route per loop, but no loop-level
                         preference: the loop changes *which* token goes
                         where, not *what the stage is for*.
  JS >> null             genuine depth specialization — loops act as stages.

Chance baselines matter here: the load-balance loss actively pushes usage
toward uniform, so raw agreement is high by construction. Everything below is
reported against its null.

The always-on shared expert is unrouted and never appears in these stats.
"""

from __future__ import annotations

import math
from contextlib import contextmanager

import torch

_EPS = 1e-9


@contextmanager
def capture_router_logits(model):
    """Collect router logits per core layer, one entry per loop iteration.

    `LoomLM.forward` runs `for r in range(n_loops): for blk in self.core`, so
    each core block's hook fires exactly n_loops times per forward and the
    call index within a layer IS the loop index. Keying by layer (rather than
    by global call order) keeps that mapping robust to prelude/coda changes.

    Requires eval mode / no grad checkpointing so each block runs once per loop.
    """
    store: dict[int, list[torch.Tensor]] = {l: [] for l in range(len(model.core))}
    handles = []

    def make_hook(layer: int):
        def hook(_mod, _inp, out):
            store[layer].append(out.detach().float())
        return hook

    for l, blk in enumerate(model.core):
        handles.append(blk.moe.router.register_forward_hook(make_hook(l)))
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


def _js_bits(p: torch.Tensor, q: torch.Tensor) -> float:
    """Jensen-Shannon divergence in bits; 0 = identical, 1 = disjoint support."""
    p = p / p.sum().clamp_min(_EPS)
    q = q / q.sum().clamp_min(_EPS)
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * ((a + _EPS).log2() - (b + _EPS).log2())).sum()
    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


class RoutingStats:
    """Streaming accumulator — nothing per-token is retained across batches."""

    def __init__(self, n_loops: int, n_layers: int, n_experts: int, top_k: int, seed: int = 0):
        self.R, self.L, self.E, self.k = n_loops, n_layers, n_experts, top_k
        z = torch.zeros
        self.usage = z(n_loops, n_layers, n_experts)        # top-k slot membership
        self.usage_half = z(2, n_loops, n_layers, n_experts)  # for the noise floor
        self.top1 = z(n_loops, n_layers, n_experts)         # argmax expert
        self.ent_sum = z(n_loops, n_layers)                 # router entropy (bits)
        self.agree = z(n_layers, n_loops, n_loops)          # top-1 match counts
        self.jacc = z(n_layers, n_loops, n_loops)           # top-k Jaccard sum
        self.tokens = 0
        self.g = torch.Generator().manual_seed(seed)
        # qualitative: which token ids re-route between first and last loop
        self.flip: dict[int, list[float]] = {}

    @torch.no_grad()
    def update(self, store: dict[int, list[torch.Tensor]], input_ids: torch.Tensor | None = None) -> None:
        n_tok = None
        flip_acc = None
        for l, calls in store.items():
            if len(calls) != self.R:
                raise RuntimeError(
                    f"core layer {l} fired {len(calls)} times, expected n_loops={self.R}; "
                    "run the model in eval() with grad checkpointing off"
                )
            logits = torch.stack(calls, 0)                    # [R, N, E]
            probs = logits.softmax(-1)
            R, N, E = probs.shape
            n_tok = N
            _, topi = probs.topk(self.k, dim=-1)              # [R, N, k]
            memb = torch.zeros_like(probs).scatter_(2, topi, 1.0)  # [R, N, E]

            self.usage[:, l] += memb.sum(1).cpu()
            mask = (torch.rand(N, generator=self.g) < 0.5).to(probs.device)
            self.usage_half[0, :, l] += memb[:, mask].sum(1).cpu()
            self.usage_half[1, :, l] += memb[:, ~mask].sum(1).cpu()

            t1 = topi[:, :, 0]                                # [R, N]
            self.top1[:, l] += torch.zeros(R, N, E, device=probs.device).scatter_(
                2, t1.unsqueeze(-1), 1.0
            ).sum(1).cpu()

            ent = -(probs * (probs + _EPS).log2()).sum(-1)    # [R, N]
            self.ent_sum[:, l] += ent.sum(1).cpu()

            for r in range(R):
                for r2 in range(R):
                    self.agree[l, r, r2] += (t1[r] == t1[r2]).sum().cpu()
                    inter = (memb[r] * memb[r2]).sum(-1)      # [N]
                    self.jacc[l, r, r2] += (inter / (2 * self.k - inter)).sum().cpu()

            if input_ids is not None:
                d = (t1[0] != t1[-1]).float()                 # re-routed loop 0 -> loop R-1
                flip_acc = d if flip_acc is None else flip_acc + d

        if input_ids is not None and flip_acc is not None:
            flat = input_ids.reshape(-1).cpu()
            rate = (flip_acc / self.L).cpu()
            for tid, fr in zip(flat.tolist(), rate.tolist()):
                slot = self.flip.setdefault(tid, [0.0, 0])
                slot[0] += fr
                slot[1] += 1

        self.tokens += int(n_tok or 0)

    # ---------- metrics ----------

    def report(self) -> dict:
        R, L, E, k = self.R, self.L, self.E, self.k
        usage = self.usage / self.usage.sum(-1, keepdim=True).clamp_min(_EPS)     # [R,L,E]
        p1 = self.top1 / self.top1.sum(-1, keepdim=True).clamp_min(_EPS)          # [R,L,E]
        ent = self.ent_sum / max(self.tokens, 1)                                  # [R,L]

        layers = []
        for l in range(L):
            # (a) depth specialization: cross-loop JS vs within-loop noise floor.
            # The null compares two HALF-size samples, so it slightly
            # overestimates the floor -> conservative against finding an effect.
            cross = [_js_bits(usage[r, l], usage[r2, l]) for r in range(R) for r2 in range(r + 1, R)]
            null = [_js_bits(self.usage_half[0, r, l], self.usage_half[1, r, l]) for r in range(R)]
            js_cross = sum(cross) / max(len(cross), 1)
            js_null = sum(null) / max(len(null), 1)

            # (b) per-token re-routing: chance-adjusted agreement (kappa-style).
            # If a loop's top-1 marginal collapses onto one expert, chance
            # agreement is ~1 and kappa is undefined — report None rather than
            # a spurious 0.0 that reads as "at chance".
            agree, kappa, jacc = {}, {}, {}
            for r in range(R):
                for r2 in range(r + 1, R):
                    a = float(self.agree[l, r, r2]) / max(self.tokens, 1)
                    c = float((p1[r, l] * p1[r2, l]).sum())   # independent draws from marginals
                    agree[f"{r}-{r2}"] = a
                    kappa[f"{r}-{r2}"] = None if c > 0.999 else (a - c) / (1.0 - c)
                    jacc[f"{r}-{r2}"] = float(self.jacc[l, r, r2]) / max(self.tokens, 1)

            # descriptive: how much of expert identity is explained by loop index
            joint = p1[:, l] / R                               # [R,E], p(loop)=1/R
            pe = joint.sum(0)
            mi = float((joint * ((joint + _EPS) / (pe.unsqueeze(0) / R + _EPS)).log2()).sum())
            h_e = float(-(pe * (pe + _EPS).log2()).sum())

            u = usage[:, l]
            layers.append({
                "layer": l,
                "usage": u.tolist(),
                "entropy_bits": ent[:, l].tolist(),
                "js_cross_bits": js_cross,
                "js_null_bits": js_null,
                "js_ratio": js_cross / max(js_null, _EPS),
                "top1_agreement": agree,
                "top1_kappa": kappa,
                "topk_jaccard": jacc,
                "nmi_expert_loop": mi / max(h_e, _EPS),
                "dead_experts": int((u.mean(0) < 1.0 / (4 * E)).sum()),
                "usage_max": float(u.mean(0).max()),
                "usage_min": float(u.mean(0).min()),
            })

        ks = [v for lyr in layers for v in lyr["top1_kappa"].values() if v is not None]
        mean_kappa = sum(ks) / len(ks) if ks else None
        mean_ratio = sum(lyr["js_ratio"] for lyr in layers) / max(L, 1)
        mean_js = sum(lyr["js_cross_bits"] for lyr in layers) / max(L, 1)
        return {
            "meta": {
                "n_loops": R, "core_layers": L, "n_experts": E, "top_k": k,
                "tokens": self.tokens, "max_entropy_bits": math.log2(E),
            },
            "summary": {
                "mean_top1_kappa": mean_kappa,
                "mean_js_cross_bits": mean_js,
                "mean_js_ratio": mean_ratio,
                "verdict": _verdict(mean_kappa, mean_js, mean_ratio),
            },
            "layers": layers,
            "top_reroute_tokens": self._top_flips(),
        }

    def _top_flips(self, min_count: int = 20, n: int = 15) -> list[dict]:
        # Note: the count floor biases this toward frequent tokens (function
        # words, punctuation). It is a qualitative read, not a statistic.
        rows = [
            {"token_id": t, "reroute_rate": s / c, "count": c}
            for t, (s, c) in self.flip.items()
            if c >= min_count
        ]
        rows.sort(key=lambda r: -r["reroute_rate"])
        return rows[:n]


def _verdict(kappa: float | None, js_bits: float, js_ratio: float) -> str:
    # Both tests must pass: the ratio alone explodes when a near-deterministic
    # router drives the half-split null to ~0, so require the divergence to be
    # non-negligible in absolute terms too (max JS = 1 bit).
    depth = js_ratio > 2.0 and js_bits > 0.01
    if kappa is None:
        return ("top-1 marginal has collapsed onto one expert -- kappa undefined; "
                "read usage/JS and the balance stats instead")
    static = kappa > 0.8
    if depth and not static:
        return "staged: loops prefer different experts AND re-route tokens"
    if depth:
        return "depth-specialized but token-stable: loops shift the expert mixture globally"
    if static:
        return "no specialization: loops re-run near-identical routing (recurrence = extra compute, not stages)"
    return "re-routing without depth preference: loops move tokens between experts, no per-loop expert identity"


def cross_source_report(reps: dict[str, dict]) -> dict:
    """Positive control: do experts specialize by DOMAIN?

    Without this, a small cross-loop JS is uninterpretable — it could mean
    "loops don't specialize" or "this router doesn't specialize by anything
    yet at 2.6B tokens". Domain specialization is the effect we EXPECT a
    working MoE router to show, measured on exactly the same scale, so it
    calibrates the instrument. If cross-source JS >> cross-loop JS, the null
    result on loops is real; if both sit at the noise floor, the router itself
    is still undifferentiated and the loop question isn't answerable yet.
    """
    names = list(reps)
    n_layers = len(reps[names[0]]["layers"])
    layers = []
    for l in range(n_layers):
        # average over loops -> one usage histogram per source
        avg = {}
        for s in names:
            u = torch.tensor(reps[s]["layers"][l]["usage"])   # [R, E]
            avg[s] = u.mean(0)
        pairs = {
            f"{a}|{b}": _js_bits(avg[a], avg[b])
            for i, a in enumerate(names) for b in names[i + 1:]
        }
        loop_js = sum(reps[s]["layers"][l]["js_cross_bits"] for s in names) / len(names)
        null_js = sum(reps[s]["layers"][l]["js_null_bits"] for s in names) / len(names)
        layers.append({
            "layer": l,
            "js_cross_source_bits": pairs,
            "mean_js_cross_source": sum(pairs.values()) / max(len(pairs), 1),
            "mean_js_cross_loop": loop_js,
            "mean_js_null": null_js,
        })
    src = sum(x["mean_js_cross_source"] for x in layers) / max(n_layers, 1)
    loop = sum(x["mean_js_cross_loop"] for x in layers) / max(n_layers, 1)
    null = sum(x["mean_js_null"] for x in layers) / max(n_layers, 1)
    return {
        "sources": names,
        "layers": layers,
        "summary": {
            "mean_js_cross_source": src,
            "mean_js_cross_loop": loop,
            "mean_js_null": null,
            "source_vs_null_ratio": src / max(null, _EPS),
            "loop_vs_null_ratio": loop / max(null, _EPS),
            "verdict": _control_verdict(src, loop, null),
        },
    }


def _control_verdict(src: float, loop: float, null: float) -> str:
    real = lambda x: x > 2 * null and x > 0.01   # same floor as the per-source verdict
    if real(src) and not real(loop):
        return ("instrument works and loops do NOT specialize: domain separates experts, "
                "loop index does not -- recurrence is extra compute, not staged experts")
    if real(src) and real(loop):
        return "both separate: experts carry domain identity AND loop identity"
    if real(loop):
        return ("loops separate but domains do not -- unusual; check the sources really "
                "differ and that load balancing is not pinning usage")
    return ("router is undifferentiated overall -- neither domain nor loop separates. "
            "Inconclusive on loops: re-run later in training before concluding anything")


def format_cross_source(rep: dict) -> str:
    s = rep["summary"]
    out = [f"\ncross-source control ({', '.join(rep['sources'])})"]
    for lyr in rep["layers"]:
        pairs = "  ".join(f"{k}:{v:.5f}" for k, v in lyr["js_cross_source_bits"].items())
        out.append(
            f"  layer {lyr['layer']}: source JS {pairs}   "
            f"(loop {lyr['mean_js_cross_loop']:.5f}, null {lyr['mean_js_null']:.5f})"
        )
    out.append(
        f"  mean JS  source={s['mean_js_cross_source']:.5f}  loop={s['mean_js_cross_loop']:.5f}  "
        f"null={s['mean_js_null']:.5f}   (x null: source={s['source_vs_null_ratio']:.2f}, "
        f"loop={s['loop_vs_null_ratio']:.2f})"
    )
    out.append(f"  CONTROL: {s['verdict']}")
    return "\n".join(out)


def format_report(rep: dict) -> str:
    m, out = rep["meta"], []
    out.append(
        f"tokens={m['tokens']:,}  loops={m['n_loops']}  core_layers={m['core_layers']}  "
        f"experts={m['n_experts']} (top-{m['top_k']}, max entropy {m['max_entropy_bits']:.2f} bits)"
    )
    for lyr in rep["layers"]:
        out.append(f"\ncore layer {lyr['layer']}")
        out.append("  loop  entropy   usage per expert")
        for r, (row, e) in enumerate(zip(lyr["usage"], lyr["entropy_bits"])):
            bars = " ".join(f"{v:.3f}" for v in row)
            out.append(f"  {r:<5} {e:>6.3f}   {bars}")
        raw = "  ".join(f"{k}:{v:.3f}" for k, v in lyr["top1_agreement"].items())
        ka = "  ".join(
            f"{k}:" + ("  n/a" if v is None else f"{v:+.3f}") for k, v in lyr["top1_kappa"].items()
        )
        ja = "  ".join(f"{k}:{v:.3f}" for k, v in lyr["topk_jaccard"].items())
        out.append(f"  top-1 agreement (raw):                {raw}")
        out.append(f"  top-1 kappa (0=chance, 1=identical):  {ka}")
        out.append(f"  top-k Jaccard:                        {ja}")
        out.append(
            f"  usage JS: cross={lyr['js_cross_bits']:.5f}  null={lyr['js_null_bits']:.5f}  "
            f"ratio={lyr['js_ratio']:.2f}   NMI(expert;loop)={lyr['nmi_expert_loop']:.4f}"
        )
        out.append(
            f"  balance: dead={lyr['dead_experts']}  usage max={lyr['usage_max']:.3f} "
            f"min={lyr['usage_min']:.4f}  (uniform={1.0 / rep['meta']['n_experts']:.3f})"
        )
    s = rep["summary"]
    kap = "n/a" if s["mean_top1_kappa"] is None else f"{s['mean_top1_kappa']:.3f}"
    out.append(
        f"\nmean top-1 kappa = {kap}   mean JS = {s['mean_js_cross_bits']:.5f} bits "
        f"({s['mean_js_ratio']:.2f}x null)"
    )
    out.append(f"VERDICT (provisional -- confirm against the cross-source control): {s['verdict']}")
    return "\n".join(out)
