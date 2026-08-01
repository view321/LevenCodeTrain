"""Loom bench: causal-LM adaptations of the levencode benchmark suite.

Same result-file shape as levencode's bench ({"meta": ..., task: {...}}), so
the WebUI bench table renders these runs unchanged. Reuses levencode's answer
parsing (extract_number), code salvage, and execution sandbox.

Tasks: heldout (WikiText-103 CE/ppl), chat_ce, arc_easy (mean-logprob MC),
gsm8k (EM), mbpp (sandboxed pass@1), speed, brierlm (sample-based, with the
positive-clamped composite — negative components mean confidently-wrong)."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from levencode.bench.sandbox import run_python
from levencode.bench.tasks import extract_number, numbers_equal, salvage_code
from levencode.config import cfg_get

from .model import LoomLM


@dataclass
class LoomBenchCtx:
    model: LoomLM
    bundle: object
    cfg: dict
    device: torch.device
    chat_format: bool
    use_concepts: bool

    def bench(self, key: str, default):
        return self.cfg.get("bench", {}).get(key, default)


# ---------- helpers ----------

def _prompt_ids(ctx: LoomBenchCtx, user_text: str) -> list[int]:
    if ctx.chat_format:
        return ctx.bundle.chat_prompt_ids([{"role": "user", "content": user_text}])
    head = [ctx.bundle.bos_id] if ctx.bundle.bos_id is not None else []
    return head + ctx.bundle.encode(user_text + "\nAnswer:")


@torch.no_grad()
def _gen_text(ctx: LoomBenchCtx, prompt_ids: list[int], max_new: int, temperature: float = 0.0) -> tuple[str, int, float]:
    x = torch.tensor([prompt_ids], dtype=torch.long, device=ctx.device)
    t0 = time.perf_counter()
    out = ctx.model.generate(
        x, max_new_tokens=max_new, temperature=temperature, top_p=0.9,
        stop_ids=tuple(ctx.bundle.stop_ids), use_concepts=ctx.use_concepts,
    )
    dt = time.perf_counter() - t0
    new = [t for t in out[0, len(prompt_ids):].tolist() if t not in ctx.bundle.stop_ids]
    return ctx.bundle.decode(new), len(new), dt


@torch.no_grad()
def mean_logprob(ctx: LoomBenchCtx, prefix_ids: list[int], cont_ids: list[int], max_tokens: int = 48) -> float:
    cont_ids = cont_ids[:max_tokens]
    ids = torch.tensor([prefix_ids + cont_ids], dtype=torch.long, device=ctx.device)
    logp = ctx.model(ids)["logits"].float().log_softmax(-1)
    start = len(prefix_ids)
    total = sum(logp[0, start + i - 1, t].item() for i, t in enumerate(cont_ids))
    return total / max(len(cont_ids), 1)


def brier_ngram_stats(
    logits: torch.Tensor,  # [L, V], logits[i] predicts gold[i]
    gold: list[int],
    temperature: float = 0.7,
    seed: int = 0,
    n_grams: tuple[int, ...] = (1, 2, 3, 4),
) -> dict:
    """Sample-based Brier-n (CALM Sec. 4): two draws per position; n-grams are
    tuples of consecutive per-position samples. Components can be negative
    (confidently wrong); the composite clamps at 0 so four negatives cannot
    sign-flip into a respectable score."""
    g = torch.Generator().manual_seed(seed)
    probs = (logits.float().cpu() / max(temperature, 1e-6)).softmax(-1)
    s1 = torch.multinomial(probs, 1, generator=g)[:, 0].tolist()
    s2 = torch.multinomial(probs, 1, generator=g)[:, 0].tolist()
    stats: dict = {}
    for n in n_grams:
        hits, tot = 0, 0
        for p in range(len(gold) - n + 1):
            y, a, b = tuple(gold[p : p + n]), tuple(s1[p : p + n]), tuple(s2[p : p + n])
            hits += int(a == y) + int(b == y) - int(a == b)
            tot += 1
        stats[f"brier_{n}"] = hits / max(tot, 1)
    comp = 1.0
    for n in n_grams:
        comp *= max(stats[f"brier_{n}"], 0.0)
    stats["brierlm"] = 100.0 * comp ** (1.0 / len(n_grams))
    return stats


def _heldout_rows(ctx: LoomBenchCtx, n_rows: int, seq_len: int) -> list[list[int]]:
    """WikiText-103 test packed into seq_len rows — canonical, truly held out."""
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    buf: list[int] = []
    rows: list[list[int]] = []
    for ex in ds:
        txt = ex.get("text", "")
        if not txt.strip():
            continue
        buf.extend(ctx.bundle.encode(txt))
        buf.append(ctx.bundle.eos_id)
        while len(buf) >= seq_len and len(rows) < n_rows:
            rows.append(buf[:seq_len])
            buf = buf[seq_len:]
        if len(rows) >= n_rows:
            break
    return rows


# ---------- tasks ----------

@torch.no_grad()
def task_heldout(ctx: LoomBenchCtx) -> dict:
    import math

    n = int(ctx.bench("heldout_n", 32))
    seq_len = int(cfg_get(ctx.cfg, "model.max_seq_len", 1024))
    rows = _heldout_rows(ctx, n, seq_len)
    ces = []
    for row in rows:
        x = torch.tensor([row], dtype=torch.long, device=ctx.device)
        ces.append(ctx.model(x, labels=x)["ce"].item())
    ce = sum(ces) / max(len(ces), 1)
    return {"heldout_ce": ce, "heldout_ppl": math.exp(min(ce, 20.0)), "n": len(ces)}


@torch.no_grad()
def task_chat_ce(ctx: LoomBenchCtx) -> dict:
    from datasets import load_dataset

    n = int(ctx.bench("chat_loss_n", 64))
    try:
        ds = load_dataset("HuggingFaceTB/smoltalk", "all", split="test", streaming=True)
    except Exception:
        ds = load_dataset("HuggingFaceTB/smoltalk", "all", split="train", streaming=True).skip(50_000)
    seq_len = int(cfg_get(ctx.cfg, "model.max_seq_len", 1024))
    ces, idx = [], 0
    for ex in ds:
        msgs = ex.get("messages")
        if not msgs or msgs[-1].get("role") != "assistant":
            continue
        try:
            prefix, answer = ctx.bundle.chat_pair_ids(msgs)
        except Exception:
            continue
        if not answer or len(prefix) + len(answer) > seq_len:
            continue
        ids = torch.tensor([prefix + answer], dtype=torch.long, device=ctx.device)
        labels = torch.tensor([[-100] * len(prefix) + answer], dtype=torch.long, device=ctx.device)
        ces.append(ctx.model(ids, labels=labels)["ce"].item())
        idx += 1
        if idx >= n:
            break
    if not ces:
        return {"error": "no usable chat samples"}
    return {"chat_ce": sum(ces) / len(ces), "n": idx}


@torch.no_grad()
def task_arc_easy(ctx: LoomBenchCtx) -> dict:
    from datasets import load_dataset

    n = int(ctx.bench("arc_n", 200))
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation")
    ds = ds.select(range(min(n, len(ds))))
    correct = total = 0
    for ex in ds:
        prompt_ids = _prompt_ids(ctx, ex["question"])
        scores = []
        for text in ex["choices"]["text"]:
            cont = ctx.bundle.encode(" " + text.strip())
            scores.append(mean_logprob(ctx, prompt_ids, cont) if cont else float("-inf"))
        pred = ex["choices"]["label"][scores.index(max(scores))]
        correct += int(pred == ex["answerKey"])
        total += 1
    return {"arc_easy_acc": correct / max(total, 1), "n": total}


@torch.no_grad()
def task_gsm8k(ctx: LoomBenchCtx) -> dict:
    from datasets import load_dataset

    n = int(ctx.bench("gsm8k_n", 100))
    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.select(range(min(n, len(ds))))
    correct = total = 0
    for ex in ds:
        q = ex["question"] + "\nThink step by step and end with the final numeric answer after ####."
        text, _, _ = _gen_text(ctx, _prompt_ids(ctx, q), max_new=256)
        correct += int(numbers_equal(extract_number(text), extract_number(ex["answer"])))
        total += 1
    return {"gsm8k_em": correct / max(total, 1), "n": total}


@torch.no_grad()
def task_mbpp(ctx: LoomBenchCtx) -> dict:
    from datasets import load_dataset

    n = int(ctx.bench("mbpp_n", 50))
    timeout = float(ctx.bench("exec_timeout_s", 5.0))
    ds = load_dataset("google-research-datasets/mbpp", "full", split="test")
    ds = ds.select(range(min(n, len(ds))))
    passed = syntax_n = total = 0
    failures = []
    for ex in ds:
        tests = "\n".join(ex["test_list"])
        q = f"{ex['text']}\nYour code should pass these tests:\n```python\n{tests}\n```"
        text, _, _ = _gen_text(ctx, _prompt_ids(ctx, q), max_new=256)
        code = salvage_code(text)
        ok_syntax = bool(code.strip()) and _syntax_ok(code)
        syntax_n += int(ok_syntax)
        ok, detail = run_python(code + "\n" + tests + "\n", timeout_s=timeout) if ok_syntax else (False, "no code")
        passed += int(ok)
        total += 1
        if not ok and len(failures) < 3:
            failures.append({"prompt": ex["text"], "generated": text[:400], "detail": detail[:200]})
    return {
        "mbpp_pass1": passed / max(total, 1),
        "mbpp_gen_syntax_rate": syntax_n / max(total, 1),
        "n": total,
        "failures": failures,
    }


def _syntax_ok(code: str) -> bool:
    import ast

    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


@torch.no_grad()
def task_speed(ctx: LoomBenchCtx) -> dict:
    rates = []
    for p in ("def fibonacci(n):\n", "The three primary colors are"):
        head = [ctx.bundle.bos_id] if ctx.bundle.bos_id is not None else []
        _, n_new, dt = _gen_text(ctx, head + ctx.bundle.encode(p), max_new=128, temperature=0.7)
        if n_new:
            rates.append(n_new / dt)
    return {"gen_tok_per_sec": sum(rates) / max(len(rates), 1)}


@torch.no_grad()
def task_brierlm(ctx: LoomBenchCtx) -> dict:
    n = int(ctx.bench("brierlm_n", 8))
    prefix_len, gold_len = 64, 32
    rows = _heldout_rows(ctx, n, prefix_len + gold_len)
    acc: dict[str, list[float]] = {}
    for i, row in enumerate(rows):
        prefix, gold = row[:prefix_len], row[prefix_len:]
        x = torch.tensor([row], dtype=torch.long, device=ctx.device)
        logits = ctx.model(x)["logits"][0]
        aligned = logits[prefix_len - 1 : prefix_len + gold_len - 1]  # [i] predicts gold[i]
        for s in (0, 1):
            r = brier_ngram_stats(aligned, gold, seed=i * 2 + s)
            for k, v in r.items():
                if k.startswith("brier_"):
                    acc.setdefault(k, []).append(v)
    out = {k: sum(v) / len(v) for k, v in acc.items()}
    comp = 1.0
    for k in sorted(out):
        comp *= max(out[k], 0.0)
    out["brierlm"] = 100.0 * comp ** (1.0 / max(len(acc), 1))
    out["n"] = len(rows)
    return out


TASKS = {
    "heldout": task_heldout,
    "chat": task_chat_ce,
    "arc_easy": task_arc_easy,
    "gsm8k": task_gsm8k,
    "mbpp": task_mbpp,
    "speed": task_speed,
    "brierlm": task_brierlm,
}


def run_loom_bench(
    model: LoomLM, bundle, cfg: dict, device, only: list[str] | None = None
) -> dict:
    from contextlib import nullcontext

    stage = cfg.get("stage", "")
    ctx = LoomBenchCtx(
        model=model, bundle=bundle, cfg=cfg, device=device,
        chat_format=bool(cfg_get(cfg, "bench.chat_format", stage == "sft")),
        use_concepts=bool(cfg_get(cfg, "bench.use_concepts", stage != "pretrain")),
    )
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    )
    results: dict = {"meta": {"time": time.time(), "stage": stage,
                              "chat_format": ctx.chat_format, "use_concepts": ctx.use_concepts}}
    for name, fn in TASKS.items():
        if only and name not in only:
            continue
        t0 = time.perf_counter()
        try:
            with autocast:
                results[name] = fn(ctx)
        except Exception as e:  # a missing dataset must not sink the whole bench
            results[name] = {"error": repr(e)[:300]}
        results[name]["seconds"] = round(time.perf_counter() - t0, 1)
        print(name, results[name], flush=True)
    return results
