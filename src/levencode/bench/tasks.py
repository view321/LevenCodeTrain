"""Benchmark tasks: general chat, reasoning (ARC-Easy), math (GSM8K), code
(MBPP pass@1), plus the differentiated evals (repair, infill) and speed.

Every task returns a flat metrics dict. Dataset-backed tasks load small fixed
subsets deterministically; failures degrade to {"error": ...} without killing
the run (the box may be offline for HF datasets)."""

from __future__ import annotations

import ast
import random
import re
from dataclasses import dataclass

import torch

from ..data.corruption import CorruptionCfg, corrupt, make_junk_sampler
from ..data.tokens import TokenizerBundle
from ..data.mix import extract_code
from ..sampling.block_sampler import BlockSamplerCfg, generate
from ..sampling.edit_sampler import EditSamplerCfg, repair
from ..util.lev import lev_reduction
from .fixtures import load_snippets
from .sandbox import run_python

NUM_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")
HASH_ANS_RE = re.compile(r"####\s*([-+]?[\d,]*\.?\d+)")


@dataclass
class BenchCtx:
    editor: torch.nn.Module
    bundle: TokenizerBundle
    cfg: dict
    device: torch.device

    def sampler_cfg(self) -> BlockSamplerCfg:
        scfg = BlockSamplerCfg.from_dict(self.cfg.get("sampler", {}))
        scfg.stop_texts = ("[/Answer]",)
        scfg.max_blocks = int(self.cfg.get("bench", {}).get("gen_max_blocks", scfg.max_blocks))
        return scfg

    def bench(self, key: str, default):
        return self.cfg.get("bench", {}).get(key, default)


# ---------- answer parsing ----------

def extract_number(text: str) -> str | None:
    m = HASH_ANS_RE.findall(text)
    cand = m[-1] if m else None
    if cand is None:
        all_nums = NUM_RE.findall(text)
        cand = all_nums[-1] if all_nums else None
    if cand is None:
        return None
    return cand.replace(",", "").rstrip(".")


def numbers_equal(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-4
    except ValueError:
        return a == b


def syntax_ok(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


_CODE_START_RE = re.compile(r"\s*(def |import |from |class |@)")


def salvage_code(text: str) -> str:
    """Best-effort code extraction from a model answer: fenced block if valid,
    else from the first code-looking line onward, trimming trailing prose lines
    until it parses. Small models often answer without fences — without this,
    MBPP scores the prose and reports a misleading 0."""
    code = extract_code(text)
    if syntax_ok(code):
        return code
    lines = (text or "").splitlines()
    starts = [i for i, l in enumerate(lines) if _CODE_START_RE.match(l)]
    if starts:
        cand = lines[starts[0]:]
        for cut in range(0, min(6, len(cand))):
            trimmed = "\n".join(cand[: len(cand) - cut] if cut else cand)
            if syntax_ok(trimmed):
                return trimmed
    return code


# ---------- scoring helpers ----------

@torch.no_grad()
def pll_choice_logprob(
    ctx: BenchCtx, prompt_ids: list[int], choice_ids: list[int], chunk: int = 8, max_tokens: int = 48
) -> float:
    """Pseudo-log-likelihood (BERT-style): for token i, mask ONLY position i
    (rest of the choice visible) and take its log-prob; average over tokens.
    Far better calibrated than the fully-masked mean-field estimate, which
    scored multiple-choice at chance level."""
    b = ctx.bundle
    choice_ids = choice_ids[:max_tokens]
    L = len(choice_ids)
    base = prompt_ids + choice_ids
    start = len(prompt_ids)
    rows = []
    for i in range(L):
        row = list(base)
        row[start + i] = b.mask_id
        rows.append(row)
    total = 0.0
    for c0 in range(0, L, chunk):
        batch = rows[c0 : c0 + chunk]
        x = torch.tensor(batch, dtype=torch.long, device=ctx.device)
        logits = ctx.editor.mlm_call()(x).float().log_softmax(-1)
        for j in range(len(batch)):
            i = c0 + j
            total += logits[j, start + i, choice_ids[i]].item()
    return total / max(L, 1)


@torch.no_grad()
def masked_answer_ce(ctx: BenchCtx, prefix: list[int], answer: list[int], t: float, seed: int) -> float:
    """CE on a deterministic subset of answer tokens masked at rate t."""
    rng = random.Random(seed)
    b = ctx.bundle
    ids = list(prefix) + list(answer)
    masked_pos = [len(prefix) + i for i in range(len(answer)) if rng.random() < t]
    if not masked_pos:
        masked_pos = [len(prefix) + rng.randrange(len(answer))]
    targets = [ids[p] for p in masked_pos]
    for p in masked_pos:
        ids[p] = b.mask_id
    x = torch.tensor([ids], dtype=torch.long, device=ctx.device)
    logits = ctx.editor.mlm_call()(x)[0].float().log_softmax(-1)
    ce = -sum(logits[p, t_].item() for p, t_ in zip(masked_pos, targets)) / len(masked_pos)
    return ce


@torch.no_grad()
def fill_span(ctx: BenchCtx, ids_with_masks: list[int], steps: int = 8) -> list[int]:
    from ..sampling.edit_sampler import _fill_masks

    x = torch.tensor([ids_with_masks], dtype=torch.long, device=ctx.device)
    call = ctx.editor.editor_call()
    x = _fill_masks(call, x, ctx.bundle.mask_id, steps, temperature=0.0, top_p=0.9)
    return x[0].tolist()


# ---------- tasks ----------

def task_chat(ctx: BenchCtx) -> dict:
    """Held-out chat masked-CE at three mask rates (lower = better)."""
    from datasets import load_dataset

    n = int(ctx.bench("chat_loss_n", 64))
    try:
        ds = load_dataset("HuggingFaceTB/smoltalk", "all", split="test", streaming=True)
    except Exception:
        ds = load_dataset("HuggingFaceTB/smoltalk", "all", split="train", streaming=True).skip(50_000)
    ces = []
    idx = 0
    max_len = int(ctx.cfg.get("model", {}).get("max_seq_len", 1024))
    for ex in ds:
        msgs = ex.get("messages")
        if not msgs or msgs[-1].get("role") != "assistant":
            continue
        try:
            prefix, answer = ctx.bundle.chat_pair_ids(msgs)
        except Exception:
            continue
        if not answer or len(prefix) + len(answer) > max_len:
            continue
        for ti, t in enumerate((0.15, 0.5, 0.85)):
            ces.append(masked_answer_ce(ctx, prefix, answer, t, seed=idx * 10 + ti))
        idx += 1
        if idx >= n:
            break
    if not ces:
        return {"error": "no usable chat samples"}
    return {"chat_masked_ce": sum(ces) / len(ces), "n": idx}


def task_arc_easy(ctx: BenchCtx) -> dict:
    from datasets import load_dataset

    n = int(ctx.bench("arc_n", 200))
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation")
    ds = ds.select(range(min(n, len(ds))))
    correct = 0
    total = 0
    for ex in ds:
        prompt_ids = ctx.bundle.chat_prompt_ids(
            [{"role": "user", "content": ex["question"]}]
        )
        scores = []
        for text in ex["choices"]["text"]:
            choice_ids = ctx.bundle.encode(text.strip())
            if not choice_ids:
                scores.append(float("-inf"))
                continue
            scores.append(pll_choice_logprob(ctx, prompt_ids, choice_ids))
        pred = ex["choices"]["label"][scores.index(max(scores))]
        correct += int(pred == ex["answerKey"])
        total += 1
    return {"arc_easy_acc": correct / max(total, 1), "n": total}


def task_gsm8k(ctx: BenchCtx) -> dict:
    from datasets import load_dataset

    n = int(ctx.bench("gsm8k_n", 100))
    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.select(range(min(n, len(ds))))
    scfg = ctx.sampler_cfg()
    correct = 0
    total = 0
    for ex in ds:
        prompt_ids = ctx.bundle.chat_prompt_ids(
            [{
                "role": "user",
                "content": ex["question"]
                + "\nThink step by step and end with the final numeric answer after ####.",
            }]
        )
        res = generate(ctx.editor.mlm_call(), ctx.bundle, prompt_ids, scfg, ctx.device)
        gold = extract_number(ex["answer"])
        pred = extract_number(res.text)
        correct += int(numbers_equal(pred, gold))
        total += 1
    return {"gsm8k_em": correct / max(total, 1), "n": total}


def repair_code_text(ctx: BenchCtx, code: str, ecfg: EditSamplerCfg) -> str:
    """Run the trained Levenshtein editor over a code string (draft -> repair)."""
    b = ctx.bundle
    head = [b.bos_id] if b.bos_id is not None else [b.eos_id]
    ids = head + b.encode(code) + [b.eos_id]
    out, _trace = repair(ctx.editor.editor_call(), b, ids, ecfg, ctx.device)
    return b.decode(out)


def task_mbpp(ctx: BenchCtx) -> dict:
    """MBPP pass@1, plus the draft+repair pipeline: failed generations get one
    pass through the edit sampler before re-execution. The delta between
    mbpp_pass1 and mbpp_pass1_selfrepair is the end-to-end value of the editor
    on the model's OWN mistakes (decohered identifiers, glued tokens) — the
    product thesis in one number."""
    from datasets import load_dataset

    n = int(ctx.bench("mbpp_n", 50))
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    ds = ds.select(range(min(n, len(ds))))
    scfg = ctx.sampler_cfg()
    ecfg = EditSamplerCfg.from_dict(ctx.cfg.get("edit_sampler", {}))
    self_repair = bool(ctx.bench("mbpp_self_repair", True))
    timeout = float(ctx.bench("exec_timeout_s", 5.0))
    passed = 0
    repaired_passed = 0
    repair_changed = 0
    gen_valid = 0
    total = 0
    failures: list[dict] = []
    for ex in ds:
        tests = ex["test_list"]
        prompt_ids = ctx.bundle.chat_prompt_ids(
            [{
                "role": "user",
                "content": ex["prompt"]
                + "\nYour code should pass this test:\n"
                + tests[0]
                + "\nWrite only the Python code.",
            }]
        )
        res = generate(ctx.editor.mlm_call(), ctx.bundle, prompt_ids, scfg, ctx.device)
        code = salvage_code(res.text)
        gen_valid += int(syntax_ok(code))
        test_block = "\n\n" + "\n".join(tests) + "\n"
        ok, detail = run_python(code + test_block, timeout)
        passed += int(ok)

        rep_ok = ok
        fixed = None
        if self_repair and not ok:
            fixed = repair_code_text(ctx, code, ecfg)
            if fixed.strip() and fixed.strip() != code.strip():
                repair_changed += 1
                rep_ok, _ = run_python(fixed + test_block, timeout)
        repaired_passed += int(rep_ok)

        total += 1
        if not rep_ok and len(failures) < 3:
            entry = {
                "prompt": ex["prompt"][:200],
                "generated": res.text[:400],
                "extracted": code[:400],
                "detail": detail,
            }
            if fixed is not None:
                entry["repaired"] = fixed[:400]
            failures.append(entry)
    out = {
        "mbpp_pass1": passed / max(total, 1),
        "mbpp_gen_syntax_rate": gen_valid / max(total, 1),
        "n": total,
        "failures": failures,
    }
    if self_repair:
        out["mbpp_pass1_selfrepair"] = repaired_passed / max(total, 1)
        out["mbpp_repair_changed"] = repair_changed / max(total, 1)
    return out


def task_repair(ctx: BenchCtx) -> dict:
    """Corrupt fixture code with known edits; the editor must recover it
    self-located (no oracle hints). The signature eval for this project.

    Also reports the failure-mode diagnostics that make the headline numbers
    interpretable: no-op rate (editor did nothing -> lev_reduction ~0), length
    ratio (runaway insertion -> ratio >> 1, hugely negative lev_reduction —
    the signature of UNTRAINED heads, i.e. any stage-1 checkpoint), and an
    oracle variant (true edit locations given, only the fill is the model's) to
    separate can't-locate from can't-fill."""
    n = int(ctx.bench("repair_n", 40))
    seed = int(ctx.cfg.get("run", {}).get("seed", 1337))
    b = ctx.bundle
    ccfg = CorruptionCfg.from_dict(ctx.cfg.get("corruption", {}))
    ecfg = EditSamplerCfg.from_dict(ctx.cfg.get("edit_sampler", {}))
    head = [b.bos_id] if b.bos_id is not None else [b.eos_id]
    # The oracle variant is independent of sampler knobs; sweeps disable it
    # after measuring it once (bench.repair_oracle: false).
    with_oracle = bool(ctx.bench("repair_oracle", True))
    exact = valid = noop = oracle_exact = oracle_valid = 0
    reductions: list[float] = []
    len_ratios: list[float] = []
    deleted = inserted = 0
    total = 0
    for i, code in enumerate(load_snippets()[:n]):
        rng = random.Random(seed + i)
        clean = head + b.encode(code) + [b.eos_id]
        junk = make_junk_sampler(b.vocab_size, frozenset(b.protected | {b.mask_id}), echo_pool=clean)
        c = corrupt(clean, rng, ccfg, junk, protected=b.protected)
        if c.n_junk() + c.n_missing() == 0:
            continue

        out, trace = repair(ctx.editor.editor_call(), b, c.corrupted, ecfg, ctx.device)
        exact += int(out == clean)
        valid += int(syntax_ok(b.decode(out)))
        noop += int(out == c.corrupted)
        reductions.append(lev_reduction(c.corrupted, out, clean))
        len_ratios.append(len(out) / max(len(clean), 1))
        deleted += trace.deleted
        inserted += trace.inserted

        if with_oracle:
            # oracle: kept tokens + the true number of masks at each gap; the
            # model only has to FILL. Upper-bounds perfect localization.
            kept = c.kept_sequence()
            gaps = c.gap_counts()
            oracle_in: list[int] = []
            for j, tok in enumerate(kept):
                oracle_in.append(tok)
                if j < len(gaps):
                    oracle_in.extend([b.mask_id] * gaps[j])
            filled = fill_span(ctx, oracle_in, steps=int(ecfg.fill_steps))
            oracle_exact += int(filled == clean)
            oracle_valid += int(syntax_ok(b.decode(filled)))
        total += 1

    if total == 0:
        return {"error": "no corrupted samples generated"}
    out = {
        "repair_exact": exact / total,
        "repair_syntax_valid": valid / total,
        "repair_lev_reduction": sum(reductions) / total,
        "repair_noop_rate": noop / total,
        "repair_len_ratio": sum(len_ratios) / total,
        "repair_mean_deleted": deleted / total,
        "repair_mean_inserted": inserted / total,
        "n": total,
    }
    if with_oracle:
        out["repair_oracle_exact"] = oracle_exact / total
        out["repair_oracle_syntax_valid"] = oracle_valid / total
    return out


def task_infill(ctx: BenchCtx) -> dict:
    """Mask one middle line of fixture code; exact-match the refill."""
    n = int(ctx.bench("infill_n", 40))
    b = ctx.bundle
    head = [b.bos_id] if b.bos_id is not None else [b.eos_id]
    exact = 0
    valid = 0
    total = 0
    for code in load_snippets()[:n]:
        lines = code.rstrip("\n").split("\n")
        candidates = [i for i in range(1, len(lines) - 1) if lines[i].strip()]
        if not candidates:
            continue
        li = candidates[len(candidates) // 2]
        pre = "\n".join(lines[:li]) + "\n"
        line = lines[li]
        suf = "\n" + "\n".join(lines[li + 1 :]) + "\n"
        pre_ids, line_ids, suf_ids = b.encode(pre), b.encode(line), b.encode(suf)
        if not line_ids:
            continue
        ids = head + pre_ids + [b.mask_id] * len(line_ids) + suf_ids + [b.eos_id]
        filled = fill_span(ctx, ids)
        start = len(head) + len(pre_ids)
        pred = filled[start : start + len(line_ids)]
        exact += int(pred == line_ids)
        valid += int(syntax_ok(b.decode(filled)))
        total += 1
    if total == 0:
        return {"error": "no infillable fixtures"}
    return {"infill_exact": exact / total, "infill_syntax_valid": valid / total, "n": total}


def task_speed(ctx: BenchCtx) -> dict:
    scfg = ctx.sampler_cfg()
    scfg.max_blocks = 4
    prompts = [
        [{"role": "user", "content": "Write a Python function that reverses a linked list."}],
        [{"role": "user", "content": "Summarize what unit tests are for."}],
    ]
    rates = []
    for messages in prompts:
        res = generate(
            ctx.editor.mlm_call(), ctx.bundle, ctx.bundle.chat_prompt_ids(messages), scfg, ctx.device
        )
        if res.new_ids:
            rates.append(res.tokens_per_sec)
    return {"gen_tok_per_sec": sum(rates) / max(len(rates), 1)}
