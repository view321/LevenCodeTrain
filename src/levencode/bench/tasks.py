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
from ..latent.sampler import LatentSamplerCfg, generate_latent
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

    def latent_sampler_cfg(self) -> LatentSamplerCfg:
        lcfg = LatentSamplerCfg.from_dict(self.cfg.get("latent_sampler", {}))
        lcfg.stop_texts = ("[/Answer]",)
        return lcfg

    def generate(self, prompt_ids: list[int]):
        """Block-diffusion generation, or the latent-guided sampler when the
        checkpoint has the latent stack and bench.latent_mode is on."""
        if self.editor.latent is not None and bool(self.bench("latent_mode", True)):
            return generate_latent(
                self.editor, self.editor.latent, self.bundle, prompt_ids, self.latent_sampler_cfg(), self.device
            )
        return generate(self.editor.mlm_call(), self.bundle, prompt_ids, self.sampler_cfg(), self.device)

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
        res = ctx.generate(prompt_ids)
        gold = extract_number(ex["answer"])
        pred = extract_number(res.text)
        correct += int(numbers_equal(pred, gold))
        total += 1
    return {"gsm8k_em": correct / max(total, 1), "n": total}


_ASSERT_NAME_RE = re.compile(r"assert\s+(\w+)\s*\(")


def contract_name(test: str) -> str | None:
    """The function name the test contract calls — must survive repair."""
    m = _ASSERT_NAME_RE.search(test or "")
    return m.group(1) if m else None


def repair_code_text(
    ctx: BenchCtx, code: str, ecfg: EditSamplerCfg, protect_names: tuple[str, ...] = ()
) -> str:
    """Run the trained Levenshtein editor over a code string (draft -> repair).
    Tokens of `protect_names` (all BPE variants, with/without leading space)
    are shielded from the delete head."""
    b = ctx.bundle
    extra: set[int] = set()
    for name in protect_names:
        for variant in (name, " " + name):
            extra.update(b.encode(variant))
    head = [b.bos_id] if b.bos_id is not None else [b.eos_id]
    ids = head + b.encode(code) + [b.eos_id]
    out, _trace = repair(
        ctx.editor.editor_call(), b, ids, ecfg, ctx.device,
        extra_protected=frozenset(extra) or None,
    )
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
        res = ctx.generate(prompt_ids)
        code = salvage_code(res.text)
        gen_valid += int(syntax_ok(code))
        test_block = "\n\n" + "\n".join(tests) + "\n"
        ok, detail = run_python(code + test_block, timeout)
        passed += int(ok)

        rep_ok = ok
        fixed = None
        if self_repair and not ok:
            name = contract_name(tests[0])
            fixed = repair_code_text(ctx, code, ecfg, protect_names=(name,) if name else ())
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


# ---------- BrierLM (CALM Sec. 4): sample-based, likelihood-free ----------

def _brierlm_from_scores(
    plan_logits_list: list[torch.Tensor],  # per fine chunk: [K, V] plan logits
    gold_chunks: list[list[int]],          # per fine chunk: [K] gold tokens
    backbone_logits: torch.Tensor,         # [n_chunks*K, V] mean-field at masked positions
    n_grams: tuple[int, ...] = (1, 2, 3, 4),
    temperature: float = 0.7,
    seed: int = 0,
) -> dict:
    """Brier-n over n-gram outcomes, estimated from 2 samples per position
    (CALM eq. 14): Brier(P, y) ~= I{x1=y} + I{x2=y} - I{x1=x2}."""
    rng = random.Random(seed)
    flat = torch.cat([p for p in plan_logits_list], dim=0) + backbone_logits  # [M, V]
    gen = torch.Generator(device=flat.device).manual_seed(seed)
    probs = torch.softmax(flat.float() / temperature, dim=-1)
    n_pos = probs.shape[0]
    s1 = torch.multinomial(probs, 1, generator=gen).squeeze(-1)
    s2 = torch.multinomial(probs, 1, generator=gen).squeeze(-1)
    gold = torch.tensor([t for c in gold_chunks for t in c], dtype=torch.long)
    stats = {}
    for n in n_grams:
        total, hits = 0, 0.0
        for p in range(n_pos - n + 1):
            y = tuple(gold[p : p + n].tolist())
            x1 = tuple(s1[p : p + n].tolist())
            x2 = tuple(s2[p : p + n].tolist())
            hits += (1 if x1 == y else 0) + (1 if x2 == y else 0) - (1 if x1 == x2 else 0)
            total += 1
        stats[f"brier_{n}"] = hits / max(total, 1)
    # Brier-n can go negative (samples collide more than they hit gold — a
    # confidently-wrong predictor); the geometric mean of an even number of
    # negatives would sign-flip to a respectable-looking positive score, so
    # clamp each component at 0: a degenerate predictor scores 0, not 47.
    brierlm = 1.0
    for n in n_grams:
        brierlm *= max(stats[f"brier_{n}"], 0.0)
    stats["brierlm"] = 100.0 * (brierlm ** (1.0 / len(n_grams)))
    return stats


def _latent_teacher_force(
    ctx: BenchCtx, prefix_ids: list[int], gold_chunks: list[list[int]], cfg: LatentSamplerCfg
) -> list[torch.Tensor]:
    """Teacher-forced chunk-latent prediction (the model's plan logits per
    gold fine chunk), mirroring the generation-time conditioning."""
    latent = ctx.editor.latent
    b = ctx.bundle
    fpc = cfg.fine_per_coarse
    k = cfg.fine_chunk_tokens
    ctx_ids = list(prefix_ids)
    prev_coarse = torch.zeros(1, 0, 2, dtype=torch.long, device=ctx.device)
    prev_fine = torch.zeros(1, 0, 2, dtype=torch.long, device=ctx.device)
    plan_logits: list[torch.Tensor] = []

    def ctx_embed(ids):
        from ..latent.sampler import _ctx_embed

        return _ctx_embed(latent, ctx.editor, ids, ctx.device, cfg.ctx_len)

    from ..latent.sampler import _plan_logits

    # mirror generate_latent's conditioning: coarse ctx pinned at the code
    # window's start, fine ctx pinned at the coarse chunk's start (training
    # never advances the pooled ctx inside a window / fine sequence)
    window_emb = fine_emb = None
    for ci, chunk in enumerate(gold_chunks):
        if ci % fpc == 0:
            if prev_coarse.shape[1] >= cfg.code_history:
                prev_coarse = prev_coarse[:, :0]
            fine_emb = ctx_embed(ctx_ids)
            if prev_coarse.shape[1] == 0:
                window_emb = fine_emb
            z_c, codes_c = latent.predict_coarse_latent(window_emb, prev_coarse, cfg.__dict__)
            prev_coarse = torch.cat([prev_coarse, codes_c.unsqueeze(0)], dim=1)
            prev_fine = torch.zeros(1, 0, 2, dtype=torch.long, device=ctx.device)
        z_f, codes_f = latent.predict_fine_latent(fine_emb, codes_c, prev_fine, cfg.__dict__)
        plan_logits.append(_plan_logits(latent, ctx.editor, z_f, k, ctx.device))
        prev_fine = torch.cat([prev_fine, codes_f.unsqueeze(0)], dim=1)
        ctx_ids = ctx_ids + chunk
    return plan_logits


@torch.no_grad()
def task_brierlm(ctx: BenchCtx) -> dict:
    """BrierLM (CALM Sec. 4): likelihood-free, sample-based LM metric. For each
    fixture context we teacher-force the gold continuation through the model
    (latent stack: plan codes + CFG residual -> adapter plan logits; token
    model: masked-softmax baseline) and draw 2 samples per position.

    Only needs samples, so it drops into the bench cleanly — and it correlates
    with CE (-0.966 Pearson) where mode-averaged generation metrics hide."""
    n = int(ctx.bench("brierlm_n", 16))
    b = ctx.bundle
    lcfg = ctx.latent_sampler_cfg()
    k = lcfg.fine_chunk_tokens
    fpc = lcfg.fine_per_coarse
    total_chunks = 4
    head = [b.bos_id] if b.bos_id is not None else [b.eos_id]
    seeds = [0, 1]
    agg: dict[str, list] = {}
    used = 0
    for code in load_snippets():
        if used >= n:
            break
        ids = b.encode(code)
        if len(ids) < 64:
            continue
        cut = len(ids) // 2
        prefix, cont = ids[:cut], ids[cut : cut + k * total_chunks]
        if len(cont) < k * total_chunks:
            continue
        gold_chunks = [cont[i * k : (i + 1) * k] for i in range(total_chunks)]
        prefix_ids = head + prefix
        masked_positions = list(range(len(prefix_ids), len(prefix_ids) + k * total_chunks))
        x = list(prefix_ids) + [b.mask_id] * (k * total_chunks)
        xt = torch.tensor([x], dtype=torch.long, device=ctx.device)
        logits = ctx.editor.mlm_call()(xt)[0].float()  # mean-field backbone at masks
        bb = logits[masked_positions, :]

        if ctx.editor.latent is not None:
            plans = _latent_teacher_force(ctx, prefix_ids, gold_chunks, lcfg)
            for s in seeds:
                r = _brierlm_from_scores(plans, gold_chunks, bb, seed=s)
                for kk, v in r.items():
                    agg.setdefault("latent_" + kk, []).append(v)
        else:
            plans = [torch.zeros_like(bb[:k]) for _ in range(total_chunks)]
            for s in seeds:
                r = _brierlm_from_scores(plans, gold_chunks, bb, seed=s)
                for kk, v in r.items():
                    agg.setdefault("sft_" + kk, []).append(v)
        used += 1
    if not agg:
        return {"error": "no usable fixtures"}
    out = {kk: sum(v) / len(v) for kk, v in agg.items()}
    out["n"] = used
    return out
