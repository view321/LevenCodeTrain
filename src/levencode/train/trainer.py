"""Single-GPU trainer for stages sft / edit / jepa.

Design: plain Python iterators (no DataLoader workers — Windows-safe), fp32
master weights with bf16 autocast on CUDA, gradient accumulation, cosine LR
with warmup, JSONL metrics + state.json for the WebUI, generation samples for
the gallery, and an automatic benchmark run at stage end.

Each metrics.jsonl row is averaged over the full window since the previous row
(log_every steps × grad_accum micro-batches), not a single-step snapshot — the
per-step 1/t-weighted diffusion loss is far too noisy to read raw."""

from __future__ import annotations

import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Iterator

import torch

from ..config import cfg_get
from ..data.collators import IGNORE, DiffusionSFTCollator, EditCollator
from ..data.corruption import CorruptionCfg
from ..data.mix import build_edit_stream, build_mixture
from ..data.tokens import TokenizerBundle
from ..model.backbone import load_tokenizer_bundle
from ..model.editor import LevencodeEditor, build_editor
from ..model.jepa import ema_momentum
from ..sampling.block_sampler import BlockSamplerCfg, generate
from ..util import resolve_device, set_seed
from .losses import (
    delete_loss,
    diffusion_fill_loss_sparse,
    insert_loss,
    jepa_loss,
    masked_ce_loss_sparse,
)
from .state import RunDir

GALLERY_PROMPTS = [
    [{"role": "user", "content": "Write a Python function that checks whether a string is a palindrome."}],
    [{"role": "user", "content": "What is 17 * 24? Think step by step and give the final answer after ####."}],
    [{"role": "user", "content": "Explain in two sentences what a hash map is."}],
]


class MetricAcc:
    """Sum/count accumulator for scalar training metrics.

    Each key is averaged over the number of times it was actually added, so
    per-branch metrics stay correctly scaled at any edit/SFT mix — `ce` is only
    added on retained-SFT micro-batches, `del_recall` only when a batch
    contains junk, etc. A plain sum divided by the global micro-batch count
    under-reports every conditional key (the bug this replaces)."""

    def __init__(self) -> None:
        self._sum: dict[str, float] = {}
        self._count: dict[str, int] = {}

    def add(self, key: str, value: float) -> None:
        self._sum[key] = self._sum.get(key, 0.0) + float(value)
        self._count[key] = self._count.get(key, 0) + 1

    def means(self) -> dict[str, float]:
        return {k: self._sum[k] / self._count[k] for k in self._sum}


class Trainer:
    def __init__(
        self,
        cfg: dict,
        sft_iter: Iterator[dict] | None = None,
        edit_iter: Iterator[dict] | None = None,
        bundle: TokenizerBundle | None = None,
        run_dir: str | Path | None = None,
    ):
        self.cfg = cfg
        self.stage = cfg["stage"]
        if self.stage not in ("sft", "edit", "jepa"):
            raise ValueError(f"Trainer handles sft/edit/jepa, not {self.stage!r} (grpo has its own runner)")
        self.device = resolve_device(cfg_get(cfg, "run.device", "auto"))
        set_seed(int(cfg_get(cfg, "run.seed", 1337)))
        self.rng = random.Random(cfg_get(cfg, "run.seed", 1337))

        repo = cfg_get(cfg, "model.repo_id")
        self.bundle = bundle or load_tokenizer_bundle(repo)
        init_from = cfg.get("init_from") or repo
        with_jepa = bool(cfg_get(cfg, "jepa.enabled", False))
        self.editor: LevencodeEditor = build_editor(
            init_from,
            insert_max=int(cfg_get(cfg, "model.insert_max", 8)),
            device=self.device,
            dtype=torch.float32,
            with_jepa=with_jepa,
            jepa_kwargs=dict(
                predictor_layers=int(cfg_get(cfg, "jepa.predictor_layers", 2)),
                predictor_heads=int(cfg_get(cfg, "jepa.predictor_heads", 8)),
            ),
        )
        self.with_jepa = with_jepa

        max_len = int(cfg_get(cfg, "model.max_seq_len", 1024))
        seed = int(cfg_get(cfg, "run.seed", 1337))
        self.sft_collator = DiffusionSFTCollator(self.bundle, cfg.get("diffusion", {}), max_len, seed)
        self.edit_collator = EditCollator(
            self.bundle,
            CorruptionCfg.from_dict(cfg.get("corruption", {})),
            insert_max=int(cfg_get(cfg, "model.insert_max", 8)),
            max_seq_len=max_len,
            seed=seed,
        )
        self._sft_iter = sft_iter
        self._edit_iter = edit_iter

        runs_root = Path(cfg_get(cfg, "run.runs_dir", "runs")) / cfg_get(cfg, "run.experiment", "levencode")
        self.run = RunDir(run_dir or runs_root / self.stage)

        t = cfg["train"]
        self.micro_bs = int(t["micro_batch_size"])
        # Edit micro-batches run three view forwards each; default them smaller.
        self.edit_micro_bs = int(t.get("edit_micro_batch_size") or max(self.micro_bs // 2, 1))
        self.grad_accum = int(t["grad_accum"])
        self.total_steps = int(t["total_steps"])
        self.grad_clip = float(t.get("grad_clip", 1.0))
        self.log_every = int(t.get("log_every", 10))
        self.ckpt_every = int(t.get("ckpt_every", 1000))
        self.sample_every = int(t.get("sample_every", 500))
        self.loss_w = t.get("loss_weights", {})
        self.retain_frac = float(t.get("retain_sft_frac", 0.5))

        self.opt = self._build_optimizer(t)
        self.warmup = int(t.get("warmup_steps", 100))
        self.base_lr = float(t["lr"])

        self.autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else nullcontext()
        )

    # ---------- setup ----------

    def _build_optimizer(self, t: dict) -> torch.optim.AdamW:
        decay, no_decay = [], []
        for name, p in self.editor.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim < 2 or "norm" in name.lower() or "bias" in name.lower():
                no_decay.append(p)
            else:
                decay.append(p)
        groups = [
            {"params": decay, "weight_decay": float(t.get("weight_decay", 0.1))},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        b1, b2 = t.get("betas", [0.9, 0.95])
        return torch.optim.AdamW(groups, lr=float(t["lr"]), betas=(b1, b2))

    def _lr_at(self, step: int) -> float:
        if step < self.warmup:
            return self.base_lr * (step + 1) / max(self.warmup, 1)
        frac = (step - self.warmup) / max(self.total_steps - self.warmup, 1)
        return self.base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(frac, 1.0))))

    def sft_stream(self) -> Iterator[dict]:
        if self._sft_iter is None:
            self._sft_iter = build_mixture(self.cfg["data"], int(cfg_get(self.cfg, "run.seed", 1337)))
        return self._sft_iter

    def edit_stream(self) -> Iterator[dict]:
        if self._edit_iter is None:
            self._edit_iter = build_edit_stream(self.cfg["data"], int(cfg_get(self.cfg, "run.seed", 1337)))
        return self._edit_iter

    def _next_batch(self, collator, stream: Iterator[dict], micro_bs: int) -> dict:
        for _ in range(50):
            samples = [next(stream) for _ in range(micro_bs)]
            batch = collator(samples)
            if batch is not None:
                return batch
        raise RuntimeError("data stream produced 50 consecutive empty batches")

    @staticmethod
    def _to(batch: dict, device: torch.device) -> dict:
        return {
            k: (v.to(device) if torch.is_tensor(v) else Trainer._to(v, device) if isinstance(v, dict) else v)
            for k, v in batch.items()
        }

    # ---------- losses ----------
    # Each _*_step computes, scales, and BACKWARDS its own losses view by view,
    # so at most one view's activation graph is alive at a time. Combined with
    # applying lm_head only at supervised positions (never materializing the
    # full [B, L, 65k] logits in the training path), this is what keeps the
    # edit/jepa stages inside 32GB.

    def _jepa_term(self, hidden: torch.Tensor, input_ids: torch.Tensor, labels: torch.Tensor, att: torch.Tensor) -> torch.Tensor:
        positions = labels != IGNORE
        clean_ids = torch.where(positions, labels, input_ids)
        target = self.editor.jepa.targets(clean_ids, att)
        pred = self.editor.jepa.predict(hidden, att)
        return jepa_loss(pred, target, positions)

    def _gathered_logits(self, h: torch.Tensor, labels: torch.Tensor):
        b_idx, pos = (labels != IGNORE).nonzero(as_tuple=True)
        logits_sel = self.backbone_lm_head(h[b_idx, pos])
        return logits_sel, labels[b_idx, pos], b_idx

    @property
    def backbone_lm_head(self):
        return self.editor.backbone.lm_head

    def _sft_step(self, batch: dict, metrics: MetricAcc) -> float:
        with self.autocast:
            h = self.editor.hidden(batch["input_ids"], batch["attention_mask"])
            logits_sel, labels_sel, b_idx = self._gathered_logits(h, batch["labels"])
            loss_fill, ce = diffusion_fill_loss_sparse(
                logits_sel, labels_sel, b_idx, batch["t"], batch["block_len"]
            )
            total = float(self.loss_w.get("fill", 1.0)) * loss_fill
            if self.with_jepa:
                jl = self._jepa_term(h, batch["input_ids"], batch["labels"], batch["attention_mask"])
                total = total + float(self.loss_w.get("jepa", 0.0)) * jl
                metrics.add("jepa_loss", jl.item())
        (total / self.grad_accum).backward()
        metrics.add("fill_loss", loss_fill.item())
        metrics.add("ce", ce.item())
        return total.item()

    def _edit_step(self, batch: dict, metrics: MetricAcc) -> float:
        d, i, f = batch["del"], batch["ins"], batch["fill"]
        total_display = 0.0

        with self.autocast:
            h_d = self.editor.hidden(d["input_ids"], d["attention_mask"])
            d_logits = self.editor.heads.delete_logits(h_d)
            dl, dacc = delete_loss(
                d_logits, d["labels"], pos_weight=float(self.loss_w.get("delete_pos_weight", 2.0))
            )
            scaled = float(self.loss_w.get("delete", 0.5)) * dl
        (scaled / self.grad_accum).backward()
        total_display += scaled.item()
        with torch.no_grad():  # junk recall: the "does it ever delete?" signal
            junk = d["labels"] == 1
            if junk.any():
                hit = (torch.sigmoid(d_logits.float()) > 0.5) & junk
                metrics.add("del_recall", (hit.sum() / junk.sum()).item())

        with self.autocast:
            h_i = self.editor.hidden(i["input_ids"], i["attention_mask"])
            i_logits = self.editor.heads.insert_logits(h_i)
            il, iacc = insert_loss(
                i_logits, i["labels"], zero_weight=float(self.loss_w.get("insert_zero_weight", 0.5))
            )
            scaled = float(self.loss_w.get("insert", 0.5)) * il
        (scaled / self.grad_accum).backward()
        total_display += scaled.item()
        with torch.no_grad():  # nonzero-gap recall: the "does it ever insert?" signal
            lab = i["labels"][:, : i_logits.shape[1]]
            nz = (lab != IGNORE) & (lab > 0)
            if nz.any():
                hit = (i_logits.argmax(-1) > 0) & nz
                metrics.add("ins_nonzero_recall", (hit.sum() / nz.sum()).item())

        with self.autocast:
            h_f = self.editor.hidden(f["input_ids"], f["attention_mask"])
            logits_sel, labels_sel, _ = self._gathered_logits(h_f, f["labels"])
            fl, facc = masked_ce_loss_sparse(logits_sel, labels_sel)
            scaled = float(self.loss_w.get("fill", 1.0)) * fl
            if self.with_jepa:
                jl = self._jepa_term(h_f, f["input_ids"], f["labels"], f["attention_mask"])
                scaled = scaled + float(self.loss_w.get("jepa", 0.0)) * jl
                metrics.add("jepa_loss", jl.item())
        (scaled / self.grad_accum).backward()
        total_display += scaled.item()

        metrics.add("del_loss", dl.item())
        metrics.add("del_acc", dacc.item())
        metrics.add("ins_loss", il.item())
        metrics.add("ins_acc", iacc.item())
        metrics.add("fill_view_loss", fl.item())
        metrics.add("fill_view_acc", facc.item())
        return total_display

    # ---------- main loop ----------

    def train(self) -> None:
        self.run.start(self.stage, self.total_steps, config=self.cfg)
        self.editor.train()
        tokens_seen = 0
        window_tokens = 0
        window_t0 = time.perf_counter()
        window = MetricAcc()  # accumulates across all steps since the last log

        try:
            for step in range(1, self.total_steps + 1):
                lr = self._lr_at(step - 1)
                for g in self.opt.param_groups:
                    g["lr"] = lr
                self.opt.zero_grad(set_to_none=True)

                step_loss = 0.0
                n_edit = 0
                for _ in range(self.grad_accum):
                    use_edit = self.stage in ("edit", "jepa") and self.rng.random() > self.retain_frac
                    if use_edit:
                        batch = self._to(
                            self._next_batch(self.edit_collator, self.edit_stream(), self.edit_micro_bs),
                            self.device,
                        )
                        n_edit += 1
                        loss_val = self._edit_step(batch, window)
                        n_tok = sum(v["input_ids"].numel() for v in batch.values())
                    else:
                        batch = self._to(
                            self._next_batch(self.sft_collator, self.sft_stream(), self.micro_bs),
                            self.device,
                        )
                        loss_val = self._sft_step(batch, window)
                        n_tok = batch["input_ids"].numel()
                    step_loss += loss_val / self.grad_accum
                    tokens_seen += n_tok
                    window_tokens += n_tok

                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.editor.parameters() if p.requires_grad], self.grad_clip
                )
                self.opt.step()
                window.add("loss", step_loss)
                window.add("edit_frac", n_edit / self.grad_accum)

                if self.with_jepa:
                    m = ema_momentum(
                        step, self.total_steps,
                        float(cfg_get(self.cfg, "jepa.ema_start", 0.996)),
                        float(cfg_get(self.cfg, "jepa.ema_end", 0.9995)),
                    )
                    self.editor.jepa.ema_update(self.editor.backbone.lfm2, m)

                if step % self.log_every == 0 or step == self.total_steps:
                    dt = time.perf_counter() - window_t0
                    tok_s = window_tokens / max(dt, 1e-9)
                    window_tokens, window_t0 = 0, time.perf_counter()
                    logged = window.means()
                    window = MetricAcc()
                    logged["tokens_seen"] = tokens_seen
                    self.run.progress(step, logged, lr=lr, tok_per_sec=tok_s)

                if self.sample_every and step % self.sample_every == 0:
                    self._dump_samples(step)

                if self.ckpt_every and step % self.ckpt_every == 0 and step < self.total_steps:
                    self.editor.save(self.run.root / "ckpt" / f"step_{step}")

            final_dir = self.run.root / "ckpt" / "final"
            self.editor.save(final_dir)
            self.bundle.tok.save_pretrained(str(final_dir / "backbone"))

            if bool(cfg_get(self.cfg, "bench.at_stage_end", True)):
                self.run.set_state(status="benchmarking")
                from ..bench.benchmark import run_benchmark

                results = run_benchmark(self.editor, self.bundle, self.cfg, self.device)
                self.run.save_bench(self.stage, results)

            self.run.finish("completed")
        except Exception:
            self.run.finish("failed")
            raise

    def _dump_samples(self, step: int) -> None:
        try:
            self.editor.eval()
            scfg = BlockSamplerCfg.from_dict(self.cfg.get("sampler", {}))
            scfg.stop_texts = ("[/Answer]",)
            samples = []
            with self.autocast:
                for messages in GALLERY_PROMPTS:
                    prompt_ids = self.bundle.chat_prompt_ids(messages)
                    res = generate(self.editor.mlm_call(), self.bundle, prompt_ids, scfg, self.device)
                    samples.append(
                        {
                            "prompt": messages[-1]["content"],
                            "output": res.text.strip(),
                            "tok_per_sec": round(res.tokens_per_sec, 1),
                        }
                    )
            self.run.save_samples(self.stage, step, samples)
        except Exception as e:  # sampling must never kill training
            self.run.save_samples(self.stage, step, [{"prompt": "<sampling failed>", "output": repr(e)}])
        finally:
            self.editor.train()
