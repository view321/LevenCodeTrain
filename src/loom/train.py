"""Loom trainer: Muon (2D trunk) + AdamW (everything else), three stages.

  pretrain  packed fineweb-edu/code/math, plain CE (concepts are a zero-init
            structural no-op — nothing concept-related trains here)
  concept   EMA copy provides pooled-hidden targets; the predictor trains by
            smooth-L1 regression, and the trunk+modulator train by CE while
            conditioned on the predictor's own (detached, causally valid)
            plans — so FiLM learns from plans of exactly the quality it will
            see at inference
  sft       chat mix, loss on answer tokens

Writes runs/<experiment>/<stage>/ via levencode's RunDir, so the existing
WebUI (state, curves, gallery, bench table) works unchanged."""

from __future__ import annotations

import copy
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Iterator

import torch

from levencode.config import cfg_get
from levencode.model.backbone import load_tokenizer_bundle
from levencode.train.state import RunDir
from levencode.util import resolve_device, set_seed

from .concept import concept_loss, pool_segments
from .config import LoomConfig
from .data import collate_pretrain, collate_sft, pretrain_stream, sft_stream
from .model import LoomLM
from .muon import Muon

GALLERY_PRETRAIN = [
    "def quicksort(arr):\n",
    "The capital of France is",
    "Question: What is 17 * 24?\nAnswer:",
]
GALLERY_CHAT = [
    [{"role": "user", "content": "Write a Python function that checks whether a string is a palindrome."}],
    [{"role": "user", "content": "What is 17 * 24? Think step by step and give the final answer after ####."}],
    [{"role": "user", "content": "Explain in two sentences what a hash map is."}],
]

STAGES = ("pretrain", "concept", "sft")


class LoomTrainer:
    def __init__(self, cfg: dict, bundle=None, train_iter: Iterator[dict] | None = None):
        self.cfg = cfg
        self.stage = cfg["stage"]
        if self.stage not in STAGES:
            raise ValueError(f"loom trainer handles {STAGES}, not {self.stage!r}")
        self.device = resolve_device(cfg_get(cfg, "run.device", "auto"))
        set_seed(int(cfg_get(cfg, "run.seed", 1337)))

        self.bundle = bundle or load_tokenizer_bundle(cfg_get(cfg, "model.tokenizer_repo"))
        init_from = cfg.get("init_from")
        if init_from and (Path(init_from) / "model.pt").exists():
            self.model = LoomLM.load(init_from, device=self.device)
            print(f"[loom] initialized from {init_from}")
        else:
            if init_from:
                raise FileNotFoundError(f"init_from={init_from!r} has no model.pt")
            lcfg = dict(cfg.get("loom", {}))
            lcfg["vocab_size"] = self.bundle.vocab_size
            self.model = LoomLM(LoomConfig.from_dict(lcfg)).to(self.device)
        self.model.train()

        # concept stage: EMA copy of the whole model provides latent targets
        self.ema: LoomLM | None = None
        if self.stage == "concept":
            if bool(cfg_get(cfg, "concept.freeze_trunk", False)):
                for n, p in self.model.named_parameters():
                    if not ("concept_predictor" in n or "modulator" in n):
                        p.requires_grad_(False)
            self.ema = copy.deepcopy(self.model).eval()
            for p in self.ema.parameters():
                p.requires_grad_(False)

        groups = self.model.param_groups()
        muon_params = [p for p in groups["muon"] if p.requires_grad]
        adamw_params = [p for p in groups["adamw"] if p.requires_grad]
        t = cfg.get("train", {})
        self.lr_muon = float(t.get("lr_muon", 0.02))
        self.lr_adamw = float(t.get("lr_adamw", 3e-4))
        self.opt_muon = (
            Muon(muon_params, lr=self.lr_muon, momentum=float(t.get("muon_momentum", 0.95)),
                 weight_decay=float(t.get("weight_decay", 0.0)))
            if muon_params else None
        )
        self.opt_adamw = torch.optim.AdamW(
            adamw_params, lr=self.lr_adamw, betas=(0.9, 0.95),
            weight_decay=float(t.get("weight_decay", 0.0)),
        )

        self.micro_bs = int(t.get("micro_batch_size", 16))
        self.grad_accum = int(t.get("grad_accum", 4))
        self.total_steps = int(t.get("total_steps", 1000))
        self.warmup = int(t.get("warmup_steps", 1000))
        self.grad_clip = float(t.get("grad_clip", 1.0))
        self.log_every = int(t.get("log_every", 20))
        self.sample_every = int(t.get("sample_every", 500))
        self.ckpt_every = int(t.get("ckpt_every", 2000))
        self.grad_checkpoint = bool(t.get("grad_checkpoint", False))
        self.seq_len = int(cfg_get(cfg, "model.max_seq_len", 1024))
        self.ema_momentum = float(cfg_get(cfg, "concept.ema_momentum", 0.999))
        self.w_concept = float(cfg_get(cfg, "concept.loss_weight", 1.0))

        seed = int(cfg_get(cfg, "run.seed", 1337))
        if train_iter is not None:
            self._rows = train_iter
        elif self.stage == "sft":
            self._rows = sft_stream(cfg["data"], self.bundle, self.seq_len, seed)
        else:
            self._rows = pretrain_stream(cfg["data"], self.bundle, self.seq_len, seed)

        exp = cfg_get(cfg, "run.experiment", "loom")
        self.run = RunDir(Path(cfg_get(cfg, "run.dir", "runs")) / exp / self.stage)
        self.autocast = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda" else nullcontext()
        )

    # ---------- batches ----------

    def _next_batch(self) -> dict:
        rows = [next(self._rows) for _ in range(self.micro_bs)]
        if self.stage == "sft":
            return collate_sft(rows, self.bundle.pad_id)
        return collate_pretrain(rows)

    # ---------- losses ----------

    def _lm_step(self, batch: dict) -> tuple[torch.Tensor, dict]:
        out = self.model(
            batch["input_ids"], labels=batch["labels"], grad_checkpoint=self.grad_checkpoint
        )
        parts = {"ce": out["ce"].item(), "aux_lb": out["aux_lb"].item(), "aux_z": out["aux_z"].item()}
        return out["loss"], parts

    def _concept_step(self, batch: dict) -> tuple[torch.Tensor, dict]:
        ids = batch["input_ids"]
        seg = self.model.cfg.segment_len
        with torch.no_grad():
            tgt = pool_segments(self.ema(ids)["hidden"], seg)  # [B, S, D]
        S = tgt.shape[1]
        preds = self.model.concept_predictor(tgt)  # [B, S+1, D]; [:, j] predicts c_j
        c_loss = concept_loss(preds[:, :S], tgt)
        # trunk + FiLM see the predictor's own plans (detached: the predictor
        # trains against latent targets, not against CE)
        out = self.model(
            ids, concepts=preds[:, :S].detach(), labels=batch["labels"],
            grad_checkpoint=self.grad_checkpoint,
        )
        loss = out["loss"] + self.w_concept * c_loss
        parts = {
            "ce": out["ce"].item(), "concept": c_loss.item(),
            "aux_lb": out["aux_lb"].item(), "aux_z": out["aux_z"].item(),
        }
        # Concept guidance should pay off (if at all) at segment BOUNDARIES —
        # the first tokens of a segment, where "which way is this going" is the
        # binding uncertainty — not in the locally-determined interior. Track
        # the split so the LCM's contribution is visible live, not inferred.
        with torch.no_grad():
            import torch.nn.functional as F

            V = out["logits"].shape[-1]
            nll = F.cross_entropy(  # bf16 logits are fine for a diagnostic
                out["logits"][:, :-1].reshape(-1, V),
                batch["labels"][:, 1:].reshape(-1),
                ignore_index=-100, reduction="none",
            ).reshape(ids.shape[0], -1)
            offs = torch.arange(1, ids.shape[1], device=ids.device) % seg
            valid = batch["labels"][:, 1:] != -100
            k = max(seg // 4, 1)
            b_mask = valid & (offs < k)
            i_mask = valid & (offs >= k)
            if b_mask.any():
                parts["ce_boundary"] = nll[b_mask].mean().item()
            if i_mask.any():
                parts["ce_interior"] = nll[i_mask].mean().item()
        return loss, parts

    @torch.no_grad()
    def _ema_update(self) -> None:
        m = self.ema_momentum
        for pe, po in zip(self.ema.parameters(), self.model.parameters()):
            pe.mul_(m).add_(po, alpha=1 - m)
        for be, bo in zip(self.ema.buffers(), self.model.buffers()):
            be.copy_(bo)

    # ---------- schedule ----------

    def _lr_frac(self, step: int) -> float:
        if step < self.warmup:
            return step / max(self.warmup, 1)
        p = (step - self.warmup) / max(self.total_steps - self.warmup, 1)
        return 0.1 + 0.45 * (1 + math.cos(math.pi * min(p, 1.0)))  # cosine to 10%

    # ---------- gallery ----------

    @torch.no_grad()
    def _gallery(self, step: int) -> None:
        try:
            self.model.eval()
            use_c = self.stage != "pretrain"
            samples = []
            prompts = GALLERY_CHAT if self.stage == "sft" else GALLERY_PRETRAIN
            for p in prompts:
                if self.stage == "sft":
                    ids = self.bundle.chat_prompt_ids(p)
                    shown = p[-1]["content"]
                else:
                    ids = ([self.bundle.bos_id] if self.bundle.bos_id is not None else []) + self.bundle.encode(p)
                    shown = p
                x = torch.tensor([ids], dtype=torch.long, device=self.device)
                t0 = time.perf_counter()
                with self.autocast:
                    out = self.model.generate(
                        x, max_new_tokens=96, temperature=0.7, top_p=0.9,
                        stop_ids=tuple(self.bundle.stop_ids), use_concepts=use_c,
                    )
                dt = time.perf_counter() - t0
                new = [t for t in out[0, len(ids):].tolist() if t not in self.bundle.stop_ids]
                samples.append({
                    "prompt": shown,
                    "output": self.bundle.decode(new).strip(),
                    "tok_per_sec": round(len(new) / max(dt, 1e-9), 1),
                })
            self.run.save_samples(self.stage, step, samples)
        except Exception as e:  # sampling must never kill training
            print(f"[loom] gallery failed: {e!r}", flush=True)
        finally:
            self.model.train()

    # ---------- main loop ----------

    def train(self) -> None:
        self.run.start(self.stage, self.total_steps, config=self.cfg)
        window: dict[str, list[float]] = {}
        tok_window = 0
        t_window = time.perf_counter()
        for step in range(1, self.total_steps + 1):
            frac = self._lr_frac(step)
            if self.opt_muon:
                for g in self.opt_muon.param_groups:
                    g["lr"] = self.lr_muon * frac
            for g in self.opt_adamw.param_groups:
                g["lr"] = self.lr_adamw * frac
            if self.opt_muon:
                self.opt_muon.zero_grad(set_to_none=True)
            self.opt_adamw.zero_grad(set_to_none=True)

            step_loss = 0.0
            for _ in range(self.grad_accum):
                batch = {k: v.to(self.device) for k, v in self._next_batch().items()}
                with self.autocast:
                    if self.stage == "concept":
                        loss, parts = self._concept_step(batch)
                    else:
                        loss, parts = self._lm_step(batch)
                (loss / self.grad_accum).backward()
                step_loss += loss.item() / self.grad_accum
                tok_window += batch["input_ids"].numel()
                for k, v in parts.items():
                    window.setdefault(k, []).append(v)
            gn = torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], self.grad_clip
            )
            window.setdefault("grad_norm", []).append(float(gn))
            if self.opt_muon:
                self.opt_muon.step()
            self.opt_adamw.step()
            if self.ema is not None:
                self._ema_update()
            window.setdefault("loss", []).append(step_loss)

            if step % self.log_every == 0 or step == self.total_steps:
                dt = time.perf_counter() - t_window
                means = {k: sum(v) / len(v) for k, v in window.items() if v}
                self.run.progress(step, means, lr=self.lr_adamw * frac, tok_per_sec=tok_window / max(dt, 1e-9))
                window.clear()
                tok_window = 0
                t_window = time.perf_counter()
            if self.sample_every and step % self.sample_every == 0:
                self._gallery(step)
            if self.ckpt_every and step % self.ckpt_every == 0:
                self.model.save(self.run.root / "ckpt" / f"step_{step}")

        self.model.save(self.run.root / "ckpt" / "final")
        self.run.finish()

        if bool(cfg_get(self.cfg, "bench.at_stage_end", True)):
            from .bench import run_loom_bench

            self.model.eval()
            results = run_loom_bench(self.model, self.bundle, self.cfg, self.device)
            self.run.save_bench(f"{self.stage}_final", results)
