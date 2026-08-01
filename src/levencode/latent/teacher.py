"""Frozen-teacher latent extraction (GRM-3.2-Turf) and the on-disk store.

Design (the distillation idea from the prompt): the teacher LLM is FROZEN — the
student never sees it during training. A precompute pass runs the teacher over
the data mix once, pools hidden states per chunk at every granularity level
(L2-normalized), and writes a compact memmap store. Training then consumes
(cached) teacher latents only, which is what makes a ~2h demo run feasible on a
single 5090 while still inheriting the semantic / context-aware / code-competent
properties of the teacher's representation.

Dual-dropout port (CALM §2.2, adapted to a frozen encoder): with probability
`dropout_tokens` each token inside a chunk is replaced by the mask token before
the teacher pass, so the pooled latent is CBOW-style *hazy* — the decodability
adapter must then learn to reconstruct the chunk from a latent that carries
semantic context rather than exact token indices."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..data.tokens import TokenizerBundle
from .chunker import HierarchicalSpec, hierarchical_spans

TEACHER_REPO = "OrionLLM/GRM-3.2-Turf"


def load_teacher(repo_id: str = TEACHER_REPO, device="cpu", dtype=None, cache_dir: str | None = None):
    """Lfm2ForCausalLM — transformers-native (config has no auto_map)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(repo_id, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        repo_id, torch_dtype=dtype or torch.bfloat16, cache_dir=cache_dir
    )
    return model.to(device), tok


class TeacherExtractor:
    """Pooled, L2-normalized teacher hidden states per chunk and per level."""

    def __init__(
        self,
        model,
        tokenizer,
        latent_dim: int | None = None,
        dropout_tokens: float = 0.15,
        pool: str = "mean",
    ):
        self.model = model
        self.tok = tokenizer
        self.dropout_tokens = dropout_tokens
        self.pool = pool
        self.rng = random.Random(0)
        self.latent_dim = latent_dim or model.config.hidden_size

    def _check_vocab(self, bundle: TokenizerBundle) -> None:
        """Both repos must share the LFM2 vocabulary for position alignment."""
        probe = "def check_palindrome(text):\n    return text == text[::-1]\n"
        a = bundle.encode(probe)
        b = self.tok.encode(probe, add_special_tokens=False)
        if a != b:
            raise ValueError(
                "teacher and student vocabularies disagree on a probe string "
                f"({len(a)} vs {len(b)} ids) — latents would not be position-aligned"
            )

    @torch.no_grad()
    def hiddens(self, ids: list[int], device) -> torch.Tensor:
        x = torch.tensor([ids], dtype=torch.long, device=device)
        out = self.model(x, use_cache=False, output_hidden_states=True)
        h = out.hidden_states[-1][0].float()  # [L, H]
        return h

    def pooled_for_spans(
        self, ids: list[int], h: torch.Tensor, spans: list[tuple[int, int]]
    ) -> torch.Tensor:
        """Mean (or last-token) pool over each span, then L2-normalize. [C, H]."""
        if self.pool == "last":
            vecs = torch.stack([h[e - 1] for _, e in spans])
        else:
            vecs = torch.stack([h[s:e].mean(0) for s, e in spans])
        return F_normalize(vecs)

    def masked_ids(self, ids: list[int], mask_id: int) -> list[int]:
        if self.dropout_tokens <= 0.0:
            return ids
        return [mask_id if self.rng.random() < self.dropout_tokens else t for t in ids]


def F_normalize(v: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(v, dim=-1)


@dataclass
class LatentExample:
    ctx_ids: list[int]                 # visible context (student encoder input)
    fine_spans: list[tuple[int, int]]  # token spans of fine chunks (relative to ctx_ids)
    coarse_of_fine: list[int]          # coarse chunk index of each fine chunk
    fine_tokens: list[list[int]]       # token ids per fine chunk
    z_fine: torch.Tensor | None        # [n_fine, H] teacher latents (may be None for store load)
    z_coarse: torch.Tensor | None      # [n_coarse, H]


class PrecomputedLatents:
    """Memmap store of per-sample chunked teacher latents.

    Layout under <dir>/:
      manifest.json    metadata (levels, teacher, dims, counts)
      samples.jsonl    per-sample ctx tokens, fine chunk tokens, nesting map
      z_fine.raw       float16 memmap [n_fine, latent_dim]
      z_coarse.raw     float16 memmap [n_coarse, latent_dim]
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def exists(self) -> bool:
        return (self.root / "manifest.json").exists()

    def manifest(self) -> dict:
        return json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))

    def write(self, examples: list[LatentExample], manifest: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        n_fine = sum(len(e.fine_tokens) for e in examples)
        n_coarse = sum(len(e.z_coarse) for e in examples)
        d = int(manifest["latent_dim"])
        zf = np.lib.format.open_memmap(self.root / "z_fine.raw", mode="w+", dtype="float16", shape=(n_fine, d))
        zc = np.lib.format.open_memmap(self.root / "z_coarse.raw", mode="w+", dtype="float16", shape=(n_coarse, d))
        rows = []
        f_off = c_off = 0
        for ex in examples:
            nf = len(ex.fine_tokens)
            nc = len(ex.z_coarse)
            if nf:
                zf[f_off : f_off + nf] = ex.z_fine.detach().float().cpu().numpy().astype("float16")
            if nc:
                zc[c_off : c_off + nc] = ex.z_coarse.detach().float().cpu().numpy().astype("float16")
            rows.append(
                {
                    "ctx_ids": ex.ctx_ids,
                    "fine_tokens": ex.fine_tokens,
                    "coarse_of_fine": ex.coarse_of_fine,
                    "f_off": f_off,
                    "c_off": c_off,
                    "n_fine": nf,
                    "n_coarse": nc,
                }
            )
            f_off += nf
            c_off += nc
        (self.root / "samples.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
        )
        manifest.update({"n_samples": len(rows), "n_fine": n_fine, "n_coarse": n_coarse})
        (self.root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _rows(self) -> list[dict]:
        lines = (self.root / "samples.jsonl").read_text(encoding="utf-8").splitlines()
        return [json.loads(l) for l in lines if l.strip()]

    def latents(self, kind: str, device) -> torch.Tensor:
        path = self.root / ("z_fine.raw" if kind == "fine" else "z_coarse.raw")
        m = np.load(path, mmap_mode="r")
        return torch.from_numpy(m.astype("float32")).to(device)

    def sample(self, idx: int) -> LatentExample:
        """Return one stored example with its latents (float32, device-agnostic)."""
        r = self._rows()[idx]
        zf = torch.from_numpy(np.load(self.root / "z_fine.raw", mmap_mode="r")[r["f_off"] : r["f_off"] + r["n_fine"]].astype("float32"))
        zc = torch.from_numpy(np.load(self.root / "z_coarse.raw", mmap_mode="r")[r["c_off"] : r["c_off"] + r["n_coarse"]].astype("float32"))
        return LatentExample(
            ctx_ids=r["ctx_ids"],
            fine_spans=[],
            coarse_of_fine=r["coarse_of_fine"],
            fine_tokens=r["fine_tokens"],
            z_fine=zf,
            z_coarse=zc,
        )

    def batch(self, idxs: list[int], ctx_len: int, coarse_window: int, fine_window: int, seed: int = 0, pad_id: int = 0):
        """Random contiguous chunk window per sample. The fine window is taken
        from the FIRST coarse chunk of the window only, so every fine chunk in
        a row shares one conditioning coarse latent (`z_coarse_cond`).

        The context is the sample prefix PLUS the text of every chunk before
        the window (reconstructed from the stored fine_tokens) — at inference
        the AR prior conditions on all committed text, so training must too."""
        import random

        rng = random.Random(seed)
        rows = self._rows()
        zf_m = np.load(self.root / "z_fine.raw", mmap_mode="r")
        zc_m = np.load(self.root / "z_coarse.raw", mmap_mode="r")
        ctx_rows, coarse_zs, fine_zs, cond_zs = [], [], [], []
        coarse_mask, fine_mask = [], []
        coarse_lens = []
        for i in idxs:
            r = rows[i]
            n_c = r["n_coarse"]
            n_f = r["n_fine"]
            if n_c == 0 or n_f == 0:
                continue
            c0 = rng.randint(0, max(0, n_c - coarse_window))
            c1 = min(n_c, c0 + coarse_window)
            n_cw = c1 - c0
            f_idx = [j for j in range(n_f) if r["coarse_of_fine"][j] == c0]
            if not f_idx:
                continue
            # start at the coarse chunk's first fine chunk so the fine AR
            # sequence matches the inference-time conditioning
            f_sel = f_idx[:fine_window]
            zc = zc_m[r["c_off"] + c0 : r["c_off"] + c1].astype("float32")
            zf = zf_m[r["f_off"] + f_sel[0] : r["f_off"] + f_sel[0] + len(f_sel)].astype("float32")
            body_before = [t for j in range(f_sel[0]) for t in r["fine_tokens"][j]]
            ctx = (r["ctx_ids"] + body_before)[-ctx_len:]
            if len(ctx) < ctx_len:
                ctx = [pad_id] * (ctx_len - len(ctx)) + ctx
            coarse_zs.append(zc)
            fine_zs.append(zf)
            cond_zs.append(zc[0])
            ctx_rows.append(ctx)
            coarse_lens.append(n_cw)
            coarse_mask.append([1.0] * n_cw + [0.0] * (coarse_window - n_cw))
            fine_mask.append([1.0] * len(f_sel) + [0.0] * (fine_window - len(f_sel)))
        if not coarse_zs:
            return None
        w_c = coarse_window
        w_f = fine_window
        return {
            "ctx_ids": torch.tensor(ctx_rows, dtype=torch.long),
            "z_coarse": torch.tensor(self._pad_rows(coarse_zs, w_c, zc_m.shape[1])),
            "z_fine": torch.tensor(self._pad_rows(fine_zs, w_f, zf_m.shape[1])),
            "z_coarse_cond": torch.tensor(np.stack(cond_zs)),
            "coarse_mask": torch.tensor(coarse_mask),
            "fine_mask": torch.tensor(fine_mask),
            "coarse_lens": torch.tensor(coarse_lens),
        }

    @staticmethod
    def _pad_rows(rows: list[np.ndarray], width: int, dim: int) -> np.ndarray:
        """Pad variable-length chunk windows to a fixed width (zeros are
        masked out by the per-window masks)."""
        out = np.zeros((len(rows), width, dim), dtype="float32")
        for i, r in enumerate(rows):
            out[i, : len(r)] = r
        return out

    def decodability_batch(self, idxs: list[int], chunk_tokens: int, n_chunks: int, seed: int = 0):
        """Batches of (teacher latent, chunk tokens) pairs for the decodability
        adapter: fixed fine-chunk K token windows sampled from the store.
        Short chunks are right-padded with IGNORE (-100), which the token-CE
        decode skips — supervising pad positions would teach the adapter to
        emit a pad token after every chunk."""
        import random

        rng = random.Random(seed)
        rows = self._rows()
        zf_m = np.load(self.root / "z_fine.raw", mmap_mode="r")
        zs, tss = [], []
        for i in idxs:
            r = rows[i]
            if not r["fine_tokens"]:
                continue
            j = rng.randrange(r["n_fine"])
            tok = r["fine_tokens"][j]
            t = tok[:chunk_tokens] + [-100] * (chunk_tokens - len(tok))
            z = zf_m[r["f_off"] + j].astype("float32")
            zs.append(z)
            tss.append(t)
        if not zs:
            return None
        return {
            "z": torch.tensor(np.stack(zs)),
            "tokens": torch.tensor(tss, dtype=torch.long),
        }

    def sample_rows(self, kind: str, n: int, seed: int = 0) -> torch.Tensor:
        """Random rows from a latent memmap without materializing the whole
        store (used for the RVQ EMA warm-up at trainer init)."""
        m = np.load(self.root / ("z_fine.raw" if kind == "fine" else "z_coarse.raw"), mmap_mode="r")
        rng = np.random.default_rng(seed)
        idx = rng.choice(m.shape[0], size=min(n, m.shape[0]), replace=False)
        idx.sort()
        return torch.from_numpy(np.asarray(m[idx]).astype("float32"))
