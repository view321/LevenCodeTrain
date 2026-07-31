"""End-to-end smoke test against the REAL 350M model (CPU-friendly).

Verifies: tokenizer specials, backbone load, block-canvas generation on the
pretrained model, masked code infill, edit-head surgery, one full training
step (SFT objective), and the edit collator on the real tokenizer.

  python scripts/smoke_test.py            # ~2-4 min on CPU
  python scripts/smoke_test.py --fast     # smaller canvases, quicker
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    device = torch.device(args.device)

    from levencode.bench.fixtures import load_snippets
    from levencode.data.collators import DiffusionSFTCollator, EditCollator
    from levencode.data.corruption import CorruptionCfg
    from levencode.model.backbone import REPO_ID, load_backbone, load_tokenizer_bundle
    from levencode.model.editor import LevencodeEditor
    from levencode.sampling.block_sampler import BlockSamplerCfg, generate
    from levencode.train.losses import diffusion_fill_loss

    print(f"\n=== 1. tokenizer ===")
    b = load_tokenizer_bundle(REPO_ID)
    print(f"  vocab={b.vocab_size} mask={b.mask_id} pad={b.pad_id} bos={b.bos_id} "
          f"eos={b.eos_id} stops={b.stop_ids}")
    check("mask token present", b.mask_id is not None and b.mask_id >= 0)
    check("chat template renders", len(b.chat_prompt_ids([{"role": "user", "content": "hi"}])) > 3)

    print(f"\n=== 2. backbone ===")
    t0 = time.perf_counter()
    model = load_backbone(REPO_ID, dtype=torch.float32, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    check("backbone loads", True, f"{n_params/1e6:.1f}M params in {time.perf_counter()-t0:.1f}s")

    print(f"\n=== 3. block-canvas generation (pretrained) ===")
    scfg = BlockSamplerCfg(
        block_size=16 if args.fast else 32,
        steps_per_block=4 if args.fast else 8,
        max_blocks=1 if args.fast else 2,
        temperature=0.0,
        stop_texts=("[/Answer]",),
    )
    editor_tmp = LevencodeEditor(model, insert_max=8).to(device)
    prompt = b.chat_prompt_ids([{"role": "user", "content": "Write a Python function that adds two numbers."}])
    t0 = time.perf_counter()
    res = generate(editor_tmp.mlm_call(), b, prompt, scfg, device)
    dt = time.perf_counter() - t0
    print(f"  output ({res.tokens_per_sec:.1f} tok/s, {dt:.1f}s):")
    print("  " + "-" * 60)
    for line in (res.text.strip() or "(empty)").split("\n")[:12]:
        print("   | " + line)
    print("  " + "-" * 60)
    check("generation produces tokens", len(res.new_ids) > 0, f"{len(res.new_ids)} tokens, {res.blocks} block(s)")

    print(f"\n=== 4. masked code infill (pretrained) ===")
    code = load_snippets()[1]  # factorial
    lines = code.rstrip("\n").split("\n")
    li = 2
    pre, line, suf = "\n".join(lines[:li]) + "\n", lines[li], "\n" + "\n".join(lines[li + 1:]) + "\n"
    pre_ids, line_ids, suf_ids = b.encode(pre), b.encode(line), b.encode(suf)
    head = [b.bos_id] if b.bos_id is not None else []
    ids = head + pre_ids + [b.mask_id] * len(line_ids) + suf_ids + [b.eos_id]
    from levencode.bench.tasks import BenchCtx, fill_span
    ctx = BenchCtx(editor=editor_tmp, bundle=b, cfg={"model": {"max_seq_len": 512}}, device=device)
    filled = fill_span(ctx, ids, steps=4 if args.fast else 8)
    start = len(head) + len(pre_ids)
    pred_line = b.decode(filled[start:start + len(line_ids)])
    print(f"  masked line : {line!r}")
    print(f"  model filled: {pred_line!r}")
    check("infill runs", b.mask_id not in filled)

    print(f"\n=== 5. edit-head surgery ===")
    x = torch.tensor([ids[:64]], dtype=torch.long, device=device)
    out = editor_tmp(x)
    check(
        "editor forward shapes",
        out["mlm_logits"].shape[:2] == x.shape
        and out["del_logits"].shape == x.shape
        and out["ins_logits"].shape[1] == x.shape[1] - 1,
    )

    print(f"\n=== 6. one SFT training step (real weights) ===")
    coll = DiffusionSFTCollator(
        b, {"block_size_min": 8, "block_size_max": 24, "full_answer_prob": 0.3, "t_min": 0.05, "eos_pad": 8},
        max_seq_len=192, seed=0,
    )
    batch = coll([
        {"messages": [{"role": "user", "content": "Add 2 and 3."}, {"role": "assistant", "content": "2 + 3 = 5, so the answer is 5."}]},
        {"messages": [{"role": "user", "content": "Write a hello world in Python."}, {"role": "assistant", "content": "print('hello world')"}]},
    ])
    batch = {k: v.to(device) for k, v in batch.items()}
    opt = torch.optim.AdamW([p for p in editor_tmp.parameters() if p.requires_grad], lr=1e-5)
    t0 = time.perf_counter()
    out = editor_tmp(batch["input_ids"], batch["attention_mask"])
    loss, ce = diffusion_fill_loss(out["mlm_logits"], batch["labels"], batch["t"], batch["block_len"])
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in editor_tmp.parameters() if p.requires_grad], 1.0)
    opt.step()
    check("train step", torch.isfinite(loss).item(), f"loss={loss.item():.3f} ce={ce.item():.3f} in {time.perf_counter()-t0:.1f}s")

    print(f"\n=== 7. edit collator on real tokenizer ===")
    ecoll = EditCollator(b, CorruptionCfg(), insert_max=8, max_seq_len=512, seed=0)
    ebatch = ecoll([{"text": code} for code in load_snippets()[:4]])
    ok = (
        ebatch is not None
        and ebatch["del"]["input_ids"].shape == ebatch["del"]["labels"].shape
        and (ebatch["fill"]["input_ids"] == b.mask_id).any().item()
    )
    check("edit views build", bool(ok))

    print("\n=== summary ===")
    failed = [r for r in RESULTS if not r[1]]
    for name, ok, _ in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if failed:
        print(f"\n{len(failed)} FAILED")
        sys.exit(1)
    print("\nall smoke checks passed")


if __name__ == "__main__":
    main()
