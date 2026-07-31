"""Benchmark orchestrator: runs every task, tolerates per-task failure."""

from __future__ import annotations

import time
from contextlib import nullcontext

import torch

from ..data.tokens import TokenizerBundle
from .tasks import (
    BenchCtx,
    task_arc_easy,
    task_chat,
    task_gsm8k,
    task_infill,
    task_mbpp,
    task_repair,
    task_speed,
)

TASKS = {
    "chat": task_chat,
    "arc_easy": task_arc_easy,
    "gsm8k": task_gsm8k,
    "mbpp": task_mbpp,
    "repair": task_repair,
    "infill": task_infill,
    "speed": task_speed,
}


def run_benchmark(
    editor: torch.nn.Module,
    bundle: TokenizerBundle,
    cfg: dict,
    device: torch.device,
    only: list[str] | None = None,
) -> dict:
    editor.eval()
    ctx = BenchCtx(editor=editor, bundle=bundle, cfg=cfg, device=device)
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    results: dict = {"meta": {"time": time.time(), "stage": cfg.get("stage")}}
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
    editor.train()
    return results
