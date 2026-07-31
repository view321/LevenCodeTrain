"""Run-state and metrics persistence consumed by the WebUI.

Layout under runs/<experiment>/<stage>/:
  state.json        current status/progress (atomic writes)
  metrics.jsonl     one row per log step
  bench/<name>.json benchmark results
  samples/<stage>.json generation gallery entries
"""

from __future__ import annotations

import time
from pathlib import Path

from ..util.io import append_jsonl, read_json, write_json_atomic


class RunDir:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._state: dict = read_json(self.root / "state.json", default={}) or {}

    # ---- state ----
    def set_state(self, **fields) -> None:
        self._state.update(fields)
        self._state["updated_at"] = time.time()
        write_json_atomic(self.root / "state.json", self._state)

    def start(self, stage: str, total_steps: int, config: dict | None = None) -> None:
        self.set_state(
            stage=stage,
            status="running",
            step=0,
            total_steps=total_steps,
            started_at=time.time(),
        )
        if config is not None:
            write_json_atomic(self.root / "config.json", config)

    def progress(self, step: int, losses: dict, lr: float, tok_per_sec: float) -> None:
        total = max(int(self._state.get("total_steps", 1)), 1)
        started = float(self._state.get("started_at", time.time()))
        elapsed = time.time() - started
        eta = (elapsed / max(step, 1)) * (total - step) if step > 0 else None
        self.set_state(step=step, pct=100.0 * step / total, eta_s=eta, last=losses, lr=lr, tok_per_sec=tok_per_sec)
        append_jsonl(
            self.root / "metrics.jsonl",
            {"step": step, "lr": lr, "tok_per_sec": tok_per_sec, **losses},
        )

    def finish(self, status: str = "completed") -> None:
        self.set_state(status=status)

    # ---- benchmark ----
    def save_bench(self, name: str, results: dict) -> None:
        write_json_atomic(self.root / "bench" / f"{name}.json", results)

    # ---- samples gallery ----
    def save_samples(self, stage: str, step: int, samples: list[dict], keep_last: int = 8) -> None:
        path = self.root / "samples" / f"{stage}.json"
        existing = read_json(path, default=[]) or []
        existing.append({"step": step, "samples": samples})
        write_json_atomic(path, existing[-keep_last:])
