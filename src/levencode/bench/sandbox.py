"""Subprocess execution of model-generated Python for pass@1 checks.

Isolation: separate process, `-I` (isolated mode: no site-packages, ignores
env vars), fresh temp cwd, hard timeout, output capped. This guards against
accidents, not adversaries — standard practice for local benchmark harnesses
(HumanEval/MBPP-style). Run the box offline-ish or accept that risk profile."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


def run_python(code: str, timeout_s: float = 5.0) -> tuple[bool, str]:
    """Returns (passed, detail). passed = process exited 0 within timeout."""
    with tempfile.TemporaryDirectory(prefix="levencode_sbx_") as td:
        script = Path(td) / "prog.py"
        script.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(script)],
                cwd=td,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
        except OSError as e:
            return False, f"oserror: {e}"
    if proc.returncode == 0:
        return True, "ok"
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()
    return False, tail[-1][:200] if tail else f"exit {proc.returncode}"
