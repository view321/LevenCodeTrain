from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable


def write_json_atomic(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def read_jsonl(path: str | Path, max_points: int | None = None) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # tolerate a torn final line while training is writing
    if max_points and len(rows) > max_points:
        stride = len(rows) / max_points
        rows = [rows[int(i * stride)] for i in range(max_points - 1)] + [rows[-1]]
    return rows


def downsample(rows: list, max_points: int) -> list:
    if len(rows) <= max_points:
        return rows
    stride = len(rows) / max_points
    return [rows[int(i * stride)] for i in range(max_points - 1)] + [rows[-1]]
