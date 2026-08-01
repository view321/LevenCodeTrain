"""YAML config loading with single-parent inheritance via `_extends`."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

_MISSING = object()


def deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge; override wins, lists are replaced wholesale."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path) -> dict:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent_rel = raw.pop("_extends", None)
    if parent_rel:
        parent = load_config(path.parent / parent_rel)
        raw = deep_merge(parent, raw)
    return raw


def apply_overrides(cfg: dict, pairs: list[str]) -> dict:
    """In-place dotted-path overrides: ["bench.mbpp_n=257", "run.device=cuda"].
    Values are coerced int -> float -> bool -> str."""
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        if not sep:
            raise ValueError(f"override must look like key=value, got {pair!r}")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        val: Any = raw
        try:
            val = int(raw)
        except ValueError:
            try:
                val = float(raw)
            except ValueError:
                if raw.lower() in ("true", "false"):
                    val = raw.lower() == "true"
        node[parts[-1]] = val
    return cfg


def cfg_get(cfg: dict, dotted: str, default: Any = _MISSING) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            if default is _MISSING:
                raise KeyError(f"config key not found: {dotted}")
            return default
        node = node[part]
    return node
