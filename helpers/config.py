"""Config loading: `configs/*.yaml` with optional per-session overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from .paths import CONFIG_DIR

_CACHE: dict[str, dict] = {}


def load(name: str, *, config_dir: Path | None = None, refresh: bool = False) -> dict:
    """Load `configs/<name>.yaml` (cached). `name` carries no extension."""
    directory = Path(config_dir) if config_dir else CONFIG_DIR
    key = f"{directory}/{name}"
    if refresh or key not in _CACHE:
        path = directory / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"config not found: {path}")
        _CACHE[key] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return copy.deepcopy(_CACHE[key])


def deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge; `override` wins. Lists are replaced, not merged."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    """`get(cfg, "inserts.timing.lead_in_s", 0.3)`."""
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def aspect_config(aspect: str | None = None) -> tuple[str, dict]:
    """Return `(aspect, its layout block)` from `configs/layout.yaml`."""
    layout = load("layout")
    key = aspect or layout.get("default_aspect", "9:16")
    aspects = layout.get("aspects", {})
    if key not in aspects:
        raise KeyError(f"unknown aspect {key!r}; known: {sorted(aspects)}")
    return key, aspects[key]
