"""Read `.env` at the repo root, without a dependency.

The README has always said to put `GEMINI_API_KEY` in `.env`, so the code has to
actually read it. Real environment variables win: a key exported in the shell,
or injected by a cloud environment, is more specific than a file checked out on
disk, and silently overriding it would be very hard to debug.

Never read a `.env` from `<videos_dir>` -- footage directories get shared.
"""

from __future__ import annotations

import os
from pathlib import Path

from .paths import REPO_ROOT

_LOADED = False


def parse_env(text: str) -> dict[str, str]:
    """Minimal KEY=VALUE parsing: comments, blank lines, quotes, `export`."""
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            out[key] = value
    return out


def load_env(path: str | Path | None = None, *, force: bool = False) -> dict[str, str]:
    """Load `.env` into `os.environ` once, without clobbering what is already set."""
    global _LOADED
    if _LOADED and not force and path is None:
        return {}
    env_path = Path(path) if path else (REPO_ROOT / ".env")
    if path is None:
        _LOADED = True
    if not env_path.exists():
        return {}
    applied = {}
    for key, value in parse_env(env_path.read_text(encoding="utf-8")).items():
        if os.environ.get(key):
            continue
        os.environ[key] = value
        applied[key] = value
    return applied
