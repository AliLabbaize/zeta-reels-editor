#!/usr/bin/env bash
# Make a fresh Claude Code session (web or local) able to run the test suite
# without a manual setup round-trip. Idempotent and quiet on the happy path.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 0

if [ ! -x .venv/bin/python ]; then
  command -v uv >/dev/null 2>&1 && uv venv .venv -q >/dev/null 2>&1
fi
if [ -x .venv/bin/python ] && ! .venv/bin/python -c "import pytest" >/dev/null 2>&1; then
  VIRTUAL_ENV=.venv uv pip install -q pytest pyyaml numpy pillow jiwer >/dev/null 2>&1
fi

missing=()
command -v ffmpeg  >/dev/null 2>&1 || missing+=("ffmpeg")
command -v ffprobe >/dev/null 2>&1 || missing+=("ffprobe")
[ -n "${GEMINI_API_KEY:-}" ] || { [ -f .env ] && grep -q '^GEMINI_API_KEY=.' .env; } || missing+=("GEMINI_API_KEY")

if [ ${#missing[@]} -gt 0 ]; then
  echo "zeta: not available in this session: ${missing[*]}"
  echo "zeta: offline paths still work (pytest, plan on cached transcripts, geometry)."
fi
exit 0
