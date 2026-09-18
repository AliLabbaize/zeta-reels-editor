#!/usr/bin/env bash
# Bring a fresh container up to where the pipeline can actually run.
#
# Every session starts from the same missing state: no venv, no ffmpeg, no
# Arabic font, no google-genai. Reporting that costs a round trip each time, so
# this installs it. Idempotent, quiet on the happy path, and never fatal: a
# session with no root or no network still has to start.
#
# Skip with ZETA_SKIP_SETUP=1.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 0
[ "${ZETA_SKIP_SETUP:-0}" = "1" ] && exit 0

log() { echo "zeta: $*"; }

# -- python env ---------------------------------------------------------------
if [ ! -x .venv/bin/python ] && command -v uv >/dev/null 2>&1; then
  uv venv .venv -q >/dev/null 2>&1
fi
if [ -x .venv/bin/python ]; then
  if ! .venv/bin/python -c "import pytest" >/dev/null 2>&1; then
    log "installing test deps"
    VIRTUAL_ENV=.venv uv pip install -q pytest pyyaml numpy pillow jiwer >/dev/null 2>&1
  fi
  # google-genai is the [llm] extra. Without it every model call fails at the
  # import, which looks like a missing key and is not.
  if ! .venv/bin/python -c "import google.genai" >/dev/null 2>&1; then
    log "installing google-genai"
    VIRTUAL_ENV=.venv uv pip install -q google-genai >/dev/null 2>&1
  fi
fi

# -- ffmpeg -------------------------------------------------------------------
# libass burns the captions; fribidi reorders RTL; harfbuzz shapes the Arabic.
# Ubuntu's build has all three, so the stock package is enough.
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  if [ "$(id -u)" = "0" ]; then
    log "installing ffmpeg"
    apt-get install -y -qq ffmpeg >/dev/null 2>&1 \
      || { apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq ffmpeg >/dev/null 2>&1; }
  fi
fi

# -- Arabic fonts -------------------------------------------------------------
# configs/captions.yaml asks for Noto Sans Arabic. Without it libass silently
# substitutes whatever it can find, and the captions burn in the wrong face with
# the wrong metrics - which looks like a styling choice, not a missing package.
if command -v fc-list >/dev/null 2>&1; then
  if ! fc-list :lang=ar family 2>/dev/null | grep -qi "noto sans arabic"; then
    if [ "$(id -u)" = "0" ]; then
      log "installing Arabic fonts"
      apt-get install -y -qq fonts-noto-core >/dev/null 2>&1 \
        || { apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq fonts-noto-core >/dev/null 2>&1; }
      fc-cache -f >/dev/null 2>&1
    fi
  fi
fi

# -- report what is still missing ---------------------------------------------
missing=()
command -v ffmpeg  >/dev/null 2>&1 || missing+=("ffmpeg")
command -v ffprobe >/dev/null 2>&1 || missing+=("ffprobe")
fc-list :lang=ar family 2>/dev/null | grep -qi "noto sans arabic" || missing+=("Noto Sans Arabic")
[ -n "${GEMINI_API_KEY:-}" ] || { [ -f .env ] && grep -q '^GEMINI_API_KEY=.' .env; } || missing+=("GEMINI_API_KEY")

if [ ${#missing[@]} -gt 0 ]; then
  log "still missing: ${missing[*]}"
  case " ${missing[*]} " in
    *GEMINI_API_KEY*) log "set GEMINI_API_KEY on the environment, or put it in .env at the repo root";;
  esac
  log "offline paths still work (pytest, plan on cached transcripts, geometry)"
fi

# Heavy extras stay opt-in: whisperx pulls torch (~2 GB) and most sessions never
# transcribe. Install with: uv pip install -e '.[align,shots,learn]'
exit 0
