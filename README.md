# Zeta reels editor

Cuts, captions and sources a Zeta facecam news video end to end.

Raw Darija footage in (with French and English code switching), out comes a cut,
word-level-captioned vertical video with **verified** screenshots of the sources
placed where they are mentioned, plus a report showing every cut and every
insert with its reason.

It learns Ali's editing style from past videos first, then applies it.

## How it works

```
ingest -> transcribe (Gemini VERBATIM) -> align (WhisperX) -> QA -> pack
      -> [learn: diff raw vs published -> style_profile.json]
      -> plan (the model rewrites the transcript; it never emits a timestamp)
      -> derive cuts (text diff) -> research + screenshots -> verify
      -> render (segments -> concat -> overlays -> captions LAST)
      -> self evaluation (<= 3 passes) -> report -> project.md
```

Two ideas carry the whole thing:

* **Gemini supplies the text, wav2vec2 supplies the time.** Language models are
  reliable editors of text and unreliable reporters of timestamps, so the editor
  returns the transcript with material deleted and the cuts are *derived* by
  diffing that against the acoustically aligned words.
* **Learning and applying are the same diff.** A published video is the raw take
  with material deleted, which is exactly what the planner produces. One engine
  (`helpers/diff_align.py`) serves both, so a fix lands in both.

## Install

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e '.[llm,align,shots]'     # add ,learn for style learning
shot-scraper install                        # Playwright browser, for screenshots
cp .env.example .env                        # then paste GEMINI_API_KEY
```

The key is read from the environment first and from `.env` at the repo root
second, so an exported or cloud-injected `GEMINI_API_KEY` always wins over the
file. `.env` is gitignored; never put a key in `<videos_dir>`, those get shared.

Needs `ffmpeg` and `ffprobe` built with libass, fribidi and harfbuzz: libass
burns the captions, fribidi reorders RTL, harfbuzz shapes the Arabic. You also
need the caption font itself (`fonts-noto-core` on Debian/Ubuntu) - libass
substitutes silently when it is missing, so the captions burn in the wrong face
rather than failing. Python 3.11.

In a Claude Code session, `scripts/session_start.sh` installs all of that for
you on startup.
WhisperX alignment runs on CPU; a GPU only makes it faster.

## Use

```bash
zeta fetch https://www.instagram.com/reel/... -v learn/published --name ep14 --pairs pairs.csv
zeta learn --pairs pairs.csv --out style_profile.json
zeta edit raw01.mp4 --profile style_profile.json --links links.txt --aspect 9:16
zeta edit raw01.mp4 --auto                  # unattended; the report is the review
zeta review edit/edl.json                   # open the decision report
```

`fetch` downloads a published video for learn mode. A published-only row teaches
insert density, layout and triggers, but not the cut ratio or the filler policy:
a finished video does not record what was removed, so those need the raw take.

Stages also run on their own: `zeta ingest`, `zeta transcribe`, `zeta plan`,
`zeta research`, `zeta render`, `zeta benchmark`.

Everything lands in `<videos_dir>/edit/`: `final.mp4`, `preview.mp4`, `edl.json`,
`transcripts/*.words.json`, `takes_packed.md`, `captions/final.{ass,srt}`,
`screenshots/<slot>/`, `decision_report.html`, `project.md`.

## Output format

Instagram-first: 9:16 at 1080x1920 by default, 4:5, 1:1 and 16:9 from the same
EDL. Screenshots are **fitted, never cropped** - a cropped headline can change
what a source appears to say - and capture width is derived from the delivered
safe area so page text survives at readable size on a phone. The geometry is in
`configs/layout.yaml` with the arithmetic explained.

## Verification

No screenshot ships unverified. Each insert carries the claim, the resolved URL
(the owner's own page before any press coverage), and a vision verdict with its
evidence. An overlay whose `verified` is not `true` fails EDL validation, so it
cannot reach the render even by accident.

## Development

```bash
.venv/bin/python -m pytest tests/ -q       # no network, no ffmpeg needed
```

`helpers/render.py` and `helpers/timeline_view.py` are vendored verbatim from
[browser-use/video-use](https://github.com/browser-use/video-use) (MIT) and are
frozen by `tests/test_vendor_unchanged.py`. Behaviour changes go in wrappers.
See `CLAUDE.md` for the build rules, `docs/ZETA_AUTO_EDITOR_SPEC.md` for the
full design, and `docs/DEVIATIONS.md` for where the build departs from it and
why.
