# Where the build departs from the spec

Each of these is a deliberate call made while implementing
`docs/ZETA_AUTO_EDITOR_SPEC.md`. They are listed so they can be overruled, not
to be quietly absorbed.

## 1. EDL paths are relative to `edit/`, not prefixed with `edit/`

The spec's EDL example writes `"file": "edit/screenshots/slot_01/overlay.mp4"`
and `"subtitles": "edit/captions/final.ass"`. The vendored `render.py` resolves
relative paths against the directory holding `edl.json`, which **is**
`<videos_dir>/edit/`, so the spec's spelling resolves to
`<videos_dir>/edit/edit/...`. Overlays would fail and the subtitle burn would be
skipped with only a warning. We emit `screenshots/<slot_id>/overlay.mp4` and
`captions/final.ass`. Absolute paths also work. Changing this back would require
editing `render.py`, which is frozen.

## 2. Screenshot capture width is derived from the destination

The spec says width 1600. A 1600 px capture fitted into the 820 px safe column
of a Reel halves every glyph, and an unreadable screenshot is decoration rather
than evidence. `configs/layout.yaml` derives capture width per aspect from
`safe_width / (capture_width * retina) >= 1.0`: 800 CSS px @2x for 9:16, 900 for
4:5 and 1:1, and the spec's 1600 for 16:9, where it is correct.

## 3. Default insert layout is `fit_card`, and it never crops

Owner's decision, encoded: the screenshot is fitted inside the Instagram safe
area over a blurred, dimmed facecam still, with the facecam kept as a PiP.
Cropping is available only in the opt-in `fullframe_pip`, because a cropped
headline can change what a source appears to say.

## 4. Caption styling is written as inline ASS override tags

`render.py` burns with `force_style=FontName=Helvetica,FontSize=18,MarginV=90`.
Helvetica has no Arabic coverage, and MarginV=90 on a 1920-tall frame puts the
caption inside the Instagram UI band. `force_style` overrides the `[V4+ Styles]`
block but **not** inline override tags, so font, size, position, border and
weight are written per cue as `{\fn…\fs…\pos…\bord…\b…}`. The style block still
carries the right values for anyone opening the file by hand.

## 5. Caption font sizes are scaled from the video-use convention

`configs/captions.yaml: font_size` follows upstream, where libass' default
`PlayResY` is 288. We set `PlayResY` to the real frame height and scale font
size by `PlayResY/288`, giving 147 px at 9:16. Read literally, 22 px on a
1920-tall frame would be invisible.

## 6. Alignment is cached like transcription

Hard rule 9 covers transcripts. Forced alignment of a ten minute take on CPU
costs more than the transcription did, so `zeta edit` also reuses a `words.json`
whose recorded source sha256 still matches the file. `--force` re-does both.

## 7. Uncertain spans live in `WordsDoc.meta`, not on the word

The `Word` contract has no "uncertain" field and `confirmed_by` means something
else, so second-opinion disagreements are recorded as index spans in
`meta["uncertain"]` with both readings and the recheck result.

## 8. One place a model still emits seconds

Learn mode's published-only fallback (Gemini agentic video understanding) returns
timestamped cuts and inserts, as the spec intends. Those numbers only ever become
aggregate statistics in `style_profile.json` — cut ratio, inserts per minute,
median durations. They never become a cut, a caption or an overlay position.
Scene detection overrides them wherever both exist.

## 9. Not yet done, and why

* **The Stage 1 benchmark has not been run.** It needs `GEMINI_API_KEY` and
  three past videos with human-corrected transcripts.
  `configs/transcribe.yaml` therefore still names `gemini_flash_lite` as the
  default, which is the spec's v2 recommendation and not a measured result.
  `zeta benchmark --manifest …` fills the table. Do not decide from vibes.
* **Alignment quality and the face-presence check in learn mode are untested on
  real footage.** The fixtures are synthetic: tone bursts and a flat insert card,
  with no face and no speech.
* **`shot-scraper`'s multi-YAML keys and the cookie-dismissal JS** have not been
  run against a live browser. One smoke run once `[shots]` is installed.
