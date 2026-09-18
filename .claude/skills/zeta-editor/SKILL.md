---
name: zeta-editor
description: Edit Zeta Darija facecam news videos end to end - transcribe (Gemini + WhisperX), learn the editing style from past videos, derive cuts from an edited transcript, place verified source screenshots, burn word-level Arabic captions, render vertical for Instagram. Use for any work inside zeta-reels-editor, or whenever the task is cutting a Darija talking-head video, sourcing on-screen screenshots, or Arabic caption timing.
---

# Zeta editor

Adapted from `browser-use/video-use` SKILL.md. Their production rules are kept;
the Zeta specifics are the Darija transcription stack, the style learner, the
verified-screenshot pipeline and the Instagram-first output.

## Principle

1. **Gemini supplies the text, wav2vec2 supplies the time.** No model timestamp
   ever survives into a cut, a caption or an overlay.
2. **Edit text, derive cuts.** The editor rewrites the transcript by deleting;
   `diff_align` turns that into ranges. An LLM that paraphrases is a caught
   error (`DiffResult.invented`), not a silent rewrite of what Ali said.
3. **Learn before you apply.** `zeta learn` measures Ali's real cut ratio, his
   real filler policy, his real insert rate. The profile beats anyone's taste.
4. **Nothing unverified ships.** Every screenshot carries a URL, a claim and a
   vision verdict. Zeta is a news channel; a wrong screenshot is a correction,
   not a blemish.
5. **Ask, confirm, execute, iterate, persist** - unless `--auto`, where the
   decision report is the review.

## Hard rules

The twelve rules in `CLAUDE.md` are correctness, not taste. Rules 1-5 are
implemented inside the vendored `helpers/render.py`; never edit it.

## Session start

1. Read `edit/project.md` if it exists, summarise the last session in one line.
2. Verify `GEMINI_API_KEY`, `ffmpeg`/`ffprobe`, and that `.venv` has the deps
   the requested stage needs (`align` for transcription, `shots` for
   screenshots, `learn` for style learning).
3. Never write anything inside the repo; all artifacts go to
   `<videos_dir>/edit/` (rule 11).

## The process

1. **Ingest.** `zeta ingest` probes every source and extracts 16 kHz mono WAV on
   the original timeline. Cache key is the source sha256 (rule 9).
2. **Transcribe + align.** `zeta transcribe`. Gemini VERBATIM Darija (Arabic
   script for Darija, Latin for French/English), WhisperX forced alignment with
   the Arabic wav2vec2 model, `interpolate_method="ignore"`. `qa_words` is a
   gate, not a report: coverage >= 95%, zero overlaps, no word over 2 s.
3. **Pack.** `takes_packed.md` - phrase lines with a gap column in ms and a
   language tag. This is the planner's reading view.
4. **Learn (once, then when the style drifts).** `zeta learn --pairs pairs.csv`
   over 3-10 raw/published pairs writes `style_profile.json` and
   `few_shot_examples.md`.
5. **Plan.** `zeta plan` returns `kept_text`, insert markers anchored to exact
   words, and a plain-English strategy. Print the strategy and wait (rule 10).
6. **Derive cuts.** Word-boundary snapping, 30-200 ms padding, silences >= 400 ms
   preferred, 150-400 ms flagged for a `timeline_view` check, never below 150 ms.
7. **Research + capture + verify.** Owner's own source first, `links.txt` before
   any search. Capture width comes from the destination aspect, not the desktop
   (see the arithmetic in `configs/layout.yaml`). Vision-verify, retry once with
   a different selector, then drop and flag.
8. **Render.** `helpers/render.py` unchanged: per-segment extract -> lossless
   concat -> overlays -> captions last.
9. **Self-eval.** `timeline_view` on the rendered output at every cut boundary
   and every insert edge. Cap at 3 fix-and-re-render passes, then flag.
10. **Report + persist.** `decision_report.html` and a `project.md` section.

## Darija specifics

* **Script mixing is normal.** Darija in Arabic script, French and English in
  Latin, inside one sentence. `textnorm` matches on sound, not spelling: the
  four alef forms, taa marbuta vs haa, diacritics and Arabic-Indic digits all
  collapse before any diff.
* **Never trust a single ASR pass on names and numbers.** Enable the Cohere
  second opinion; disagreement marks the spans worth a human look, instead of
  re-reading the whole transcript.
* **Alignment aliases.** Digits, Latin names and French words get a grapheme
  alias the Arabic acoustic model can actually align. Seed them from
  `configs/transcribe.yaml: custom_vocabulary`.
* **Captions must be tested on bidi samples.** A mixed RTL/LTR line can render
  correctly in one player and scrambled in another. `captions.check_bidi` lists
  them; look at three before trusting a style change.

## Instagram specifics

* Default output is 9:16 at 1080x1920. The safe area in `configs/layout.yaml`
  is UI geometry, not taste: the Reels caption block, the tab bar and the right
  action rail genuinely cover those bands.
* Screenshots are **fitted, never cropped** (`fit_card`). Cropping is only
  available in `fullframe_pip`, for images whose edges carry no information.
* Capture width is derived from the safe width so that one CSS pixel of the page
  survives as at least one device pixel in the delivered frame. A desktop-width
  capture halves every glyph and the insert becomes decoration.

## Anti-patterns

* Letting the planner emit timestamps.
* A second copy of the diff logic, or of the output-time arithmetic.
* Editing the vendored render path instead of wrapping it.
* Shipping an insert whose verification failed.
* Re-transcribing an unchanged source.
* Deciding the transcription backend from vibes instead of `zeta benchmark`.
