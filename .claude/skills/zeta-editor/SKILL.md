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
7. **Research the story, then find the visuals.** This is the part Ali judges
   the whole edit by. Follow "Screenshots: the approach Ali approved" below;
   never fall back to generic headlines or homepages to fill slots.
8. **Render.** `helpers/render.py` unchanged: per-segment extract -> lossless
   concat -> overlays -> captions last.
9. **Self-eval.** `timeline_view` on the rendered output at every cut boundary
   and every insert edge. Cap at 3 fix-and-re-render passes, then flag.
10. **Report + persist.** `decision_report.html` and a `project.md` section.

## Screenshots: the approach Ali approved (2026-09-19)

Screens are a **supporting visual for what Ali says at that moment**, not
decoration and not random headlines. Worked on the Hugging Face hack video
(`~/Movies/zeta/IMG_5824`): 13 cards, approved in one pass.

1. **Read the script first.** Go through the transcript moment by moment and
   write down each claim Ali makes (what happened, a number, who did what).
2. **Understand the real story.** Find the ORIGINAL source of the information:
   the incident / technical report, the official disclosure, the paper, the
   filing, the repo, the post that started it. Read it. Also read Ali's own
   Substack article on the topic (zeta233.substack.com, list posts via
   `/api/v1/archive`): it cites the exact posts and headlines he refers to.
3. **Match each claim to the passage that proves it.** Have Gemini read the full
   source text and return the verbatim sentence(s) + page for each moment; check
   them yourself. The right visual type depends on the moment:
   * a statement of fact -> the paragraph from the original report/disclosure,
     key sentence **highlighted in yellow**, source + page on a small top line;
   * "the headlines say..." -> the actual post/headline he means (his article
     links it); X blocks bots, crop the embed from his Substack instead;
   * a company/product named -> its official page **with its logo**;
   * a number -> the source line that states it (e.g. "198 of 898 tasks");
   * his own article -> his Substack headline block;
   * code, a tool, a benchmark -> the repo README / docs section.
   Different videos need different types: do not assume it is always a report.
4. **Flag accuracy.** If the source says something different from the script,
   tell Ali before rendering (e.g. "they did investigate on July 5; the choice
   not to stop the run was June 27"). Zeta is a news channel.
5. **Crop for legibility.** Crop exactly the information: headline block
   (`screenshot.capture_headline_block`: kicker, headline, subtitle, byline,
   stop at the lead picture, 2x, viewport clip), PDF passages snapped to whole
   lines, title pages with the empty space packed out. Never cut text in half,
   never capture a fixed window. Look at every card yourself before Ali does.
6. **Density.** About 3 per minute across the whole video (his learned profile
   is `~/Movies/zeta/learn/style_profile.json`). 3 cards for 3 minutes is not
   enough.
7. **Review before rendering.** Show every card on the review page
   (`helpers/review_shots.py`, `zeta edit --review-shots`): time, what he says,
   the card, Keep/Drop, feedback, "use this link". Any feedback -> do not
   render; fix and show again. Render only after a clean approval.

Look (learned from his published reel): bare cut-out cards (`cutout` layout, no
card/shadow), over the chest band, captions clear below, top band free for his
title text. English-only captions (`configs/captions.yaml: english.mode: only`).

Blocked by bots: openai.com (Cloudflare), Bloomberg, x.com, Google, DuckDuckGo.
Workarounds: the PDF on cdn.openai.com, the embed in his Substack, Bing is NOT
acceptable (mixed-story result pages).

## Iterate fast (Ali will not wait through renders)

A full render is ~4.5 min per 3 min part, so never discover a layout problem in
one. Order of work:

1. **Proof frames, seconds.** `zeta proof -v <dir> [--edl x.json] --open`
   composites the real frame + the real card + the real burned captions at every
   insert (`helpers/proof.py`). Every layout bug this pipeline shipped (card over
   his mouth, captions on a card's last line, a card too small to read) was
   visible here. Check the sheet yourself, then show him.
2. **Patch, ~30 s.** One card wrong in a rendered video? Overlay the fixed card
   on that window only: `ffmpeg -i part.mp4 -i fixed.mov -filter_complex
   "[1:v]setpts=PTS-STARTPTS+T/TB[o];[0:v][o]overlay=enable='between(t,T,T+D)'"
   -c:v h264_videotoolbox -b:v 18M -c:a copy`. A bigger card fully covers a
   smaller one in the same band.
3. **Reused cuts.** `zeta render` goes through `helpers/fast_render.py`, which
   calls render.py's own steps but skips re-cutting when ranges, sources and fps
   are unchanged (~2 min saved per revision). `--fresh` forces a re-cut.
4. **Both parts at once.** Render each part in its OWN folder (`render_pN/`):
   render.py writes `clips_graded/` and `base.mp4` next to the EDL, so parallel
   parts in one folder clobber each other.
5. **`--no-self-eval`** while iterating: the post-render vision pass costs 1-2 min.
6. Alignment runs on the Apple GPU (`mps`) automatically: 3x faster, identical
   times. Screenshot capture + verification run 4 at a time.

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

## Fact-check before he posts

Zeta is a news channel and he asks for it: check every claim against the source
and tell him what does not hold BEFORE the render. Real finds on the Hugging
Face two-parter: "10% of humanity will die" (the researcher said a >10% CHANCE),
"the engineer just restarted it" (they did investigate on July 5; the "not
required" call was June 27), and an agents-debating-ethics beat with no source
anywhere. Say it plainly, name the passage, and offer the one-line fix.

## Anti-patterns

* Letting the planner emit timestamps.
* A second copy of the diff logic, or of the output-time arithmetic.
* Editing the vendored render path instead of wrapping it.
* Shipping an insert whose verification failed.
* Re-transcribing an unchanged source.
* Deciding the transcription backend from vibes instead of `zeta benchmark`.
* Filling screenshot slots with generic headlines, homepages or search-result
  pages because the specific source was hard to capture.
* Rendering before Ali has approved the screenshots.
* Deleting sentences of Ali's without a learned cut ratio (fillers/repeats only).
* Discovering a layout problem in a render instead of in `zeta proof`.
* Re-rendering a whole part to fix one card (patch that window instead).
* Deciding a look for him ("smaller so it does not cover his face") instead of
  showing both and asking: he wants his own article card BIG, and screens at the
  TOP under his title line, mouth never covered.
