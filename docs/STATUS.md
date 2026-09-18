# Status

Updated 2026-09-18. Branch `claude/gracious-ptolemy-k838op`.

## Where the build is

Every stage of `docs/ZETA_AUTO_EDITOR_SPEC.md` is implemented and wired behind
`zeta` (see `cli.py --help`). 364 tests pass. The pipeline renders end to end.

What is **verified working**:

* Fixture take -> cuts -> captions -> a real ffmpeg render, checked by decoding a
  frame: Darija shapes, joins, reads right to left, karaoke highlight tracks the
  spoken word, caption baseline clear of the Reels UI band
  (`tests/test_end_to_end.py`).
* `zeta edit` leaves every documented artifact and writes nothing into the repo
  (`tests/test_cli.py`).
* `GEMINI_API_KEY` valid; `gemini-3.5-flash-lite` and `gemini-3.1-flash-lite`
  answer real calls, including a real Arabic WAV.
* ffmpeg with libass, fribidi and harfbuzz; Noto Sans Arabic installed by
  `scripts/session_start.sh`.

## What has never run on real material

This is the honest list. Everything here is written and unit-tested; none of it
has met Ali's footage.

1. **WhisperX alignment on Darija.** The timing authority for every artifact.
   Expect the alias map (digits, Latin names, French words handed to an Arabic
   acoustic model) to need work. `helpers/align_whisperx.py`.
2. **The QA gate on real speech** - coverage >= 95%, zero overlaps, no word over
   2 s. It fails the run by design, so it is the first thing that will stop a
   real take.
3. **`gemini_transcribe`** after the part-reading fix (commit "Read the
   transcription models' answers"). Diagnosed live, fixed against a stub;
   needs one real call to confirm.
4. **The Stage 1 benchmark.** Never run. `configs/transcribe.yaml` names
   `gemini_flash_lite` as default because the spec recommends it, not because it
   won anything. Needs 3 clips with human-corrected verbatim transcripts.
5. **Learn mode on real pairs.** Scene detection and the face-presence check
   cannot be rehearsed on the synthetic fixtures - they have no face.
6. **Screenshots.** shot-scraper's multi-YAML keys, the cookie-dismissal JS and
   the vision verification have never hit a live page.
7. **`zeta fetch`.** Instagram and TikTok are blocked by the build environment's
   network policy; only the argv construction and the CSV rows are tested.

## Next, in order

1. Short Darija clip -> `zeta transcribe`. Fix whatever the QA gate catches.
   That is where the alias map gets real.
2. Same clip -> `zeta edit --auto --no-screenshots` for a full render on real
   speech.
3. `zeta learn --pairs pairs.csv` once Ali has 3+ raw/published pairs. Raw takes
   are what teach the cut ratio and filler policy; a published-only row cannot.
4. `zeta benchmark` once the human-corrected transcripts exist. Do not change
   the default backend before this.
5. Screenshots last: they need a live browser and are the only stage that can
   put a factual error on screen.

## Waiting on Ali

* 3-10 raw/published pairs plus `pairs.csv` (format in `helpers/fetch.py`).
* 3 clips with **verbatim** human-corrected transcripts - every filler kept. A
  cleaned-up reference measures each backend's filler habits instead of its
  accuracy, and picks the wrong default.
* Open spec questions: may `--auto` publish drafts without confirmation?
  Cohere second opinion local or API? Neither blocks anything.

## Read before changing anything

`CLAUDE.md` (hard rules, module map), `docs/DEVIATIONS.md` (ten deliberate
departures from the spec, with reasons), `docs/VENDOR.md` (why
`helpers/render.py` and `helpers/timeline_view.py` are frozen).
