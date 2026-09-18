# Status

Updated 2026-09-18. Branch `claude/trusting-wright-rbpwnt`.

## Where the build is

Every stage of `docs/ZETA_AUTO_EDITOR_SPEC.md` is implemented and wired behind
`zeta` (see `cli.py --help`). 367 tests pass. The pipeline renders end to end.

What is **verified working**:

* Fixture take -> cuts -> captions -> a real ffmpeg render, checked by decoding a
  frame: Darija shapes, joins, reads right to left, karaoke highlight tracks the
  spoken word, caption baseline clear of the Reels UI band
  (`tests/test_end_to_end.py`).
* `zeta edit` leaves every documented artifact and writes nothing into the repo
  (`tests/test_cli.py`).
* `zeta ingest` accepts an **audio-only** file (tested with `.m4a`): it probes
  0x0 with no video stream and extracts the 16 kHz mono WAV as usual. Since
  every downstream stage reads only that WAV (`ingest.py:44`), the whole of
  Stage 1 can be exercised from audio, at roughly 150 KB/minute as Opus - which
  matters when the only way in is a 30 MB chat upload.
* `GEMINI_API_KEY` valid; `gemini-3.5-flash-lite` and `gemini-3.1-flash-lite`
  answer real calls, including a real Arabic WAV. A 13 s synthesised Arabic clip
  came back as 15 words of Arabic-script Darija through the `gemini_flash_lite`
  backend - the **default** text source, so Stage 1 has a working backend.
* The `audio_transcription` part reader (`LLM._response_text`). Confirmed
  against the live API, not a stub: `gemini-3.5-transcribe` answers with
  `resp.text is None` and the transcript in
  `candidates[0].content.parts[0].audio_transcription.text`; `_response_text`
  returns it. The SDK prints its own warning about the non-text part, which is
  independent confirmation of the shape.
* ffmpeg with libass, fribidi and harfbuzz; Noto Sans Arabic installed by
  `scripts/session_start.sh`.
* The `[align]` extra installs: torch 2.8.0+cu128 (CPU only here,
  `cuda.is_available() == False`), torchaudio 2.8.0, whisperx 3.8.6. It pulls
  matplotlib transitively via pyannote-audio, which is why the three
  `tests/test_screenshot.py` matplotlib tests no longer skip: the suite is now
  367 passed, 0 skipped, from the same 367 tests as 364 passed / 3 skipped.

## What has never run on real material

This is the honest list. Everything here is written and unit-tested; none of it
has met Ali's footage.

1. **WhisperX alignment on Darija - and it cannot run in this container at
   all.** `huggingface.co` is blocked by the environment's egress policy
   (ProxyError, not a 404), so `wx.load_align_model` cannot fetch
   `jonatasgrosman/wav2vec2-large-xlsr-53-arabic` and raises the misleading
   "could not be found in huggingface ... or torchaudio". `download.pytorch.org`
   is blocked too, so the `english_model` torchaudio bundle is no fallback, and
   `hf-mirror.com` is blocked as well. PyPI and GitHub are reachable; this is
   specific to the model hosts.

   Verified end to end on 2026-09-18 with a synthesised Arabic clip: `zeta
   ingest` and Gemini transcription both succeeded and wrote
   `transcripts/ar_clip.words.json` with `"aligner": null`, then
   `cli.py:140 -> align_whisperx.align` died. So Stage 1's text half works here
   and its time half cannot, which means **no footage can complete `zeta
   transcribe` in this environment** until either `huggingface.co` is
   allow-listed for the environment
   (https://code.claude.com/docs/en/claude-code-on-the-web) or the model cache
   is populated some other way. Nothing about the alias map, the QA gate or the
   Darija quality can be learned until then. `helpers/align_whisperx.py`.

   Separately, and still true once it does run: expect the alias map (digits,
   Latin names, French words handed to an Arabic acoustic model) to need work.
2. **The QA gate on real speech** - coverage >= 95%, zero overlaps, no word over
   2 s. It fails the run by design, so it is the first thing that will stop a
   real take.
3. **`gemini_transcribe` is broken against the live API, and the part-reading
   fix is not what is wrong with it.** The part reader works (above). The
   backend around it cannot complete a call, for three independent reasons
   found by real calls to `gemini-3.5-transcribe`:
   * `system_instruction` -> 400 `Developer instruction is not enabled for this
     model`. `_gemini_chunked` always sends `SYSTEM_PROMPT`, so every call dies
     here. This is what the earlier "empty response" diagnosis was standing in
     front of.
   * `response_schema` / `response_mime_type` -> 400 `JSON mode is not enabled
     for this model`. Both Gemini backends share `_gemini_chunked`, which always
     sends the schema.
   * Even with both removed the model answers **plain text**, not
     `{"segments": [...]}`. `LLM._coerce` would raise `model did not return
     JSON`, and `_parse_segments` expects segments. There is no timestamped
     segment structure to stitch, so `plan_chunks`/`stitch_chunks` have nothing
     to work with and `word_timestamp: true` has no observed effect.

   Fixing this is a design decision, not a patch: the VERBATIM instruction of
   hard rule 8 cannot be delivered as a system instruction, and prompting may
   not work at all - audio-only and audio-plus-prompt returned byte-identical
   answers. Not urgent: `gemini_transcribe` is only a benchmark candidate and
   is not the default backend.

   Quality note, weak evidence: on the synthesised clip it returned only
   `Assalamu alaikum.` - the first phrase of 13 s, in Latin transliteration
   rather than Arabic script - identically across repeat runs. The audio is
   espeak-ng, not a human, so this does not predict real Darija; but
   `gemini_flash_lite` transcribed the whole of that same file.
4. **A 4xx from the SDK is retried like a transport failure.** The 400s above
   each took four attempts with exponential backoff before surfacing. Deviation
   9 fixed exactly this for `LLMResponseError`; a `ClientError` still lands in
   the generic `except Exception` branch of `_call_with_retry`. Cheap to fix,
   and it is what made the first diagnosis slow.

5. **The Stage 1 benchmark.** Never run. `configs/transcribe.yaml` names
   `gemini_flash_lite` as default because the spec recommends it, not because it
   won anything. Needs 3 clips with human-corrected verbatim transcripts.
6. **Learn mode on real pairs.** Scene detection and the face-presence check
   cannot be rehearsed on the synthetic fixtures - they have no face.
7. **Screenshots.** shot-scraper's multi-YAML keys, the cookie-dismissal JS and
   the vision verification have never hit a live page.
8. **`zeta fetch`.** Instagram and TikTok are blocked by the build environment's
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
