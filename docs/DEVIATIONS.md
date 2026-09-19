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

## 9. `gemini_transcribe` answers in a part the SDK does not fold into `.text`

The transcription models return an `audio_transcription` part, not a text part,
so `resp.text` is empty for a perfectly good transcript. The client reads the
parts directly before giving up. An unusable answer is also no longer retried:
the same request returns the same shape four times, each after a longer backoff,
and the real reason arrives minutes late. Retries still apply to transport
failures, and the fallback model still gets its turn.

**Confirmed against the live API (2026-09-18), and it is not the whole story.**
`gemini-3.5-transcribe` does answer with `resp.text is None` and the transcript
in `candidates[0].content.parts[0].audio_transcription.text`; `_response_text`
reads it, and the SDK's own "non-text parts in the response" warning names the
same part. But the backend still cannot complete a call: the model rejects
`system_instruction` (400 `Developer instruction is not enabled for this
model`) and JSON mode (400 `JSON mode is not enabled for this model`), both of
which `_gemini_chunked` sends unconditionally, and it answers in plain text
rather than the `{"segments": [...]}` the parser needs. So the part reader is
right and currently unreachable in production. See `docs/STATUS.md` item 3;
untangling it is a design decision, because hard rule 8's VERBATIM instruction
has nowhere to go once `system_instruction` is refused.

## 10. Not yet done, and why

* **The Stage 1 benchmark has not been run.** It needs `GEMINI_API_KEY` and
  three past videos with human-corrected transcripts. It also needs
  `gemini_transcribe` to work at all - see deviation 9 - since that backend is
  one of the candidates being scored.
  `configs/transcribe.yaml` therefore still names `gemini_flash_lite` as the
  default, which is the spec's v2 recommendation and not a measured result.
  `zeta benchmark --manifest …` fills the table. Do not decide from vibes.
* **Alignment quality and the face-presence check in learn mode are untested on
  real footage.** The fixtures are synthetic: tone bursts and a flat insert card,
  with no face and no speech.
* **`zeta fetch` has not been run against a live platform.** Instagram and
  TikTok are blocked by this build environment's network policy (403 at the
  proxy), so only the argv construction and the pairs.csv rows are tested.
  Expect to need `--cookies-from-browser` for Instagram.
* **`shot-scraper`'s multi-YAML keys and the cookie-dismissal JS** have not been
  run against a live browser. One smoke run once `[shots]` is installed.

## 11. `gemini_flash_lite` runs gemini-3.1-flash-lite first, not 3.5

Found on the first real take (`IMG_5829.MOV`, 28 s, 2026-09-18). With the
spec's prompt, both flash-lite models wrote Darija in Arabizi (`l9aw`, `m3a`,
`ba3diyatohom`) and ignored "Arabic script". The Arabic wav2vec2 model cannot
align that honestly, and `pack_transcripts` tagged the Darija phrases `[fr]`
and `[en]`. With one concrete mixed-script example added to `SYSTEM_PROMPT`,
3.1 answered in Arabic script with French and English left in Latin
(`part 2`, `agents`, `packages`), identically on two runs at temperature 0.
3.5 stayed in Arabizi. So the backend is unchanged and only its model order
is swapped in `configs/transcribe.yaml`. This is not a benchmark result: it
chooses the only model whose output meets the spec's script requirement.
Known slip: 3.1 rendered the French opener "vraiment" as "فعلا", which is a
translation and not verbatim. `zeta benchmark` should still score both models.

## 12. The cold-start profile has no cut_ratio target

`cli._load_profile` used to invent `cut_ratio: 0.25` when there was no
`style_profile.json`. On the first real take (28 s, few fillers) the planner's
first answer was a clean 0.11 filler trim. Validation rejected it as "cut more
aggressively", and the retry broke its own insert anchor while cutting real
content. Until `zeta learn` has measured Ali's ratio, the cold start sets
`None`: `derive_cuts` already treats that as "no band", and the prompt says
to cut only what the rules say.

## 13. Source search runs on Claude Code, not Gemini grounding

Ali's Gemini key gets 429 `RESOURCE_EXHAUSTED` on every `google_search`
grounded call, while plain calls succeed: the key's tier has no search quota.
Ali pays for Claude, not for Gemini search, and wants Gemini kept to
transcription. `research._claude_search` therefore runs `claude -p` (headless
Claude Code with WebSearch/WebFetch, structured output via `--json-schema`) when
`configs/sources.yaml: search.backend` is `claude_cli`. It runs from a temp dir
with no setting sources, so no hooks or plugins load. This is a second LLM door
beside `gemini_client`, confined to the one function `resolve_source` already
marked as the swap point. The candidates go through the same allowlist and
blocklist, and vision verification stays on Gemini. Under `ZETA_LLM_MOCK` the
Gemini mock is used, so tests stay offline.

The first live run also showed the planner inventing `prefer_source` URLs
(`zeta.ma/tech-updates`) and claims ("vulnerability") the speaker never made.
Both are now forbidden in `plan_edit.SYSTEM_PROMPT`.

## 14. `fast_render` reuses the cut segments; render.py still does the work

Rule "per segment extract then lossless concat" is unchanged: `helpers/fast_render.py`
calls `render.extract_all_segments`, `render.concat_segments`,
`render.build_final_composite` and `render.apply_loudnorm_two_pass` in that
order and adds one thing, a cache key over (ranges, sources, grade, fps,
preview/draft) written next to `base.mp4`. A revision that only changes cards or
captions skips the extraction (~2 min of a ~4.5 min part). `--fresh` forces it.
render.py itself is untouched (docs/VENDOR.md); `tests/test_fast_render.py`
proves the base is reused only when the cut is identical.

## 15. Alignment runs on the Apple GPU when there is no CUDA

`resolve_device` now answers `mps` before `cpu`. Measured on the 43 s outro:
cpu 45.3 s, mps 15.4 s, word times identical to the millisecond (max diff
0.000 s). `alignment.device` in `configs/transcribe.yaml` still overrides.

## 16. Screenshot capture and verification run four at a time

`cli.stage_research` walks every slot's candidates in a thread pool: each slot
is a browser page plus a vision call and touches only its own directory. On the
6 min two-parter this was the slowest part of research.
