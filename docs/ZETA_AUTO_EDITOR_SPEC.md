# Zeta Auto Editor: build spec for Claude Code

Goal: end to end editor for Zeta facecam news videos (Darija with French and English code switching). Input is raw footage plus optional source links. Output is a cut, captioned video with verified screenshots of the sources placed where they are mentioned. The system first learns the editing style from a few past videos, then applies it to new raw footage.

Date: 2026 09 18. Owner: Ali (Zeta). Builder: Claude Code.

## 0. Decision summary (read first)

| Decision | Choice | Why |
| --- | --- | --- |
| Base | Fork `browser-use/video-use` (MIT, ~24k stars, "edit videos with Claude Code") | Already does transcript → packed view → LLM plan → EDL → ffmpeg render → self evaluation, with 12 production rules that took real trial and error. Do not rewrite this. |
| Transcription | Replace the ElevenLabs Scribe layer with the Gemini + WhisperX Darija pipeline, behind the same `{"words": [...]}` contract | `Moh4696/freecut` proved the transcription layer can be swapped without touching anything downstream. Scribe does not do Darija. |
| Cut logic | "Rewrite the transcript, derive the cuts" (pattern from `mdpierre/videoclieditor rewrite-edit`) | LLMs edit text reliably and hallucinate timestamps. Cuts come from diffing edited text against word level timings, never from LLM timestamps. |
| Style learning | Diff raw vs published transcripts + scene detection on published videos → `style_profile.json` + few shot examples | Same diff engine as the cut logic, so learning and applying use one code path. |
| Screenshots | Claim extraction → source resolution (owner's original source first) → `shot-scraper` capture → vision verification → overlay slot in the EDL | video-use already supports timed overlays; screenshots are just overlay clips with metadata. |
| Captions | Word level ASS burned last, times remapped to the output timeline | video-use hard rules 1 and 5. |

## 1. Reference repositories and what to take from each

| Repo | Take | Ignore |
| --- | --- | --- |
| https://github.com/browser-use/video-use | `helpers/render.py`, `pack_transcripts.py`, `timeline_view.py`, `grade.py`, EDL schema, SKILL.md hard rules, `project.md` memory pattern, self evaluation loop (max 3 passes) | ElevenLabs Scribe, color grading presets, launch video palette |
| https://github.com/Moh4696/freecut | Pluggable transcription backend design, each backend emits `{"words": [...]}` | Its whisper default (normalizes fillers, weak on Darija) |
| https://github.com/mdpierre/videoclieditor | `rewrite-edit` workflow: final transcript in, `edit_plan.json` + `decision_report.html` out; `plan-edit` preview before render | Apple Silicon only tooling |
| https://github.com/Railly/vcut | Detect report (silence, filler, clipping, black and frozen frames) as data; render gated behind approval; presets `clean` at minus 30 dB for talking head | Node packaging |
| https://github.com/helenlpang/talking-head-video-editor | Claude prompt for repeat take detection; image overlay UX | Flask UI |
| https://github.com/WyattBlue/auto-editor | Optional export of the EDL to Premiere, Resolve or Final Cut XML for manual review | Its cut logic (dB threshold only) |
| https://github.com/simonw/shot-scraper | Playwright screenshots by URL, CSS selector or JS selector; `multi` YAML batch mode | |
| https://github.com/m-bain/whisperX | Forced alignment `load_align_model()` + `align()`; latest release 3.8.6 (matches the current pipeline) | Its whisper transcription for Darija |
| https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-arabic | Arabic alignment model (current) | |
| https://huggingface.co/blog/CohereLabs/cohere-transcribe-arabic-07-2026-release | `cohere-transcribe-arabic-07-2026`, 2B, Apache 2.0, best open Arabic ASR incl. Casablanca (Moroccan) test set; second opinion transcript | Timing (it does not replace WhisperX alignment) |
| https://github.com/Breakthrough/PySceneDetect | Content detector to find screenshot inserts in past published videos | |
| https://ai.google.dev/gemini-api/docs/transcribe | `gemini-3.5-transcribe`: word level timestamps, custom vocabulary up to 1000 terms, VERBATIM mode | SMART mode (removes fillers, kills the editorial signal) |
| https://ai.google.dev/gemini-api/docs/video-understanding | Agentic video understanding (3.5 Flash Lite, 3.6, 3.7, 3.8 Flash) for cheap analysis of published videos | Static mode on long videos (about 100 tokens per second) |

## 2. Inputs and outputs

Inputs
* `raw/*.mp4` facecam footage (one or several takes)
* `links.txt` optional: URLs Ali already has for the story (skips search)
* `style_profile.json` produced by learn mode
* `configs/` (fillers, vocabulary, caption style, source allowlist)

Outputs, all in `<videos_dir>/edit/`
* `final.mp4` and `preview.mp4`
* `edl.json` (EDL v2, see section 6)
* `transcripts/<name>.words.json` canonical word level transcript
* `takes_packed.md` phrase level reading view
* `captions/final.ass`, `captions/final.srt` (output timeline)
* `screenshots/<slot_id>/` capture + `meta.json` (url, claim, verification result)
* `decision_report.html` every cut and every insert with reason
* `project.md` session memory

## 3. Pipeline

```
ingest → transcribe (Gemini) → align (WhisperX) → QA → pack
      → [learn mode: diff vs published → style_profile.json]
      → plan (LLM rewrites transcript + marks insert triggers)
      → derive cuts (diff) → research + screenshots → verify
      → render (segments → concat → overlays → captions LAST)
      → self evaluation (≤ 3 passes) → report → persist
```

### Stage 0. Ingest
* `ffprobe` every source: fps, resolution, duration, audio channels.
* Extract 16 kHz mono WAV, original timeline preserved (no trimming here, ever).
* Cache key = sha256 of the source file. Never re transcribe unchanged sources (video-use rule 9).

### Stage 1. Transcription, Darija workflow v2

Changes versus the September 2026 v1 workflow (Gemini 3.1 Flash Lite + WhisperX):

| Step | v1 | v2 | Reason |
| --- | --- | --- | --- |
| Text model | `gemini-3.1-flash-lite` | `gemini-3.5-flash-lite` (GA July 2026), fallback `gemini-3.1-flash-lite` | Newer stable lite model, same price tier. Keep the prompt: verbatim Darija in Arabic script, French and English in Latin, JSON segments `start, end, text`, timestamps approximate. Use structured output (JSON schema) instead of asking for JSON in prose. |
| New candidate transcriber | none | `gemini-3.5-transcribe` (GA August 26 2026): `mode=VERBATIM`, `word_timestamp=True`, `custom_vocabulary=[...]` | Native word timestamps and vocabulary biasing (names, tickers, French terms). Google's own docs say word timestamps degrade accuracy and Darija coverage is unverified, so this is a benchmark candidate, not the default. |
| Second opinion ASR | none | `cohere-transcribe-arabic-07-2026` (open weights, run locally or via API, 25 MB file cap via API so chunk audio) | Disagreement between Gemini and Cohere marks the "uncertain passages" automatically. Step 3 review then targets only flagged spans instead of the whole transcript. |
| Review of uncertain spans | manual | Gemini re transcribes only flagged spans from short audio excerpts (5 to 15 s with 1 s margin), then a merge step | Same idea as v1 step 3, now triggered by data. |
| Alignment | WhisperX `align()` with `jonatasgrosman/wav2vec2-large-xlsr-53-arabic`, aliases for numbers, names, French words; English spans with `WAV2VEC2_ASR_BASE_960H` bounded by neighbouring words; `interpolate_method="ignore"` | unchanged (WhisperX 3.8.6 is still the latest release) | Timing authority stays local and acoustic. Gemini supplies text, wav2vec2 supplies time. |
| QA | manual check of missing timestamps, overlaps, low confidence | scripted: `qa_words.py` fails the run if coverage < 95 %, any overlap, or any word > 2 s; re aligns flagged windows automatically once | Needed for unattended batch mode. |
| Output | SRT/ASS | canonical `words.json` (below) consumed by captions AND the cut engine | One artifact, two consumers. |

Canonical word record (matches the freecut/video-use contract, extended):

```json
{"word": "الشركة", "display": "الشركة", "alias": "asharika", "start": 12.41, "end": 12.79,
 "score": 0.91, "lang": "ary", "src": "gemini", "confirmed_by": ["cohere"]}
```

Benchmark to run once, on 3 old videos with human corrected transcripts: WER and alignment coverage for (a) v1 pipeline, (b) gemini-3.5-flash-lite + WhisperX, (c) gemini-3.5-transcribe words only, (d) gemini-3.5-transcribe text + WhisperX timing. Pick the winner as default in `configs/transcribe.yaml`. Do not decide from vibes.

### Stage 2. Pack
* Reuse `pack_transcripts.py`: phrase lines `[start-end] text`, break on silence ≥ 0.5 s.
* Add a gap column: silence before each phrase, in ms. The planner needs it.
* Add per phrase language tag (ary, fr, en, mixed).

### Stage 3. Learn mode (`zeta learn`)

Input: pairs `(raw.mp4, published.mp4)` for 3 to 10 past videos. Published only is acceptable with reduced signal.

Steps
1. Transcribe both with Stage 1 (published audio is the facecam audio, so it aligns).
2. Align token sequences with `difflib.SequenceMatcher` on normalized tokens (strip diacritics, unify alef and taa marbuta, lowercase Latin). Output kept spans and cut spans of the raw, with times.
3. Classify each cut span with the LLM into: `filler`, `false_start`, `retake` (later repeat of same content), `tangent`, `dead_air`, `intro_trim`, `outro_trim`, `other`. Store the text and 5 s of context each side.
4. Detect visual inserts in the published video: PySceneDetect content detector + a face presence check (OpenCV or mediapipe): a scene where the face bounding box disappears or shrinks below 40 % of its median size = insert. Sample one frame per insert.
5. Label each insert frame with Gemini vision (`gemini-3.5-flash-lite`): type (article headline, X post, chart, table, code, logo, other), what it shows, likely source, layout (full frame, picture in picture, card), text visible.
6. Map each insert to the spoken words in a window of minus 1 s to plus 1 s around its start. Extract the trigger: first mention of a company, person, number, quote, date, or URL.
7. Cheaper alternative for step 4 to 6 when raw is missing: Gemini agentic video understanding on the published file, asking for a timestamped list of cuts and inserts. Use scene detection as ground truth when both exist.

Output `style_profile.json`:

```json
{
  "cut_ratio": 0.31,
  "median_kept_gap_ms": 380,
  "max_kept_gap_ms": 900,
  "filler_policy": {"remove": ["يعني", "زعما", "euh", "donc", "bon", "genre"], "keep": ["إيوا", "صافي"]},
  "retake_policy": "keep_last_complete",
  "intro_trim_s": 2.1,
  "outro_trim_s": 1.4,
  "inserts": {
    "per_minute": 2.4,
    "median_duration_s": 4.5,
    "lead_in_s": 0.3,
    "layout": "fullframe_pip",
    "pip_corner": "bottom_right",
    "pip_scale": 0.28,
    "triggers": ["first_mention_company", "number", "quote", "headline"]
  },
  "examples": "few_shot_examples.md"
}
```

`few_shot_examples.md`: 5 to 10 before/after transcript excerpts around real cuts and 5 real insert triggers with the frame description. This file is injected into the planner prompt.

### Stage 4. Plan (`zeta plan`)

The planner never outputs timestamps. It outputs an edited transcript and insert markers.

Prompt shape (system):

```
You are the editor of Zeta, a Darija news channel. You receive the verbatim phrase transcript
of a raw take with gap durations, the style profile, and worked examples.
Return JSON:
  {"kept_text": "<full transcript with removed material deleted, nothing paraphrased>",
   "inserts": [{"after_text": "<exact 3 to 6 words before which the visual should appear>",
                "claim": "<what must be visible>", "entity": "<company/person/source>",
                "prefer_source": "<owner's own page or post if known>"}],
   "strategy": "<4 to 8 sentences in plain English>"}
Rules: never change or reorder words you keep. Delete whole phrases where possible.
Remove fillers in filler_policy.remove, false starts, and earlier retakes (keep the last complete take).
Keep the hook and the sentence that names the source. Respect cut_ratio within ±0.10.
Insert markers only where the profile triggers apply.
```

Mechanics
* `derive_cuts.py` aligns `kept_text` to `words.json` with the same diff engine as learn mode → kept ranges on the source timeline.
* Snap every edge to a word boundary (rule 6). Pad 30 to 200 ms, default 50 before and 80 after (rule 7). Prefer cutting inside silences ≥ 400 ms; between 150 and 400 ms require a `timeline_view` check; never below 150 ms.
* Merge kept ranges separated by less than 120 ms.
* Interactive mode: print `strategy`, wait for confirmation (rule 11). Batch mode: `--auto` skips it and the report is the review.

### Stage 5. Research and screenshots (`zeta research`)

1. Claim extraction from `kept_text` plus the planner's `inserts`: entity, number, quote, date.
2. Source resolution, in this priority: URL from `links.txt` → the entity's own domain or official account (company blog, press page, SEC filing, original X post, official GitHub release) → reputable press as last resort. Implement with Gemini Search grounding or a search API; keep the resolver a single function so it can be swapped. Log every candidate and the chosen one.
3. Capture with `shot-scraper` (Playwright): `--selector` on the article header or the post container, width 1600, retina 2x, wait for network idle, cookie banner dismissal via `--javascript`. PDFs: `pdftoppm` on the page containing the number. Numbers without a good page: render a clean chart with matplotlib from the cited figures, labeled with the source.
4. Verify with Gemini vision: "Does this image visibly show <claim>? Answer JSON {visible: bool, evidence: str}". Retry with a different selector once, then drop the slot and flag it in the report. Never ship an unverified screenshot. Never fabricate a source.
5. Build the overlay clip: PNG → MP4 at output resolution with the profile layout (full frame with facecam PiP, or card), optional slow zoom of 3 %, duration from profile, start at `trigger_word_time - lead_in_s` on the OUTPUT timeline (rule 4 PTS shift). Audio is untouched.

### Stage 6. Render
* Vendor `render.py` unchanged: per segment extract with 30 ms audio fades → lossless concat → overlays with `setpts=PTS-STARTPTS+T/TB` → captions last.
* Facecam PiP: implemented as part of the overlay clip (composite the facecam crop into the screenshot frame), so `render.py` still sees one overlay file per slot.

### Stage 7. Captions
* `captions.py`: `words.json` → ASS with output timeline offsets (`output_time = word.start - segment_start + segment_offset`).
* Chunking: 2 to 4 words per line for vertical, 4 to 7 for horizontal, break on punctuation or gaps ≥ 300 ms.
* Font with Arabic coverage: Noto Sans Arabic or IBM Plex Sans Arabic; Latin words inline in the same line; libass handles RTL shaping. Bidi mixed lines must be tested on 3 samples before trusting the style.
* Export SRT alongside for platform upload.

### Stage 8. Self evaluation and report
* Run `timeline_view` on the rendered output at every cut boundary (±1.5 s) and at every insert start and end. Checks: visual jump, waveform spike, caption hidden by overlay, overlay showing wrong frames, `ffprobe` duration equals EDL total.
* Cap at 3 fix and re render passes, then flag.
* `decision_report.html`: table of cuts (time, class, text removed, reason), table of inserts (time, claim, source URL, verification evidence, thumbnail), style profile deltas (actual vs target cut ratio, inserts per minute).
* Append to `project.md`.

## 4. EDL v2 (backward compatible with video-use render.py)

```json
{
  "version": 2,
  "sources": {"raw01": "/abs/path/raw01.mp4"},
  "ranges": [
    {"source": "raw01", "start": 2.42, "end": 6.85, "beat": "HOOK",
     "quote": "...", "reason": "kept, hook", "cut_before": {"class": "intro_trim", "text": "..."}}
  ],
  "overlays": [
    {"file": "edit/screenshots/slot_01/overlay.mp4", "start_in_output": 14.2, "duration": 4.5,
     "meta": {"claim": "SpaceX IPO filing values the company at ...", "url": "https://...",
              "source_type": "owner", "verified": true, "evidence": "headline and figure visible",
              "trigger_word": "SpaceX", "trigger_time_output": 14.5, "layout": "fullframe_pip"}}
  ],
  "subtitles": "edit/captions/final.ass",
  "style_profile": "style_profile.json",
  "total_duration_s": 87.4
}
```

## 5. Repository layout

```
zeta-editor/
  CLAUDE.md                  build rules for Claude Code (this spec, condensed)
  SKILL.md                   adapted from video-use SKILL.md, Zeta specifics added
  helpers/
    ingest.py
    transcribe_gemini.py     backends: gemini_flash_lite | gemini_transcribe | cohere_arabic
    align_whisperx.py        aliases, English spans, interpolate_method ignore
    qa_words.py
    pack_transcripts.py      vendored + gap and lang columns
    learn_style.py           diff + scene detect + vision labels → style_profile.json
    plan_edit.py             planner prompt, returns kept_text + inserts + strategy
    derive_cuts.py           text diff → ranges (shared with learn_style)
    research.py              claims → sources → shots.yml
    screenshot.py            shot-scraper wrapper + pdf + chart fallback
    verify_screenshot.py     vision check
    build_overlay.py         png → overlay.mp4 with PiP and PTS
    captions.py              ASS + SRT on output timeline
    render.py                vendored from video-use
    timeline_view.py         vendored from video-use
    self_eval.py
    report.py
  configs/
    transcribe.yaml          default backend, models, custom vocabulary
    fillers_darija.yaml      seed list, overwritten by the profile
    captions.yaml            fonts, sizes, margins per aspect ratio
    sources.yaml             owner domain map (company → official domains and handles), blocklist
  tests/
    fixtures/                30 s clips with known cuts and one known insert
  cli.py                     zeta learn | plan | research | render | edit (all stages) | review
```

CLI
* `zeta learn --pairs pairs.csv --out style_profile.json`
* `zeta edit raw01.mp4 --profile style_profile.json --links links.txt [--auto] [--aspect 9:16]`
* `zeta review edit/edl.json` opens the report; `--export resolve` writes an XML via auto-editor for manual finishing.

## 6. Environment

| Item | Value |
| --- | --- |
| Python | 3.11, `uv` |
| WhisperX | 3.8.6, PyTorch 2.8.0, CPU works (alignment only); GPU optional |
| ffmpeg, ffprobe | required, libass enabled for ASS burn in |
| Node | 22+ only if HyperFrames or Remotion overlays are used later |
| Playwright | via `shot-scraper install` |
| Keys | `GEMINI_API_KEY` required; `COHERE_API_KEY` optional; search API key optional |
| Hosting | local first; batch mode later on a VPS |

## 7. Build plan with acceptance tests

| Phase | Deliverable | Acceptance |
| --- | --- | --- |
| 0 | Fork video-use, run its pipeline end to end on one English clip | `final.mp4` renders, self evaluation passes |
| 1 | Gemini + WhisperX backend behind `{"words": [...]}` | On 3 Darija clips: alignment coverage ≥ 95 %, zero overlaps, `qa_words.py` green; benchmark table from Stage 1 filled |
| 2 | Plan + derive cuts + captions | No cut inside a word (unit test), cut ratio within ±0.10 of target, captions readable on 3 bidi samples |
| 3 | Learn mode | On a held out pair, predicted cut spans reach IoU ≥ 0.7 against real cuts; insert detection recall ≥ 0.8 |
| 4 | Research + screenshots + verification | Every shipped insert has a URL and `verified: true`; a fabricated or unverified insert fails CI |
| 5 | Self evaluation, report, `--auto` batch mode | One raw video to `final.mp4` unattended in under 15 minutes on CPU for a 10 minute source |

## 8. Hard rules (inherited, keep verbatim in CLAUDE.md)

1. Captions applied last in the filter chain.
2. Per segment extract then lossless concat, never a single pass filtergraph.
3. 30 ms audio fades at every boundary.
4. Overlays use `setpts=PTS-STARTPTS+T/TB`.
5. Caption times use output timeline offsets.
6. Never cut inside a word.
7. Pad every cut edge, 30 to 200 ms.
8. Word level verbatim ASR only; no SMART or normalized modes.
9. Cache transcripts per source hash.
10. Strategy confirmation before execution unless `--auto`.
11. All outputs in `<videos_dir>/edit/`.
12. Zeta additions: no unverified screenshot ships; owner's original source is preferred over secondary coverage; the LLM never emits timestamps, cuts are derived from text diffs.

## 9. Open decisions for Ali before phase 2

* Default insert layout: full frame screenshot with facecam PiP, or card over the facecam.
* Target aspect ratios (9:16 for Reels and TikTok, 16:9 for YouTube, both from one EDL).
* Whether `--auto` is allowed to publish drafts without the confirmation step.
* Cohere second opinion: local run (needs a GPU or patience) or API.

## 10. Kickoff prompt for Claude Code

```
Read ZETA_AUTO_EDITOR_SPEC.md fully. Then:
1. Clone https://github.com/browser-use/video-use into ./vendor and read its SKILL.md, install.md and helpers/.
2. Read https://github.com/Moh4696/freecut for the pluggable transcription backend pattern.
3. Create the repository layout in section 5. Copy render.py and timeline_view.py from vendor.
4. Implement phase 0 and phase 1 only. Stop and show me the phase 1 benchmark table before phase 2.
Do not change render.py. Do not let any model emit timestamps. Ask me only for API keys.
```
