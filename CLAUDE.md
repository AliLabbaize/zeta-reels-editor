# Zeta auto editor - build rules

End-to-end editor for Zeta facecam news videos (Darija with French and English
code switching). Raw footage plus optional source links in; a cut, captioned
video with **verified** screenshots of the sources placed where they are
mentioned out. `zeta learn` learns the editing style from past videos, `zeta
edit` applies it to new footage.

Built on a fork of [browser-use/video-use](https://github.com/browser-use/video-use)
(MIT). Full design: `docs/ZETA_AUTO_EDITOR_SPEC.md`.

## Hard rules (non-negotiable; violating any of these is a silent failure)

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
12. Zeta additions: no unverified screenshot ships; owner's original source is
    preferred over secondary coverage; the LLM never emits timestamps, cuts are
    derived from text diffs.

Rules 1-5 live inside `helpers/render.py`, which is vendored verbatim.
**Never edit `helpers/render.py` or `helpers/timeline_view.py`** - see
`docs/VENDOR.md`. `tests/test_vendor_unchanged.py` enforces it.

## Architecture in one paragraph

Gemini supplies the *text*, wav2vec2 supplies the *time*. Every downstream
consumer reads one artifact, `transcripts/<name>.words.json`. The editor model
returns the transcript with material deleted; `helpers/diff_align.py` diffs that
against the aligned words and the gaps ARE the cuts. Learn mode runs the same
diff between a raw take and its published version, so learning and applying
share one code path and one class of bug. Screenshots are overlay clips with
provenance metadata; an overlay with `verified != true` fails EDL validation and
never reaches the render.

## Module map

| Layer | Modules |
| --- | --- |
| Contracts | `paths`, `config`, `textnorm`, `words`, `diff_align`, `edl`, `gemini_client` |
| Stage 0-2 | `ingest`, `transcribe_gemini`, `align_whisperx`, `qa_words`, `pack_transcripts`, `benchmark` |
| Stage 3-4 | `learn_style`, `plan_edit`, `derive_cuts` |
| Stage 5 | `research`, `screenshot`, `verify_screenshot`, `build_overlay` |
| Stage 6-8 | `render` (vendored), `captions`, `timeline_view` (vendored), `self_eval`, `report` |

## Code rules

* Python 3.11, `uv`. Every module imports with **stdlib + PyYAML only**; heavy
  and optional deps (`google-genai`, `whisperx`, `torch`, `cv2`, `scenedetect`,
  `matplotlib`, PIL, numpy) are imported lazily inside the function that needs
  them, with an error naming the extra to install. `zeta plan` must not pay for
  torch, and the test suite must run on a bare interpreter.
* Every helper is importable (`from helpers.x import ...`) and runnable
  (`python helpers/x.py --help`).
* One LLM door: `helpers/gemini_client.py`. No module imports `google.genai`.
* One diff engine: `helpers/diff_align.py`. Never a second copy of the logic.
* One timeline mapping: `EDL.to_output_time`. Never re-derive it.
* Model IDs live in `configs/*.yaml`, never in code, so a renamed model is a
  config edit.
* Tests never hit the network: `ZETA_LLM_MOCK=1` plus
  `gemini_client.register_mock`.
* Comments explain **why**, where the reason is not obvious from the code.

Deliberate departures from the spec are recorded in `docs/DEVIATIONS.md`.
Add to that file rather than silently absorbing a decision.

## Things that look like improvements and are not

* Letting the planner return timestamps "to save a step". It hallucinates them.
* Cropping a screenshot to fill the frame. A cropped headline can change what
  the source appears to say. `fit_card` never crops.
* Normalising fillers in ASR. Fillers are the editorial signal learn mode reads.
* Re-transcribing an unchanged source. Immutable outputs of immutable inputs.
* Skipping a failed verification "because the URL is obviously right". Rule 12.
