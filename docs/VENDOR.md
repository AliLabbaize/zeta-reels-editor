# Vendored code

`helpers/render.py` and `helpers/timeline_view.py` are copied **verbatim** from
[browser-use/video-use](https://github.com/browser-use/video-use) (MIT).

| Field | Value |
| --- | --- |
| Upstream commit | `9575612f066aa517354790a645fd90f9f95a743b` |
| Upstream date | 2026-08-30 |
| Copied on | 2026-09-18 |
| `render.py` sha1 | `5f908e643b4c96d186e6eed8d5e384ca200be86f` |
| `timeline_view.py` sha1 | `dea86d6e20722d0656e54a8d986951f3b2917dc7` |

Upstream licence: `docs/LICENSE.video-use`.

## Rules

* **Never edit these two files.** Hard rules 1-5 of the pipeline live inside them and
  they are the reason the render is correct. Behaviour changes go in a wrapper
  (`helpers/edl.py`, `helpers/build_overlay.py`, `helpers/captions.py`), not here.
* They are invoked as **subprocesses**, never imported, so their module-level
  `from grade import ...` fallback stays on the upstream path.
* `tests/test_vendor_unchanged.py` fails if either sha1 drifts. If you deliberately
  re-sync with upstream, update the table and the test in the same commit.

## What we do NOT take from upstream

ElevenLabs Scribe transcription (no Darija), colour-grade presets, the launch-video
palette, `transcribe.py`, `transcribe_batch.py`, `grade.py`.
