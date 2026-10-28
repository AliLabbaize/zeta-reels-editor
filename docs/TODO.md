# TODO

## Refactor

- The per-video scripts in each video folder (`retranscribe.py`,
  `fix_words.py`, `make_cut.py`, `capture_tweets.py`, `capture_passage.py`,
  `build_cards.py`, `make_overlays.py`) are copied and edited for every reel.
  Move the parts that repeat into `helpers/` and the CLI.
- `zeta render` does not write captions; `stage_captions` has to be run by
  hand first. Make render do it, or fail loudly.
- The chunked re-transcription (flash-lite drops or mistimes words on long
  takes) should be a backend option of `zeta transcribe`, not a copied script.
- The caption moderation rewrite changed "alive" into "unalive". Give it a
  test so it cannot change meaning.
- Keep the vendored files untouched (`helpers/render.py`,
  `helpers/timeline_view.py`).
