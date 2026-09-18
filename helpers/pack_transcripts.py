"""Stage 2: the reading view the planner is given instead of raw JSON.

An LLM asked to edit 4000 word records returns 4000 word records, badly. Asked
to delete lines from a phrase list, it deletes lines. So `words.json` is packed
into one line per phrase, split wherever the speaker paused for at least half a
second:

    [12.41-15.02]  gap=380ms  [ary]  واش هاد الشركة غادي تدخل للبورصة

Two columns beyond the upstream video-use format, both because the planner
needs them and cannot recover them from the text:

  * `gap` - the silence BEFORE this phrase, in milliseconds. A cut between two
    phrases separated by 40 ms is audible; the same cut across a 600 ms pause
    is invisible. The planner is told to prefer deleting whole phrases that sit
    behind a long gap, and the style profile's `median_kept_gap_ms` is measured
    in the same unit.
  * `[lang]` - ary | fr | en | mixed, from `textnorm.phrase_lang`. The filler
    policy is per language, and a French aside is a different editorial object
    from a Darija one.

This is OUR packer: it reads `zeta.words.v1`, not ElevenLabs Scribe output.

Run standalone:
    python helpers/pack_transcripts.py --videos-dir ./videos
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

try:
    from . import textnorm
    from .paths import EditPaths
    from .words import Word, WordsDoc
except ImportError:  # running as `python helpers/pack_transcripts.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from helpers import textnorm
    from helpers.paths import EditPaths
    from helpers.words import Word, WordsDoc

# Stage 2 of the spec. Below this, a "pause" is just articulation.
SILENCE_S = 0.5

HEADER = """# Packed takes

One line per phrase: `[start-end]  gap=<silence before, ms>  [lang]  text`
Times are SOURCE seconds. Split on silence >= {silence:.0f} ms.
"""


@dataclass
class Phrase:
    start: float
    end: float
    gap_before_ms: int
    lang: str
    text: str
    word_start: int
    word_end: int

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return {"start": round(self.start, 3), "end": round(self.end, 3),
                "gap_before_ms": self.gap_before_ms, "lang": self.lang,
                "text": self.text, "word_start": self.word_start,
                "word_end": self.word_end}


def _finish(doc: WordsDoc, i: int, j: int, gap_ms: int) -> Phrase | None:
    chunk = doc.words[i:j]
    if not chunk:
        return None
    timed = [w for w in chunk if w.timed]
    if not timed:
        return None
    tokens = [w.display or w.word for w in chunk]
    return Phrase(
        start=timed[0].start,
        end=timed[-1].end,
        gap_before_ms=gap_ms,
        lang=textnorm.phrase_lang([w.word for w in chunk]),
        text=" ".join(tokens),
        word_start=i,
        word_end=j,
    )


def phrases(doc: WordsDoc, silence_s: float = SILENCE_S,
            max_words: int = 0) -> list[Phrase]:
    """Group words into phrases, breaking on silence >= `silence_s`.

    Untimed words (the aligner could not place them; QA has already decided the
    transcript is good enough to use) stay inside the phrase they were spoken
    in. They contribute text but never a boundary, because a boundary needs a
    time to be worth anything.
    """
    out: list[Phrase] = []
    start_i = 0
    prev_end: float | None = None
    # Silence before the first phrase is measured from t=0: that lead-in is what
    # `intro_trim_s` in the style profile is learned from.
    gap_ms = 0

    for i, w in enumerate(doc.words):
        if not w.timed:
            continue
        gap = (w.start - prev_end) if prev_end is not None else w.start
        breaks = (prev_end is not None and gap >= silence_s) or \
                 (max_words > 0 and i - start_i >= max_words)
        if breaks:
            phrase = _finish(doc, start_i, i, gap_ms)
            if phrase:
                out.append(phrase)
            start_i = i
            gap_ms = int(round(max(0.0, gap) * 1000))
        elif prev_end is None:
            gap_ms = int(round(max(0.0, gap) * 1000))
        prev_end = w.end

    last = _finish(doc, start_i, len(doc.words), gap_ms)
    if last:
        out.append(last)
    return out


def format_phrase(p: Phrase) -> str:
    return (f"[{p.start:.2f}-{p.end:.2f}]  gap={p.gap_before_ms}ms  "
            f"[{p.lang}]  {p.text}")


def pack_doc(doc: WordsDoc, *, name: str | None = None,
             silence_s: float = SILENCE_S, max_words: int = 0) -> str:
    """One source's block: a heading line plus its phrase lines."""
    ph = phrases(doc, silence_s, max_words)
    title = name or (doc.source or {}).get("name") or "take"
    spoken = sum(p.duration for p in ph)
    head = (f"## {title}  ({len(ph)} phrases, {spoken:.1f}s spoken, "
            f"{len(doc.words)} words, coverage {doc.coverage() * 100:.1f}%)")
    return "\n".join([head, ""] + [format_phrase(p) for p in ph])


def find_transcripts(edit_paths: EditPaths) -> list[Path]:
    return sorted(edit_paths.transcripts.glob("*.words.json"))


def pack(
    edit_paths: EditPaths,
    *,
    silence_s: float = SILENCE_S,
    max_words: int = 0,
    write: bool = True,
    paths: Sequence[str | Path] | None = None,
) -> str:
    """Pack every transcript under `edit/transcripts/` into `edit/takes_packed.md`."""
    files = [Path(p) for p in paths] if paths else find_transcripts(edit_paths)
    if not files:
        raise FileNotFoundError(
            f"no *.words.json under {edit_paths.transcripts}; run transcribe + align first")

    blocks = [HEADER.format(silence=silence_s * 1000)]
    for f in files:
        doc = WordsDoc.load(f)
        blocks.append(pack_doc(doc, name=f.name.replace(".words.json", ""),
                               silence_s=silence_s, max_words=max_words))
    text = "\n\n".join(blocks).rstrip() + "\n"

    if write:
        edit_paths.edit.mkdir(parents=True, exist_ok=True)
        edit_paths.packed.write_text(text, encoding="utf-8")
    return text


# -- cli ----------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Pack words.json transcripts into the phrase view (Stage 2).")
    ap.add_argument("--videos-dir", required=True)
    ap.add_argument("--silence", type=float, default=SILENCE_S,
                    help="phrase break threshold in seconds (default 0.5)")
    ap.add_argument("--max-words", type=int, default=0,
                    help="also break after N words (0 = only on silence)")
    ap.add_argument("--stdout", action="store_true", help="print instead of writing")
    args = ap.parse_args(argv)

    edit_paths = EditPaths.for_videos_dir(args.videos_dir)
    try:
        text = pack(edit_paths, silence_s=args.silence, max_words=args.max_words,
                    write=not args.stdout)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.stdout:
        print(text)
    else:
        print(f"wrote {edit_paths.packed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
