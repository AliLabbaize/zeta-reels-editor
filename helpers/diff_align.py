"""The one diff engine.

Both halves of this project are the same operation:

  * `zeta plan`  - the editor returns the transcript with material deleted.
    Diff it against the raw word list; the matched blocks ARE the cuts.
  * `zeta learn` - a published video's transcript is the raw transcript with
    material deleted. Same diff, same blocks, and the gaps are what Ali cut.

Because learning and applying run through one code path, a bug shows up in both
and a fix lands in both. Nothing here ever invents a timestamp: it only says
which of the ORIGINAL words survived, and their timings come from the aligner.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Sequence

from . import textnorm
from .words import Word, WordsDoc


@dataclass(frozen=True)
class Span:
    """Half-open word-index span `[start, end)` in the reference word list."""

    start: int
    end: int

    def __len__(self) -> int:
        return max(0, self.end - self.start)

    def __bool__(self) -> bool:
        return self.end > self.start


@dataclass
class DiffResult:
    """Which reference words survived, and which did not."""

    kept: list[Span]
    cut: list[Span]
    # Fraction of reference words removed. Compared against style_profile.cut_ratio.
    cut_ratio: float
    # Reference words that the edited text matched, in order (for sanity checks).
    matched: int
    # Tokens present in the edited text but absent from the reference. Non-empty
    # means the model paraphrased instead of deleting -- a plan-level error.
    invented: list[str]


def _norm_list(tokens: Sequence[str]) -> list[str]:
    return [textnorm.normalize_token(t) for t in tokens]


def _word_tokens(doc: WordsDoc | Sequence[Word]) -> list[str]:
    words = doc.words if isinstance(doc, WordsDoc) else list(doc)
    return [w.norm for w in words]


def match_spans(ref: Sequence[str], edited: Sequence[str]) -> list[tuple[int, int, int]]:
    """Matching blocks as `(ref_start, edited_start, size)`, longest-first order.

    `autojunk` is disabled: it drops tokens appearing in >1% of a sequence over
    200 items, which in a transcript means the most common Darija words. With it
    on, long takes silently mis-align.
    """
    sm = SequenceMatcher(a=list(ref), b=list(edited), autojunk=False)
    return [(m.a, m.b, m.size) for m in sm.get_matching_blocks() if m.size > 0]


def _merge_adjacent(spans: list[Span]) -> list[Span]:
    out: list[Span] = []
    for s in sorted(spans, key=lambda x: x.start):
        if out and s.start <= out[-1].end:
            out[-1] = Span(out[-1].start, max(out[-1].end, s.end))
        else:
            out.append(s)
    return out


def _invert(spans: list[Span], total: int) -> list[Span]:
    out: list[Span] = []
    cursor = 0
    for s in spans:
        if s.start > cursor:
            out.append(Span(cursor, s.start))
        cursor = max(cursor, s.end)
    if cursor < total:
        out.append(Span(cursor, total))
    return out


def diff_tokens(ref_tokens: Sequence[str], edited_tokens: Sequence[str]) -> DiffResult:
    """Core primitive: align two normalised token sequences."""
    blocks = match_spans(ref_tokens, edited_tokens)
    kept = _merge_adjacent([Span(a, a + size) for a, _b, size in blocks])
    matched = sum(size for _a, _b, size in blocks)
    total = len(ref_tokens)
    cut = _invert(kept, total)
    kept_count = sum(len(s) for s in kept)
    cut_ratio = 1.0 - (kept_count / total) if total else 0.0

    covered_b: list[tuple[int, int]] = sorted((b, b + size) for _a, b, size in blocks)
    invented: list[str] = []
    cursor = 0
    for b0, b1 in covered_b:
        if b0 > cursor:
            invented.extend(edited_tokens[cursor:b0])
        cursor = max(cursor, b1)
    invented.extend(edited_tokens[cursor:])

    return DiffResult(kept=kept, cut=cut, cut_ratio=cut_ratio, matched=matched,
                      invented=[t for t in invented if t])


def diff_text_against_words(doc: WordsDoc, edited_text: str) -> DiffResult:
    """`zeta plan`: which words of `doc` survive in the editor's `kept_text`."""
    return diff_tokens(_word_tokens(doc), _norm_list(textnorm.tokenize(edited_text)))


def diff_words(raw: WordsDoc, published: WordsDoc) -> DiffResult:
    """`zeta learn`: which words of the raw take survive in the published one."""
    return diff_tokens(_word_tokens(raw), _word_tokens(published))


def spans_to_time(doc: WordsDoc, spans: list[Span]) -> list[tuple[float, float]]:
    """Wall-clock ranges for word spans, skipping spans with no timed word."""
    out: list[tuple[float, float]] = []
    for s in spans:
        a, b = doc.span_time(s.start, s.end)
        if b > a:
            out.append((a, b))
    return out


def span_text(doc: WordsDoc, span: Span, sep: str = " ") -> str:
    return sep.join((w.display or w.word) for w in doc.words[span.start:span.end])


def context_around(doc: WordsDoc, span: Span, seconds: float = 5.0) -> tuple[str, str]:
    """`(before, after)` text within `seconds` either side of a span."""
    a, b = doc.span_time(span.start, span.end)
    if b <= a:
        return ("", "")
    before = " ".join(w.display or w.word for w in doc.in_range(max(0.0, a - seconds), a))
    after = " ".join(w.display or w.word for w in doc.in_range(b, b + seconds))
    return (before, after)
