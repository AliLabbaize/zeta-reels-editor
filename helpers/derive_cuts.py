"""Text diff to cut ranges: the only place a cut is ever decided.

The planner hands back the transcript with material deleted. Nothing else. This
module turns that text into timed ranges by diffing it against the aligned word
list, so every number on the timeline comes from the aligner and none from a
model (hard rule 12).

The pipeline here is fixed, and the order matters:

    diff -> classify -> move unsafe boundaries into silence -> invert to kept
    spans -> pad the edges -> merge ranges that are not worth cutting apart

Boundary safety before padding, padding before merging: a boundary that moves
changes which silence the pad lives in, and a pad that eats the whole removal
means there was never a cut worth making.
"""

from __future__ import annotations

# Runnable both as `from helpers.derive_cuts import derive` and as
# `python helpers/derive_cuts.py`. The latter starts with no package context, so
# the repo root goes on the path and the package name is set before the relative
# imports below execute.
if __package__ in (None, ""):  # pragma: no cover - script entry only
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    __package__ = "helpers"

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import config, diff_align, edl, textnorm
from .diff_align import Span
from .words import WordsDoc

# Hard rule 7: every edge is padded, and the pad is bounded on both sides.
PAD_BEFORE_MS = 50.0
PAD_AFTER_MS = 80.0
PAD_MIN_MS = 30.0
PAD_MAX_MS = 200.0

# Silence classes at a cut point (spec, stage 4 mechanics).
SAFE_GAP_MS = 400.0     # cut here without asking
CHECK_GAP_MS = 150.0    # cut here, but the caller should eyeball it
MERGE_GAP_MS = 120.0    # two kept ranges closer than this are one range

# How far a boundary may travel to find silence: a cut point that has to move a
# second and a half is not the same edit any more, so we stop and flag instead.
MAX_SHIFT_WORDS = 4
MAX_SHIFT_S = 1.5

CUT_RATIO_TOLERANCE = 0.10

CLASSES = (
    "filler", "false_start", "retake", "tangent",
    "dead_air", "intro_trim", "outro_trim", "other",
)


class ParaphraseError(ValueError):
    """The edited text contains words the source never said.

    Deleting is reversible and verifiable; rewriting is neither. A paraphrase
    cannot be turned into cuts at all -- the invented words have no timings --
    so this is fatal rather than something to silently drop.
    """

    def __init__(self, invented: list[str]):
        self.invented = invented
        shown = ", ".join(repr(t) for t in invented[:12])
        more = f" (+{len(invented) - 12} more)" if len(invented) > 12 else ""
        super().__init__(
            f"the edited text contains {len(invented)} word(s) absent from the "
            f"source: {shown}{more}. The editor paraphrased instead of deleting.")


@dataclass
class CutSpan:
    """One removal, in word indices and in source seconds."""

    span: Span                  # half-open word-index span removed from the source
    start: float                # removed region on the source timeline, after padding
    end: float
    text: str                   # what was removed, display form
    klass: str
    reason: str
    gap_before_ms: float        # silence at the head boundary of the removal
    gap_after_ms: float         # silence at the tail boundary
    needs_visual_check: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        d = {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "words": [self.span.start, self.span.end],
            "class": self.klass,
            "text": self.text,
            "reason": self.reason,
            "gap_before_ms": round(self.gap_before_ms, 1),
            "gap_after_ms": round(self.gap_after_ms, 1),
        }
        if self.needs_visual_check:
            d["needs_visual_check"] = True
        if self.notes:
            d["notes"] = self.notes
        return d


@dataclass
class CutPlan:
    """Everything the renderer and the report need about one derived edit."""

    ranges: list[edl.Range]
    cuts: list[CutSpan]
    source_name: str
    kept_spans: list[Span]
    cut_ratio: float                    # achieved, by time, after padding and merging
    word_cut_ratio: float               # achieved, by word count (the diff's own number)
    target_cut_ratio: float | None
    tolerance: float = CUT_RATIO_TOLERANCE
    needs_visual_check: list[dict] = field(default_factory=list)
    dropped: list[CutSpan] = field(default_factory=list)  # cuts merged away
    source_duration: float = 0.0

    @property
    def kept_duration(self) -> float:
        return sum(r.duration for r in self.ranges)

    @property
    def cut_ratio_delta(self) -> float | None:
        """Achieved minus target. Positive means the edit cut more than the style."""
        if self.target_cut_ratio is None:
            return None
        return self.cut_ratio - self.target_cut_ratio

    @property
    def within_band(self) -> bool:
        d = self.cut_ratio_delta
        return True if d is None else abs(d) <= self.tolerance + 1e-9

    def word_index_time(self, doc: WordsDoc, index: int) -> tuple[float, float]:
        """Source `(start, end)` of one word. The only bridge from index to time."""
        return doc.span_time(index, index + 1)

    def to_dict(self) -> dict:
        return {
            "source": self.source_name,
            "ranges": [r.to_dict() for r in self.ranges],
            "cuts": [c.to_dict() for c in self.cuts],
            "dropped": [c.to_dict() for c in self.dropped],
            "cut_ratio": round(self.cut_ratio, 4),
            "word_cut_ratio": round(self.word_cut_ratio, 4),
            "target_cut_ratio": self.target_cut_ratio,
            "cut_ratio_delta": (None if self.cut_ratio_delta is None
                                else round(self.cut_ratio_delta, 4)),
            "within_band": self.within_band,
            "kept_duration_s": round(self.kept_duration, 3),
            "source_duration_s": round(self.source_duration, 3),
            "needs_visual_check": self.needs_visual_check,
        }

    def summary(self) -> str:
        delta = self.cut_ratio_delta
        band = "n/a" if delta is None else f"{delta:+.3f} vs target {self.target_cut_ratio}"
        return (f"{len(self.ranges)} ranges, {len(self.cuts)} cuts, "
                f"{self.kept_duration:.2f}s kept of {self.source_duration:.2f}s "
                f"(cut_ratio {self.cut_ratio:.3f}, {band}), "
                f"{len(self.needs_visual_check)} boundaries to eyeball")


# -------- timing helpers -----------------------------------------------------


def _source_bounds(doc: WordsDoc) -> tuple[float, float]:
    """`(0.0, duration)`. The take starts at zero even if speech does not."""
    timed = doc.timed_words()
    end = float(doc.source.get("duration_s") or 0.0)
    if timed:
        end = max(end, timed[-1].end)
    return (0.0, end)


def _prev_word_end(doc: WordsDoc, index: int, floor: float) -> float:
    """End of the nearest timed word before `index`, else the source start."""
    for w in reversed(doc.words[:index]):
        if w.timed:
            return w.end
    return floor


def _next_word_start(doc: WordsDoc, index: int, ceil: float) -> float:
    """Start of the nearest timed word at or after `index`, else the source end."""
    for w in doc.words[index:]:
        if w.timed:
            return w.start
    return ceil


def _gap_ms_before(doc: WordsDoc, index: int, floor: float) -> float:
    """Silence immediately before word `index`, in ms."""
    w = doc.words[index] if 0 <= index < len(doc.words) else None
    if w is None or not w.timed:
        return 0.0
    return max(0.0, (w.start - _prev_word_end(doc, index, floor)) * 1000.0)


def _gap_class(gap_ms: float) -> str:
    if gap_ms >= SAFE_GAP_MS:
        return "safe"
    if gap_ms >= CHECK_GAP_MS:
        return "check"
    return "unsafe"


# -------- boundary safety ----------------------------------------------------


def _boundary_time(doc: WordsDoc, index: int, floor: float, ceil: float) -> float:
    """Wall clock of the word boundary in front of word `index`."""
    if index <= 0:
        return floor
    if index >= len(doc.words):
        return ceil
    return _next_word_start(doc, index, ceil)


def _safe_boundary(
    doc: WordsDoc,
    index: int,
    *,
    lo: int,
    hi: int,
    floor: float,
    ceil: float,
) -> tuple[int, float, str, list[str]]:
    """Move a cut boundary to the nearest word boundary that sits in silence.

    `index` is the word before which the boundary lies; `lo`/`hi` bound how far
    it may move without colliding with the neighbouring edit. A boundary in less
    than 150 ms of silence is never usable -- the cut lands in the middle of
    running speech and the listener hears the seam -- so we walk outwards for a
    usable one. Ties go to the candidate that KEEPS more material: leaving a
    filler in is a smaller mistake than deleting a word the editor wanted.
    """
    notes: list[str] = []
    gap = _gap_ms_before(doc, index, floor)
    if index <= 0 or index >= len(doc.words):
        # The head of the take and its tail are silence by construction.
        return index, gap, "safe", notes
    if _gap_class(gap) != "unsafe":
        return index, gap, _gap_class(gap), notes

    origin = _boundary_time(doc, index, floor, ceil)
    best: tuple[float, int, int, float, str] | None = None
    for cand in range(max(lo, index - MAX_SHIFT_WORDS), min(hi, index + MAX_SHIFT_WORDS) + 1):
        if cand == index or cand <= 0 or cand >= len(doc.words):
            continue
        cls = _gap_class(_gap_ms_before(doc, cand, floor))
        if cls == "unsafe":
            continue
        dist = abs(_boundary_time(doc, cand, floor, ceil) - origin)
        if dist > MAX_SHIFT_S:
            continue
        # Sort key: distance, then prefer the safest silence, then prefer the
        # candidate on the "keep more" side of the original boundary.
        rank = (round(dist, 4), 0 if cls == "safe" else 1, 0 if cand > index else 1)
        if best is None or rank < best[:3] + (0.0, ""):
            best = (rank[0], rank[1], rank[2], _gap_ms_before(doc, cand, floor), cls)
            best_index = cand
    if best is None:
        notes.append(f"no silence >= {CHECK_GAP_MS:.0f} ms within "
                     f"{MAX_SHIFT_WORDS} words; boundary left in speech ({gap:.0f} ms)")
        return index, gap, "unsafe", notes
    notes.append(f"boundary moved {best_index - index:+d} word(s) into "
                 f"{best[3]:.0f} ms of silence")
    return best_index, best[3], best[4], notes


# -------- classification -----------------------------------------------------


def _filler_tokens(profile: dict) -> set[str]:
    """Normalised filler vocabulary: the profile wins, the seed config is cold start."""
    raw: Any = (profile.get("filler_policy") or {}).get("remove")
    if raw is None:
        try:
            raw = config.load("fillers_darija").get("remove", {})
        except (FileNotFoundError, OSError):
            raw = []
    items: list[str] = []
    if isinstance(raw, dict):
        for v in raw.values():
            items.extend(v or [])
    else:
        items.extend(raw or [])
    out: set[str] = set()
    for phrase in items:
        out.update(textnorm.normalize_tokens(str(phrase)))
    return out


def _speech_seconds(doc: WordsDoc, span: Span) -> float:
    return sum(w.duration for w in doc.words[span.start:span.end] if w.timed)


def classify_cut(doc: WordsDoc, span: Span, fillers: set[str]) -> tuple[str, str]:
    """Best-effort class and human reason for one removal.

    Heuristic on purpose: `zeta learn` asks the LLM to classify PAST cuts, where
    it has the whole published video as evidence. At plan time the class is only
    report copy, and a wrong label must never change what gets rendered.
    """
    n = len(doc.words)
    removed = [w.norm for w in doc.words[span.start:span.end] if w.norm]
    following = [w.norm for w in doc.words[span.end:span.end + 40] if w.norm]
    seconds = doc.span_time(span.start, span.end)
    span_s = max(0.0, seconds[1] - seconds[0])

    if span.start == 0:
        return "intro_trim", f"trimmed {span_s:.2f}s of lead-in before the hook"
    if span.end >= n:
        return "outro_trim", f"trimmed {span_s:.2f}s of tail after the last kept word"
    if removed and all(t in fillers for t in removed):
        return "filler", f"filler only: {' '.join(removed)}"
    for k in (3, 2, 1):
        if len(removed) >= k and len(following) >= k:
            if removed[-k:] == following[:k] or removed[:k] == following[:k]:
                return "false_start", f"restarted on {' '.join(following[:k])}"
    if len(removed) >= 4 and following:
        repeated = sum(1 for t in set(removed) if t in set(following))
        if repeated / len(set(removed)) >= 0.6:
            return "retake", "same content is said again later; earlier take dropped"
    if span_s >= 0.4 and _speech_seconds(doc, span) / span_s < 0.4:
        return "dead_air", f"{span_s:.2f}s span that is mostly silence"
    if len(removed) >= 12:
        return "tangent", f"{len(removed)} words off the story line"
    return "other", f"{len(removed)} words removed by the editor"


# -------- the engine ---------------------------------------------------------


def _pad_ms(profile: dict, key: str, default: float) -> float:
    """Pad from the profile, clamped to the 30-200 ms window of hard rule 7."""
    raw = config.get(profile, f"cuts.{key}", default)
    try:
        val = float(raw)
    except (TypeError, ValueError):
        val = default
    return min(PAD_MAX_MS, max(PAD_MIN_MS, val))


def _adjusted_cuts(doc: WordsDoc, cuts: list[Span], floor: float, ceil: float
                   ) -> list[tuple[Span, float, float, str, str, list[str]]]:
    """Snap every cut boundary into silence, dropping cuts that shrink to nothing."""
    out: list[tuple[Span, float, float, str, str, list[str]]] = []
    n = len(doc.words)
    for idx, span in enumerate(cuts):
        # A boundary may not cross into the neighbouring cut, nor past the words
        # this cut already owns on the far side.
        prev_end = out[-1][0].end if out else 0
        next_start = cuts[idx + 1].start if idx + 1 < len(cuts) else n

        start, gap_before, cls_before, notes = _safe_boundary(
            doc, span.start, lo=prev_end, hi=span.end, floor=floor, ceil=ceil)
        end, gap_after, cls_after, notes_after = _safe_boundary(
            doc, span.end, lo=max(start + 1, prev_end), hi=next_start, floor=floor, ceil=ceil)
        notes = notes + notes_after
        if end <= start:
            # Both boundaries collapsed onto each other: the removal was too
            # small to survive the silence rule, so nothing is cut here.
            continue
        out.append((Span(start, end), gap_before, gap_after, cls_before, cls_after, notes))
    return out


def derive(doc: WordsDoc, kept_text: str, profile: dict, source_name: str) -> CutPlan:
    """Diff `kept_text` against `doc` and return the cut plan for `source_name`.

    Raises `ParaphraseError` when the edited text is not a subsequence of what
    was actually said.
    """
    result = diff_align.diff_text_against_words(doc, kept_text)
    if result.invented:
        raise ParaphraseError(result.invented)
    if not result.kept:
        raise ValueError("the edited text matched no words of the source at all")

    floor, ceil = _source_bounds(doc)
    fillers = _filler_tokens(profile)
    n = len(doc.words)

    adjusted = _adjusted_cuts(doc, list(result.cut), floor, ceil)
    kept_spans = _invert_spans([a[0] for a in adjusted], n)
    if not kept_spans:
        raise ValueError("every word was cut: nothing left to render")

    # Word boundaries in, word boundaries out (hard rule 6): the only times that
    # ever reach the EDL are word starts and word ends, plus the pad below.
    raw_ranges = [doc.span_time(s.start, s.end) for s in kept_spans]
    pad_before = _pad_ms(profile, "pad_before_ms", PAD_BEFORE_MS) / 1000.0
    pad_after = _pad_ms(profile, "pad_after_ms", PAD_AFTER_MS) / 1000.0

    padded: list[tuple[float, float]] = []
    for span, (a, b) in zip(kept_spans, raw_ranges):
        # The pad may only use the silence adjacent to the boundary. Reaching
        # further would put the cut point inside a word -- a removed word is
        # still a word (hard rule 6) -- and would also risk the pad of the
        # neighbouring range overlapping this one.
        head_room = max(0.0, a - _prev_word_end(doc, span.start, floor))
        tail_room = max(0.0, _next_word_start(doc, span.end, ceil) - b)
        padded.append((max(floor, a - min(pad_before, head_room)),
                       min(ceil, b + min(pad_after, tail_room))))

    ranges, kept_spans, dropped_idx = _merge_close(kept_spans, padded)

    cut_records: list[CutSpan] = []
    dropped: list[CutSpan] = []
    for i, (span, gap_before, gap_after, cls_before, cls_after, notes) in enumerate(adjusted):
        klass, reason = classify_cut(doc, span, fillers)
        # The removal as rendered runs from the padded end of the range before
        # it to the padded start of the range after it.
        left = ranges[i][1] if i < len(ranges) else None
        record = CutSpan(
            span=span,
            start=0.0, end=0.0,
            text=diff_align.span_text(doc, span),
            klass=klass, reason=reason,
            gap_before_ms=gap_before, gap_after_ms=gap_after,
            needs_visual_check=("check" in (cls_before, cls_after)
                                or "unsafe" in (cls_before, cls_after)),
            notes=list(notes),
        )
        if cls_before == "unsafe" or cls_after == "unsafe":
            record.notes.append("no safe silence at one edge")
        (dropped if i in dropped_idx else cut_records).append(record)

    _place_cut_times(cut_records, ranges, floor, ceil)

    flags: list[dict] = []
    for c in cut_records:
        if not c.needs_visual_check:
            continue
        flags.append({
            "source": source_name,
            "start": round(c.start, 3),
            "end": round(c.end, 3),
            "gap_before_ms": round(c.gap_before_ms, 1),
            "gap_after_ms": round(c.gap_after_ms, 1),
            "class": c.klass,
            "why": "; ".join(c.notes) or
                   f"cut point sits in {min(c.gap_before_ms, c.gap_after_ms):.0f} ms of "
                   f"silence (< {SAFE_GAP_MS:.0f} ms): run timeline_view here",
        })

    kept_duration = sum(b - a for a, b in ranges)
    total = max(1e-9, ceil - floor)
    edl_ranges = _build_ranges(doc, source_name, ranges, kept_spans, cut_records)

    return CutPlan(
        ranges=edl_ranges,
        cuts=cut_records,
        source_name=source_name,
        kept_spans=kept_spans,
        cut_ratio=max(0.0, 1.0 - kept_duration / total),
        word_cut_ratio=result.cut_ratio,
        target_cut_ratio=(None if profile.get("cut_ratio") is None
                          else float(profile["cut_ratio"])),
        tolerance=float(profile.get("cut_ratio_tolerance", CUT_RATIO_TOLERANCE)),
        needs_visual_check=flags,
        dropped=dropped,
        source_duration=ceil - floor,
    )


def _invert_spans(cuts: list[Span], total: int) -> list[Span]:
    out: list[Span] = []
    cursor = 0
    for s in sorted(cuts, key=lambda x: x.start):
        if s.start > cursor:
            out.append(Span(cursor, s.start))
        cursor = max(cursor, s.end)
    if cursor < total:
        out.append(Span(cursor, total))
    return out


def _merge_close(spans: list[Span], padded: list[tuple[float, float]]
                 ) -> tuple[list[tuple[float, float]], list[Span], set[int]]:
    """Fuse kept ranges less than 120 ms apart; report which cuts that removed.

    A removal shorter than a blink costs a visible seam and a pair of audio
    fades to save nothing, and after padding a short removal is shorter still.
    Not cutting is strictly better, so the material comes back.
    """
    ranges: list[tuple[float, float]] = []
    kept: list[Span] = []
    dropped: set[int] = set()
    for i, (span, (a, b)) in enumerate(zip(spans, padded)):
        if ranges and a - ranges[-1][1] < MERGE_GAP_MS / 1000.0:
            ranges[-1] = (ranges[-1][0], b)
            kept[-1] = Span(kept[-1].start, span.end)
            dropped.add(i - 1)  # the cut between kept range i-1 and i
        else:
            ranges.append((a, b))
            kept.append(span)
    return ranges, kept, dropped


def _place_cut_times(cuts: list[CutSpan], ranges: list[tuple[float, float]],
                     floor: float, ceil: float) -> None:
    """Give each surviving cut the hole it leaves on the source timeline."""
    for i, cut in enumerate(cuts):
        cut.start = ranges[i][1] if i < len(ranges) else floor
        cut.end = ranges[i + 1][0] if i + 1 < len(ranges) else ceil
        if cut.end < cut.start:
            cut.end = cut.start


def _build_ranges(doc: WordsDoc, source_name: str, ranges: list[tuple[float, float]],
                  kept_spans: list[Span], cuts: list[CutSpan]) -> list[edl.Range]:
    out: list[edl.Range] = []
    for i, ((a, b), span) in enumerate(zip(ranges, kept_spans)):
        cut_before = None
        if i > 0 and i - 1 < len(cuts):
            c = cuts[i - 1]
            cut_before = {"class": c.klass, "text": _clip(c.text), "reason": c.reason,
                          "removed_s": round(c.duration, 3)}
            if c.needs_visual_check:
                cut_before["needs_visual_check"] = True
        out.append(edl.Range(
            source=source_name, start=a, end=b,
            # Quotes are report copy, and the report is read by a human on a
            # phone: keep the whole transcript out of the EDL.
            quote=_clip(diff_align.span_text(doc, span)),
            reason="kept",
            cut_before=cut_before,
        ))
    return out


def _clip(text: str, limit: int = 200) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


# -------- cli ----------------------------------------------------------------


def _load_kept_text(args: argparse.Namespace) -> str:
    if args.kept_text:
        return Path(args.kept_text).read_text(encoding="utf-8")
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    return plan["kept_text"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="derive_cuts",
        description="Diff an edited transcript against words.json and emit cut ranges.")
    ap.add_argument("--words", required=True, help="path to <name>.words.json")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--kept-text", help="file holding the edited transcript")
    src.add_argument("--plan", help="edit_plan.json holding kept_text")
    ap.add_argument("--profile", help="style_profile.json (for cut_ratio and fillers)")
    ap.add_argument("--source-name", help="EDL source key (default: words.json source name)")
    ap.add_argument("--out", help="write the plan as JSON here")
    args = ap.parse_args(argv)

    doc = WordsDoc.load(args.words)
    profile = json.loads(Path(args.profile).read_text(encoding="utf-8")) if args.profile else {}
    name = args.source_name or doc.source.get("name") or Path(args.words).stem.split(".")[0]

    plan = derive(doc, _load_kept_text(args), profile, name)
    print(plan.summary())
    for c in plan.cuts:
        flag = " [check]" if c.needs_visual_check else ""
        print(f"  {c.start:8.2f}-{c.end:7.2f}  {c.klass:<12}{flag} {_clip(c.text, 70)}")
    if not plan.within_band:
        print(f"  ! cut ratio is {plan.cut_ratio_delta:+.3f} outside the profile band")
    if args.out:
        Path(args.out).write_text(json.dumps(plan.to_dict(), ensure_ascii=False, indent=1),
                                  encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
