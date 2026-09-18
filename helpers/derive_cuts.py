"""Text diff to cut ranges: the only place a cut is ever decided.

The planner hands back the transcript with material deleted. Nothing else. This
module turns that text into timed ranges by diffing it against the aligned word
list, so every number on the timeline comes from the aligner and none from a
model (hard rule 12).

The order of operations is fixed, and it matters:

    diff -> move unsafe boundaries into silence -> invert to kept spans ->
    pad the edges -> merge ranges that were never worth cutting apart

Boundary safety before padding, because a boundary that moves changes which
silence the pad lives in. Padding before merging, because a pad that eats the
whole removal proves the cut was not worth making.
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
CHECK_GAP_MS = 150.0    # cut here, but the caller should eyeball it first
MERGE_GAP_MS = 120.0    # two kept ranges closer than this are really one range

# How far a boundary may travel to find silence. A cut point that has to move a
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

    Deleting is reversible and checkable; rewriting is neither. A paraphrase
    cannot be turned into cuts at all -- invented words have no timings -- so
    this is fatal rather than something to quietly drop.
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
    text: str                   # what was removed, display form
    klass: str
    reason: str
    gap_before_ms: float        # silence at the head boundary of the removal
    gap_after_ms: float         # silence at the tail boundary
    start: float = 0.0          # the hole it leaves on the source timeline,
    end: float = 0.0            # i.e. padded end of the range before it -> next
    needs_visual_check: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
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
    """Everything the renderer and the decision report need about one edit."""

    ranges: list[edl.Range]
    cuts: list[CutSpan]
    source_name: str
    kept_spans: list[Span]              # word-index spans, one per EDL range
    cut_ratio: float                    # achieved, by time, after pad and merge
    word_cut_ratio: float               # achieved, by word count (the diff's number)
    target_cut_ratio: float | None
    tolerance: float = CUT_RATIO_TOLERANCE
    needs_visual_check: list[dict] = field(default_factory=list)
    dropped: list[CutSpan] = field(default_factory=list)  # cuts the merge undid
    source_duration: float = 0.0

    @property
    def kept_duration(self) -> float:
        return sum(r.duration for r in self.ranges)

    @property
    def cut_ratio_delta(self) -> float | None:
        """Achieved minus target. Positive means this edit cut harder than the style."""
        if self.target_cut_ratio is None:
            return None
        return self.cut_ratio - self.target_cut_ratio

    @property
    def within_band(self) -> bool:
        d = self.cut_ratio_delta
        return True if d is None else abs(d) <= self.tolerance + 1e-9

    def range_of_word(self, index: int) -> int | None:
        """Which EDL range a source word ended up in, or None if it was cut.

        Insert markers travel as word indices, never as seconds (hard rule 12),
        so this is how an insert finds its place on the output timeline.
        """
        for i, s in enumerate(self.kept_spans):
            if s.start <= index < s.end:
                return i
        return None

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
        band = "no target" if delta is None else f"{delta:+.3f} vs target {self.target_cut_ratio}"
        return (f"{len(self.ranges)} ranges, {len(self.cuts)} cuts, "
                f"{self.kept_duration:.2f}s kept of {self.source_duration:.2f}s "
                f"(cut_ratio {self.cut_ratio:.3f}, {band}), "
                f"{len(self.needs_visual_check)} boundaries to eyeball")


# -------- timing helpers -----------------------------------------------------


def _source_bounds(doc: WordsDoc) -> tuple[float, float]:
    """`(0.0, duration)`. A take starts at zero even when speech does not."""
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
    if not (0 <= index < len(doc.words)):
        return 0.0
    w = doc.words[index]
    if not w.timed:
        return 0.0
    return round(max(0.0, (w.start - _prev_word_end(doc, index, floor)) * 1000.0), 6)


# Word times come out of the aligner as floats, so a gap that is exactly at a
# threshold lands a few femtoseconds either side of it. Without this slack the
# same 150 ms pause is usable in one take and not in the next.
_GAP_EPS_MS = 1e-3


def _gap_class(gap_ms: float) -> str:
    if gap_ms >= SAFE_GAP_MS - _GAP_EPS_MS:
        return "safe"
    if gap_ms >= CHECK_GAP_MS - _GAP_EPS_MS:
        return "check"
    return "unsafe"


def _boundary_time(doc: WordsDoc, index: int, floor: float, ceil: float) -> float:
    """Wall clock of the word boundary sitting in front of word `index`."""
    if index <= 0:
        return floor
    if index >= len(doc.words):
        return ceil
    return _next_word_start(doc, index, ceil)


# -------- boundary safety ----------------------------------------------------


def _safe_boundary(doc: WordsDoc, index: int, *, lo: int, hi: int,
                   floor: float, ceil: float) -> tuple[int, float, str, list[str]]:
    """Move a cut boundary to the nearest word boundary that sits in silence.

    `index` is the word in front of which the boundary lies; `lo`/`hi` bound how
    far it may move without colliding with the neighbouring edit. Under 150 ms
    of silence there is no cut point at all: the seam lands inside running
    speech and is audible however well it is padded, so the boundary walks
    outwards looking for a usable gap. Ties go to the candidate that KEEPS more
    material -- leaving a filler in is a smaller mistake than deleting a word
    the editor wanted.
    """
    notes: list[str] = []
    gap = _gap_ms_before(doc, index, floor)
    if index <= 0 or index >= len(doc.words):
        # The head and the tail of the take are silence by construction.
        return index, gap, "safe", notes
    if _gap_class(gap) != "unsafe":
        return index, gap, _gap_class(gap), notes

    origin = _boundary_time(doc, index, floor, ceil)
    best_rank: tuple | None = None
    best: tuple[int, float, str] | None = None
    for cand in range(max(lo, index - MAX_SHIFT_WORDS), min(hi, index + MAX_SHIFT_WORDS) + 1):
        if cand == index or cand <= 0 or cand >= len(doc.words):
            continue
        cand_gap = _gap_ms_before(doc, cand, floor)
        cls = _gap_class(cand_gap)
        if cls == "unsafe":
            continue
        dist = abs(_boundary_time(doc, cand, floor, ceil) - origin)
        if dist > MAX_SHIFT_S:
            continue
        rank = (round(dist, 4), 0 if cls == "safe" else 1, 0 if cand > index else 1)
        if best_rank is None or rank < best_rank:
            best_rank, best = rank, (cand, cand_gap, cls)

    if best is None:
        notes.append(f"no silence >= {CHECK_GAP_MS:.0f} ms within {MAX_SHIFT_WORDS} "
                     f"words; boundary left inside speech ({gap:.0f} ms)")
        return index, gap, "unsafe", notes
    notes.append(f"boundary moved {best[0] - index:+d} word(s) into {best[1]:.0f} ms of silence")
    return best[0], best[1], best[2], notes


@dataclass
class _Adjusted:
    """A cut after the silence rule has had its say."""

    span: Span
    gap_before_ms: float
    gap_after_ms: float
    cls_before: str
    cls_after: str
    moved: bool
    notes: list[str]


def _adjusted_cuts(doc: WordsDoc, cuts: list[Span], floor: float,
                   ceil: float) -> list[_Adjusted]:
    """Snap every cut boundary into silence; drop cuts that shrink to nothing."""
    out: list[_Adjusted] = []
    n = len(doc.words)
    for idx, span in enumerate(cuts):
        # A boundary may not cross into the neighbouring cut, nor past the far
        # boundary of this one.
        prev_end = out[-1].span.end if out else 0
        next_start = cuts[idx + 1].start if idx + 1 < len(cuts) else n

        start, gap_before, cls_before, notes = _safe_boundary(
            doc, span.start, lo=prev_end, hi=span.end, floor=floor, ceil=ceil)
        end, gap_after, cls_after, notes_after = _safe_boundary(
            doc, span.end, lo=max(start + 1, prev_end), hi=next_start, floor=floor, ceil=ceil)
        if end <= start:
            # Both boundaries collapsed onto each other: the removal could not
            # survive the silence rule, so nothing is cut here.
            continue
        out.append(_Adjusted(
            span=Span(start, end), gap_before_ms=gap_before, gap_after_ms=gap_after,
            cls_before=cls_before, cls_after=cls_after,
            moved=(start != span.start or end != span.end),
            notes=notes + notes_after))
    return out


# -------- classification -----------------------------------------------------


def _filler_tokens(profile: dict) -> set[str]:
    """Normalised filler vocabulary. The profile wins; the seed config is cold start."""
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

    Heuristic on purpose. `zeta learn` asks the LLM to classify PAST cuts, where
    the published video is the evidence; at plan time the class is report copy
    only, and a wrong label must never change what gets rendered.
    """
    n = len(doc.words)
    removed = [w.norm for w in doc.words[span.start:span.end] if w.norm]
    following = [w.norm for w in doc.words[span.end:span.end + 40] if w.norm]
    a, b = doc.span_time(span.start, span.end)
    span_s = max(0.0, b - a)

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
            return "retake", "the same content is said again later; earlier take dropped"
    if span_s >= 0.4 and _speech_seconds(doc, span) / span_s < 0.4:
        return "dead_air", f"{span_s:.2f}s span that is mostly silence"
    if len(removed) >= 12:
        return "tangent", f"{len(removed)} words off the story line"
    return "other", f"{len(removed)} words removed by the editor"


# -------- the engine ---------------------------------------------------------


def _pad_s(profile: dict, key: str, default: float) -> float:
    """Pad from the profile, clamped to the 30-200 ms window of hard rule 7."""
    raw = config.get(profile, f"cuts.{key}", default)
    try:
        val = float(raw)
    except (TypeError, ValueError):
        val = default
    return min(PAD_MAX_MS, max(PAD_MIN_MS, val)) / 1000.0


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


def _pad_ranges(doc: WordsDoc, kept: list[Span], profile: dict,
                floor: float, ceil: float) -> list[tuple[float, float]]:
    """Word-boundary ranges plus breathing room, taken only from adjacent silence.

    The pad may use the silence next to the boundary and nothing more. Reaching
    further would put the cut point inside a word -- a removed word is still a
    word (hard rule 6) -- and would let the pad of one range run into its
    neighbour, which the EDL rejects as overlapping ranges.
    """
    before = _pad_s(profile, "pad_before_ms", PAD_BEFORE_MS)
    after = _pad_s(profile, "pad_after_ms", PAD_AFTER_MS)
    out: list[tuple[float, float]] = []
    for span in kept:
        a, b = doc.span_time(span.start, span.end)
        head_room = max(0.0, a - _prev_word_end(doc, span.start, floor))
        tail_room = max(0.0, _next_word_start(doc, span.end, ceil) - b)
        out.append((max(floor, a - min(before, head_room)),
                    min(ceil, b + min(after, tail_room))))
    return out


def _merge_close(padded: list[tuple[float, float]]) -> list[list[int]]:
    """Group kept ranges that end up less than 120 ms apart.

    A removal shorter than a blink buys nothing and costs a visible seam plus a
    pair of audio fades, and after padding a short removal is shorter still. Not
    cutting is strictly better, so the material comes back.
    """
    groups: list[list[int]] = []
    for i, (a, _b) in enumerate(padded):
        if groups and a - padded[groups[-1][-1]][1] < MERGE_GAP_MS / 1000.0:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


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
    n = len(doc.words)
    adjusted = _adjusted_cuts(doc, list(result.cut), floor, ceil)
    kept_spans = _invert_spans([a.span for a in adjusted], n)
    if not kept_spans:
        raise ValueError("every word was cut: nothing left to render")

    padded = _pad_ranges(doc, kept_spans, profile, floor, ceil)
    groups = _merge_close(padded)
    ranges = [(padded[g[0]][0], padded[g[-1]][1]) for g in groups]
    merged_spans = [Span(kept_spans[g[0]].start, kept_spans[g[-1]].end) for g in groups]
    group_of = {i: gi for gi, g in enumerate(groups) for i in g}

    # Kept spans are the complement of the cuts, so a cut that does not start at
    # word 0 is preceded by exactly one kept span, and one that does not end at
    # the last word is followed by exactly one.
    ends_at = {s.end: i for i, s in enumerate(kept_spans)}
    starts_at = {s.start: i for i, s in enumerate(kept_spans)}

    fillers = _filler_tokens(profile)
    cuts: list[CutSpan] = []
    dropped: list[CutSpan] = []
    cut_before_group: dict[int, CutSpan] = {}

    for adj in adjusted:
        span = adj.span
        klass, reason = classify_cut(doc, span, fillers)
        record = CutSpan(
            span=span, text=diff_align.span_text(doc, span), klass=klass, reason=reason,
            gap_before_ms=adj.gap_before_ms, gap_after_ms=adj.gap_after_ms,
            # Anything but two wide silences gets a look: a moved boundary means
            # this cut no longer removes exactly what the editor asked for.
            needs_visual_check=(adj.moved
                                or "check" in (adj.cls_before, adj.cls_after)
                                or "unsafe" in (adj.cls_before, adj.cls_after)),
            notes=list(adj.notes),
        )
        if "unsafe" in (adj.cls_before, adj.cls_after):
            record.notes.append("no safe silence at one edge: check before rendering")

        prev_i = ends_at.get(span.start)
        next_i = starts_at.get(span.end)
        if (prev_i is not None and next_i is not None
                and group_of[prev_i] == group_of[next_i]):
            # The merge pass put the two sides back together: this cut is gone.
            # Report it where the words still are, not at the seam that no
            # longer exists.
            record.start, record.end = doc.span_time(span.start, span.end)
            record.notes.append(
                f"restored: the removal was shorter than {MERGE_GAP_MS:.0f} ms once padded")
            dropped.append(record)
            continue

        record.start = ranges[group_of[prev_i]][1] if prev_i is not None else floor
        record.end = ranges[group_of[next_i]][0] if next_i is not None else ceil
        cuts.append(record)
        if next_i is not None:
            cut_before_group[group_of[next_i]] = record

    edl_ranges = [
        edl.Range(
            source=source_name, start=a, end=b,
            # Quotes are report copy read on a phone: keep the whole transcript
            # out of the EDL.
            quote=_clip(diff_align.span_text(doc, span)),
            reason="kept",
            cut_before=_cut_before_dict(cut_before_group.get(gi)),
        )
        for gi, ((a, b), span) in enumerate(zip(ranges, merged_spans))
    ]

    kept_duration = sum(b - a for a, b in ranges)
    total = max(1e-9, ceil - floor)
    return CutPlan(
        ranges=edl_ranges,
        cuts=cuts,
        source_name=source_name,
        kept_spans=merged_spans,
        cut_ratio=max(0.0, 1.0 - kept_duration / total),
        word_cut_ratio=result.cut_ratio,
        target_cut_ratio=(None if profile.get("cut_ratio") is None
                          else float(profile["cut_ratio"])),
        tolerance=float(profile.get("cut_ratio_tolerance", CUT_RATIO_TOLERANCE)),
        needs_visual_check=[_flag(source_name, c) for c in cuts if c.needs_visual_check],
        dropped=dropped,
        source_duration=ceil - floor,
    )


def _cut_before_dict(cut: CutSpan | None) -> dict | None:
    if cut is None:
        return None
    d = {"class": cut.klass, "text": _clip(cut.text), "reason": cut.reason,
         "removed_s": round(cut.duration, 3)}
    if cut.needs_visual_check:
        d["needs_visual_check"] = True
    return d


def _flag(source_name: str, c: CutSpan) -> dict:
    return {
        "source": source_name,
        "start": round(c.start, 3),
        "end": round(c.end, 3),
        "gap_before_ms": round(c.gap_before_ms, 1),
        "gap_after_ms": round(c.gap_after_ms, 1),
        "class": c.klass,
        "why": "; ".join(c.notes) or
               f"cut point sits in {min(c.gap_before_ms, c.gap_after_ms):.0f} ms of silence "
               f"(< {SAFE_GAP_MS:.0f} ms): run timeline_view here",
    }


def _clip(text: str, limit: int = 200) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


# -------- cli ----------------------------------------------------------------


def _load_kept_text(args: argparse.Namespace) -> str:
    if args.kept_text:
        return Path(args.kept_text).read_text(encoding="utf-8")
    return json.loads(Path(args.plan).read_text(encoding="utf-8"))["kept_text"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="derive_cuts",
        description="Diff an edited transcript against words.json and emit cut ranges.")
    ap.add_argument("--words", required=True, help="path to <name>.words.json")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--kept-text", help="file holding the edited transcript")
    src.add_argument("--plan", help="edit_plan.json holding kept_text")
    ap.add_argument("--profile", help="style_profile.json (cut_ratio and fillers)")
    ap.add_argument("--source-name", help="EDL source key (default: the words.json source)")
    ap.add_argument("--out", help="write the derived plan as JSON here")
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
