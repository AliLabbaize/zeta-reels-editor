"""EDL v2 - build, validate, and map source time to output time.

v2 is v1 plus per-range editorial provenance and per-overlay source metadata.
`helpers/render.py` (vendored, unchanged) reads only the v1 subset -- `sources`,
`ranges[].source/start/end`, `overlays[].file/start_in_output/duration`,
`subtitles` -- and ignores the rest, so v2 files render on the upstream code.

The output timeline is the concatenation of the ranges in order, so:

    output_time = source_time - range.start + range.offset

Everything downstream of the cut (captions, overlays, self-eval) uses
`to_output_time`; nothing recomputes that arithmetic on its own.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

VERSION = 2


@dataclass
class Range:
    source: str
    start: float
    end: float
    beat: str | None = None
    quote: str | None = None
    reason: str | None = None
    cut_before: dict | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"source": self.source,
                             "start": round(self.start, 3),
                             "end": round(self.end, 3)}
        for k in ("beat", "quote", "reason", "cut_before"):
            v = getattr(self, k)
            if v:
                d[k] = v
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Range":
        return cls(source=d["source"], start=float(d["start"]), end=float(d["end"]),
                   beat=d.get("beat"), quote=d.get("quote"), reason=d.get("reason"),
                   cut_before=d.get("cut_before"))


@dataclass
class Overlay:
    file: str
    start_in_output: float
    duration: float
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {"file": self.file,
             "start_in_output": round(self.start_in_output, 3),
             "duration": round(self.duration, 3)}
        if self.meta:
            d["meta"] = self.meta
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Overlay":
        return cls(file=d["file"], start_in_output=float(d["start_in_output"]),
                   duration=float(d["duration"]), meta=d.get("meta", {}))


class EDLError(ValueError):
    """An EDL that would render wrong. Always fatal -- never rendered anyway."""


@dataclass
class EDL:
    sources: dict[str, str] = field(default_factory=dict)
    ranges: list[Range] = field(default_factory=list)
    overlays: list[Overlay] = field(default_factory=list)
    subtitles: str | None = None
    style_profile: str | None = None
    aspect: str | None = None
    grade: str | None = None
    meta: dict = field(default_factory=dict)

    # -- timeline -----------------------------------------------------------
    @property
    def total_duration_s(self) -> float:
        return sum(r.duration for r in self.ranges)

    def offsets(self) -> list[float]:
        """Output-timeline start of each range."""
        out, acc = [], 0.0
        for r in self.ranges:
            out.append(acc)
            acc += r.duration
        return out

    def to_output_time(self, source: str, t: float) -> float | None:
        """Map a source timestamp to the output timeline (None if it was cut)."""
        for r, off in zip(self.ranges, self.offsets()):
            if r.source == source and r.start <= t < r.end:
                return t - r.start + off
        return None

    def to_source_time(self, t_out: float) -> tuple[str, float] | None:
        for r, off in zip(self.ranges, self.offsets()):
            if off <= t_out < off + r.duration:
                return (r.source, r.start + (t_out - off))
        return None

    # -- io -----------------------------------------------------------------
    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "version": VERSION,
            "sources": self.sources,
            "ranges": [r.to_dict() for r in self.ranges],
        }
        if self.grade:
            d["grade"] = self.grade
        if self.overlays:
            d["overlays"] = [o.to_dict() for o in self.overlays]
        if self.subtitles:
            d["subtitles"] = self.subtitles
        if self.style_profile:
            d["style_profile"] = self.style_profile
        if self.aspect:
            d["aspect"] = self.aspect
        if self.meta:
            d["meta"] = self.meta
        d["total_duration_s"] = round(self.total_duration_s, 3)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "EDL":
        return cls(
            sources=dict(d.get("sources", {})),
            ranges=[Range.from_dict(r) for r in d.get("ranges", [])],
            overlays=[Overlay.from_dict(o) for o in d.get("overlays", [])],
            subtitles=d.get("subtitles"),
            style_profile=d.get("style_profile"),
            aspect=d.get("aspect"),
            grade=d.get("grade"),
            meta=d.get("meta", {}),
        )

    @classmethod
    def load(cls, path: str | Path) -> "EDL":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> Path:
        self.validate()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
        return p

    # -- validation ---------------------------------------------------------
    def validate(self, *, strict_overlays: bool = True) -> "EDL":
        if not self.ranges:
            raise EDLError("EDL has no ranges: nothing to render")
        for i, r in enumerate(self.ranges):
            if r.source not in self.sources:
                raise EDLError(f"range {i}: source {r.source!r} not in sources")
            if r.end <= r.start:
                raise EDLError(f"range {i}: end {r.end} <= start {r.start}")
            if r.start < 0:
                raise EDLError(f"range {i}: negative start {r.start}")

        # Ranges of the same source must not overlap: overlapping ranges play
        # the same audio twice and break the output-time mapping above.
        by_source: dict[str, list[Range]] = {}
        for r in self.ranges:
            by_source.setdefault(r.source, []).append(r)
        for src, rs in by_source.items():
            ordered = sorted(rs, key=lambda x: x.start)
            for a, b in zip(ordered, ordered[1:]):
                if b.start < a.end - 1e-6:
                    raise EDLError(
                        f"source {src}: ranges overlap ({a.start:.3f}-{a.end:.3f} "
                        f"and {b.start:.3f}-{b.end:.3f})")

        total = self.total_duration_s
        for i, o in enumerate(self.overlays):
            if o.duration <= 0:
                raise EDLError(f"overlay {i}: duration {o.duration} <= 0")
            if o.start_in_output < 0:
                raise EDLError(f"overlay {i}: negative start_in_output")
            if o.start_in_output + o.duration > total + 1e-3:
                raise EDLError(
                    f"overlay {i}: runs past the end of the cut "
                    f"({o.start_in_output + o.duration:.2f}s > {total:.2f}s)")
            if strict_overlays and o.meta:
                # Zeta hard rule 12: nothing unverified ships.
                if not o.meta.get("verified", False):
                    raise EDLError(f"overlay {i}: meta.verified is not true")
                if not o.meta.get("url"):
                    raise EDLError(f"overlay {i}: no source url")

        ordered_ov = sorted(self.overlays, key=lambda o: o.start_in_output)
        for a, b in zip(ordered_ov, ordered_ov[1:]):
            if b.start_in_output < a.start_in_output + a.duration - 1e-6:
                raise EDLError(
                    f"overlays overlap in time: {a.file} and {b.file}")
        return self


def build(
    sources: dict[str, str],
    ranges: Iterable[Range],
    *,
    overlays: Iterable[Overlay] = (),
    subtitles: str | None = None,
    style_profile: str | None = None,
    aspect: str | None = None,
    meta: dict | None = None,
) -> EDL:
    return EDL(sources=dict(sources), ranges=list(ranges), overlays=list(overlays),
               subtitles=subtitles, style_profile=style_profile, aspect=aspect,
               meta=meta or {})
