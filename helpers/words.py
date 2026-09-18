"""The canonical word-level transcript: one artifact, two consumers.

`words.json` is read by the caption builder AND by the cut engine. Nothing else
in the pipeline is allowed to hold timing. Shape:

    {
      "schema": "zeta.words.v1",
      "source": {"name": "raw01", "path": "/abs/raw01.mp4", "sha256": "...",
                 "duration_s": 612.4},
      "language": "ary",
      "backend": "gemini_flash_lite",
      "aligner": "whisperx:jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
      "words": [
        {"word": "الشركة", "display": "الشركة", "alias": "asharika",
         "start": 12.41, "end": 12.79, "score": 0.91, "lang": "ary",
         "src": "gemini", "confirmed_by": ["cohere"]}
      ],
      "meta": {...}
    }

`word` is the matching form, `display` is what the viewer reads, `alias` is the
grapheme string handed to the aligner when the raw token has no acoustic model
coverage (digits, Latin names inside an Arabic model).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable

from . import textnorm

SCHEMA = "zeta.words.v1"


@dataclass
class Word:
    word: str
    start: float | None = None
    end: float | None = None
    display: str | None = None
    alias: str | None = None
    score: float | None = None
    lang: str | None = None
    src: str | None = None
    confirmed_by: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.display is None:
            self.display = self.word
        if self.lang is None:
            self.lang = textnorm.token_lang(self.word)

    @property
    def timed(self) -> bool:
        return self.start is not None and self.end is not None

    @property
    def duration(self) -> float:
        return (self.end - self.start) if self.timed else 0.0

    @property
    def norm(self) -> str:
        return textnorm.normalize_token(self.word)

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: v for k, v in d.items() if v not in (None, [], "")}

    @classmethod
    def from_dict(cls, d: dict) -> "Word":
        return cls(
            word=d.get("word") or d.get("text") or "",
            start=_f(d.get("start")),
            end=_f(d.get("end")),
            display=d.get("display"),
            alias=d.get("alias"),
            score=_f(d.get("score")),
            lang=d.get("lang"),
            src=d.get("src"),
            confirmed_by=list(d.get("confirmed_by") or []),
        )


def _f(v: Any) -> float | None:
    return None if v is None else float(v)


@dataclass
class WordsDoc:
    words: list[Word] = field(default_factory=list)
    source: dict = field(default_factory=dict)
    language: str = "ary"
    backend: str | None = None
    aligner: str | None = None
    meta: dict = field(default_factory=dict)

    # -- io -----------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> "WordsDoc":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "WordsDoc":
        return cls(
            words=[Word.from_dict(w) for w in data.get("words", [])],
            source=data.get("source", {}),
            language=data.get("language", "ary"),
            backend=data.get("backend"),
            aligner=data.get("aligner"),
            meta=data.get("meta", {}),
        )

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA,
            "source": self.source,
            "language": self.language,
            "backend": self.backend,
            "aligner": self.aligner,
            "words": [w.to_dict() for w in self.words],
            "meta": self.meta,
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
        return p

    # -- views --------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.words)

    def text(self, sep: str = " ") -> str:
        return sep.join(w.display or w.word for w in self.words)

    def timed_words(self) -> list[Word]:
        return [w for w in self.words if w.timed]

    def in_range(self, start: float, end: float) -> list[Word]:
        """Words that overlap [start, end)."""
        out = []
        for w in self.words:
            if not w.timed:
                continue
            if w.end <= start or w.start >= end:
                continue
            out.append(w)
        return out

    def index_range(self, start: float, end: float) -> tuple[int, int]:
        """Half-open word-index span covering [start, end)."""
        lo, hi = None, 0
        for i, w in enumerate(self.words):
            if not w.timed or w.end <= start or w.start >= end:
                continue
            if lo is None:
                lo = i
            hi = i + 1
        return (0, 0) if lo is None else (lo, hi)

    def span_time(self, i: int, j: int) -> tuple[float, float]:
        """Wall-clock [start, end) of the half-open word span [i, j)."""
        chunk = [w for w in self.words[i:j] if w.timed]
        if not chunk:
            return (0.0, 0.0)
        return (chunk[0].start, chunk[-1].end)

    def duration(self) -> float:
        timed = self.timed_words()
        return (timed[-1].end - timed[0].start) if timed else 0.0

    def gaps(self) -> list[float]:
        """Silence before each timed word, in seconds (first word: 0)."""
        timed = self.timed_words()
        out = [0.0]
        for prev, cur in zip(timed, timed[1:]):
            out.append(max(0.0, cur.start - prev.end))
        return out

    # -- quality ------------------------------------------------------------
    def coverage(self) -> float:
        return (len(self.timed_words()) / len(self.words)) if self.words else 0.0

    def overlaps(self) -> list[tuple[int, int]]:
        timed = [(i, w) for i, w in enumerate(self.words) if w.timed]
        return [
            (a[0], b[0])
            for a, b in zip(timed, timed[1:])
            # 1 ms of slack: aligners emit exact-touching boundaries.
            if b[1].start < a[1].end - 1e-3
        ]

    def long_words(self, max_duration_s: float) -> list[int]:
        return [i for i, w in enumerate(self.words) if w.timed and w.duration > max_duration_s]


def from_plain(text: str, **kw: Any) -> WordsDoc:
    """A WordsDoc with no timings, from plain text. Used before alignment."""
    return WordsDoc(words=[Word(word=t) for t in textnorm.tokenize(text)], **kw)


def from_segments(segments: Iterable[dict], src: str | None = None, **kw: Any) -> WordsDoc:
    """Flatten `[{start, end, text}]` segments into untimed words.

    Segment timestamps from an LLM transcriber are approximate by construction,
    so they are kept only as `meta.segments` hints for the aligner and never
    written onto the words themselves (hard rule: the model does not emit time).
    """
    seg_list = list(segments)
    doc = WordsDoc(**kw)
    for s in seg_list:
        for tok in textnorm.tokenize(s.get("text", "")):
            doc.words.append(Word(word=tok, src=src))
    doc.meta["segments"] = [
        {"start": _f(s.get("start")), "end": _f(s.get("end")), "text": s.get("text", "")}
        for s in seg_list
    ]
    return doc
