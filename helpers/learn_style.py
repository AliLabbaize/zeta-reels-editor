"""Stage 3: learn Zeta's editing style from episodes that already shipped.

The premise is that a published episode IS the raw take with material deleted,
so the same diff engine that applies an edit can read one back out of a pair of
videos. What Ali did becomes `style_profile.json`; how he did it becomes
`few_shot_examples.md`, which goes straight into the planner prompt.

    raw.words.json + published.words.json
        -> diff_align.diff_words        -> cut spans, with 5 s of context
        -> LLM                          -> a class for each cut
    published.mp4
        -> PySceneDetect + face check   -> insert spans
        -> Gemini vision                -> what each insert shows
        -> published words, -1s..+1s    -> the word that triggered it

Only aggregate statistics survive into the profile: the per-episode detail lives
in the examples file. Nothing here produces a timestamp that any later stage
renders from -- this module reads the past, `derive_cuts` writes the future.
"""

from __future__ import annotations

# See derive_cuts: makes `python helpers/learn_style.py` work as well as imports.
if __package__ in (None, ""):  # pragma: no cover - script entry only
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    __package__ = "helpers"

import argparse
import csv
import json
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

from . import config, derive_cuts, diff_align, gemini_client, paths, textnorm
from .diff_align import Span
from .words import WordsDoc

CUT_CLASSES = derive_cuts.CLASSES
CONTEXT_S = 5.0                 # spec step 3: 5 s of context each side of a cut
TRIGGER_WINDOW_S = 1.0          # spec step 6: -1 s .. +1 s around an insert start
FACE_SHRINK_RATIO = 0.40        # spec step 4: below 40 % of the median face box
MIN_INSERT_S = 0.6              # shorter than this is a transition, not an insert
LONG_PAUSE_S = 3.0              # longer than this is a structural break, not a beat

CLASSIFY_BATCH = 20


class MissingExtra(RuntimeError):
    """An optional dependency is needed for this step and is not installed."""


# -------- records -------------------------------------------------------------


@dataclass
class Pair:
    published: Path
    raw: Path | None = None
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = self.published.stem

    @property
    def has_raw(self) -> bool:
        return self.raw is not None


@dataclass
class CutObservation:
    """One thing Ali removed, on the RAW timeline."""

    pair: str
    span: Span
    start: float
    end: float
    text: str
    before: str
    after: str
    klass: str = "other"
    why: str = ""
    at_head: bool = False
    at_tail: bool = False

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        return {"pair": self.pair, "start": round(self.start, 3), "end": round(self.end, 3),
                "words": [self.span.start, self.span.end], "class": self.klass,
                "text": self.text, "before": self.before, "after": self.after,
                "why": self.why, "at_head": self.at_head, "at_tail": self.at_tail}


@dataclass
class InsertObservation:
    """One visual in the published cut, on the PUBLISHED timeline."""

    pair: str
    start: float
    end: float
    frame: str | None = None
    face_area: float = 0.0          # face box area in the sampled frame, px^2
    median_face_area: float = 0.0   # median face box over the whole episode
    label: dict = field(default_factory=dict)
    trigger_word: str = ""
    trigger_word_index: int = -1
    trigger_kind: str = ""
    trigger_time: float | None = None
    trigger_text: str = ""
    detector: str = "scenedetect"

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def lead_in_s(self) -> float | None:
        """How long the visual is up before the word that justifies it is said."""
        if self.trigger_time is None:
            return None
        return round(self.trigger_time - self.start, 3)

    @property
    def pip_scale(self) -> float | None:
        """Facecam PiP size, inferred from how much the face box shrank."""
        if self.face_area <= 0 or self.median_face_area <= 0:
            return None
        return round((self.face_area / self.median_face_area) ** 0.5, 3)

    def to_dict(self) -> dict:
        return {"pair": self.pair, "start": round(self.start, 3), "end": round(self.end, 3),
                "duration_s": round(self.duration, 3), "frame": self.frame,
                "label": self.label, "trigger_word": self.trigger_word,
                "trigger_word_index": self.trigger_word_index,
                "trigger_kind": self.trigger_kind, "trigger_text": self.trigger_text,
                "lead_in_s": self.lead_in_s, "pip_scale": self.pip_scale,
                "detector": self.detector}


@dataclass
class PairObservation:
    name: str
    has_raw: bool
    raw_duration: float = 0.0
    published_duration: float = 0.0
    cuts: list[CutObservation] = field(default_factory=list)
    inserts: list[InsertObservation] = field(default_factory=list)
    kept_gaps_ms: list[float] = field(default_factory=list)
    word_cut_ratio: float = 0.0
    published_tokens: list[str] = field(default_factory=list)

    @property
    def cut_ratio(self) -> float:
        """Time-based, to match what `derive_cuts` reports at plan time."""
        if self.has_raw and self.raw_duration > 0 and self.published_duration > 0:
            return max(0.0, 1.0 - self.published_duration / self.raw_duration)
        return self.word_cut_ratio

    def to_dict(self) -> dict:
        return {"pair": self.name, "has_raw": self.has_raw,
                "raw_duration_s": round(self.raw_duration, 3),
                "published_duration_s": round(self.published_duration, 3),
                "cut_ratio": round(self.cut_ratio, 4),
                "cuts": [c.to_dict() for c in self.cuts],
                "inserts": [i.to_dict() for i in self.inserts]}


# -------- input ---------------------------------------------------------------


def load_pairs(csv_path: str | Path) -> list[Pair]:
    """Read `pairs.csv`: `raw,published`, with an empty raw allowed.

    A published-only row still teaches insert density, layout and triggers; it
    just cannot teach what was removed, because nothing says what was said.
    """
    rows: list[Pair] = []
    base = Path(csv_path).resolve().parent
    with Path(csv_path).open(encoding="utf-8", newline="") as fh:
        for n, row in enumerate(csv.reader(fh)):
            cells = [c.strip() for c in row if c is not None]
            if not cells or not any(cells) or cells[0].startswith("#"):
                continue
            if n == 0 and cells[0].lower() in ("raw", "raw_path"):
                continue
            raw_cell = cells[0]
            pub_cell = cells[1] if len(cells) > 1 and cells[1] else raw_cell
            if len(cells) == 1 or not cells[1]:
                raw_cell = ""          # published-only row
            rows.append(Pair(published=_resolve(base, pub_cell),
                             raw=_resolve(base, raw_cell) if raw_cell else None,
                             name=cells[2] if len(cells) > 2 else ""))
    if not rows:
        raise ValueError(f"{csv_path} has no usable rows (expected `raw,published`)")
    return rows


def _resolve(base: Path, cell: str) -> Path:
    p = Path(cell).expanduser()
    return p if p.is_absolute() else (base / p)


def words_for(media: Path, *, edit_paths: paths.EditPaths | None = None,
              transcribe: Callable[[Path], WordsDoc] | None = None) -> WordsDoc:
    """The word doc for a media file: cached next to it, or freshly transcribed.

    Hard rule 9 is why the cache is checked first: learning over ten episodes
    twice must not transcribe twenty files twice.
    """
    for cand in _words_candidates(media, edit_paths):
        if cand.exists():
            return WordsDoc.load(cand)
    if transcribe is not None:
        return transcribe(media)
    return _transcribe_via_stage1(media)


def _words_candidates(media: Path, edit_paths: paths.EditPaths | None) -> list[Path]:
    out = [media.with_suffix(".words.json"), media.parent / f"{media.stem}.words.json"]
    if edit_paths is not None:
        out.append(edit_paths.words_json(media.stem))
    out.append(paths.EditPaths.for_videos_dir(media.parent).words_json(media.stem))
    return out


def _transcribe_via_stage1(media: Path) -> WordsDoc:
    """Lazy bridge to stage 1. Kept lazy so learn mode imports without torch."""
    try:
        from . import transcribe_gemini  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on the sibling module
        raise MissingExtra(
            f"no transcript for {media.name} and the stage-1 transcriber is not "
            f"available ({exc}). Run `zeta transcribe {media}` first, or put "
            f"{media.stem}.words.json next to the file.") from exc
    for attr in ("transcribe_file", "transcribe", "run"):
        fn = getattr(transcribe_gemini, attr, None)
        if callable(fn):
            return fn(media)
    raise MissingExtra(  # pragma: no cover
        f"helpers.transcribe_gemini exposes no transcribe entry point for {media.name}")


# -------- step 2: the diff ----------------------------------------------------


def diff_pair(raw_doc: WordsDoc, published_doc: WordsDoc, pair_name: str
              ) -> tuple[list[CutObservation], float]:
    """Cut spans of the raw take, through the one diff engine.

    Returns the observations and the word-level cut ratio. Spans with no timing
    are dropped: an untimed word cannot teach anything about pacing.
    """
    result = diff_align.diff_words(raw_doc, published_doc)
    n = len(raw_doc.words)
    out: list[CutObservation] = []
    for span in result.cut:
        start, end = raw_doc.span_time(span.start, span.end)
        if end <= start:
            continue
        before, after = diff_align.context_around(raw_doc, span, CONTEXT_S)
        out.append(CutObservation(
            pair=pair_name, span=span, start=start, end=end,
            text=diff_align.span_text(raw_doc, span), before=before, after=after,
            at_head=(span.start == 0), at_tail=(span.end >= n)))
    return out, result.cut_ratio


# -------- step 3: classify the cuts ------------------------------------------

CLASSIFY_SYSTEM = """\
You are studying how a Darija news channel edits its own footage. For each span
that the editor DELETED you are given the words removed and the speech either
side of the hole. Say why it went, choosing exactly one class:

  filler       a hesitation or discourse crutch with no content
  false_start  an aborted phrase that is immediately restarted
  retake       content that is said again, more cleanly, later
  tangent      a complete but off-topic digression
  dead_air     silence, breathing, or a pause with no speech
  intro_trim   lead-in before the story starts
  outro_trim   tail after the story ends
  other        none of the above

Judge only from the evidence. Do not invent timestamps.
"""

CLASSIFY_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "cuts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "class": {"type": "string", "enum": list(CUT_CLASSES)},
                    "why": {"type": "string"},
                },
                "required": ["index", "class"],
                "propertyOrdering": ["index", "class", "why"],
            },
        }
    },
    "required": ["cuts"],
}


def classify_cuts(cuts: Sequence[CutObservation], raw_doc: WordsDoc, *,
                  llm: gemini_client.LLM | None = None,
                  profile: dict | None = None) -> list[CutObservation]:
    """Label every cut span. Falls back to the plan-time heuristic without a model.

    The heuristic is the same code `derive_cuts` uses, so an offline profile is
    consistent with an offline edit -- weaker, never contradictory.
    """
    if not cuts:
        return list(cuts)
    fillers = derive_cuts._filler_tokens(profile or {})
    for c in cuts:
        c.klass, c.why = derive_cuts.classify_cut(raw_doc, c.span, fillers)

    llm = llm or gemini_client.default_llm()
    if not llm.available:
        return list(cuts)

    for start in range(0, len(cuts), CLASSIFY_BATCH):
        batch = list(cuts)[start:start + CLASSIFY_BATCH]
        payload = [{"index": i, "removed": c.text, "before": c.before, "after": c.after,
                    "seconds": round(c.duration, 2),
                    "position": "head" if c.at_head else "tail" if c.at_tail else "middle"}
                   for i, c in enumerate(batch)]
        answer = llm.generate(
            "Classify these deleted spans:\n"
            + json.dumps(payload, ensure_ascii=False, indent=1),
            system=CLASSIFY_SYSTEM, schema=CLASSIFY_SCHEMA)
        if isinstance(answer, str):
            answer = json.loads(answer)
        for item in (answer or {}).get("cuts", []):
            i = int(item.get("index", -1))
            klass = item.get("class")
            if 0 <= i < len(batch) and klass in CUT_CLASSES:
                batch[i].klass = klass
                batch[i].why = item.get("why", "") or batch[i].why
    return list(cuts)


# -------- step 4: find the visual inserts ------------------------------------


def detect_inserts(video: str | Path, pair_name: str, frames_dir: str | Path, *,
                   threshold: float = 27.0, min_insert_s: float = MIN_INSERT_S
                   ) -> list[InsertObservation]:
    """Scenes of the published cut where the facecam is gone or shrunk to a PiP.

    Content detection alone would also fire on a jump cut in the facecam itself,
    which is why the face box decides: no face, or a face below 40 % of this
    episode's median face size, means something else owns the frame.
    """
    try:
        from scenedetect import ContentDetector, detect as scene_detect  # type: ignore
    except ImportError as exc:
        raise MissingExtra(
            "PySceneDetect is required to find inserts: "
            "uv pip install -e '.[learn]'") from exc
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise MissingExtra(
            "OpenCV is required for the face-presence check: "
            "uv pip install -e '.[learn]'") from exc

    video = Path(video)
    scenes = [(s.get_seconds(), e.get_seconds())
              for s, e in scene_detect(str(video), ContentDetector(threshold=threshold))]
    if not scenes:
        return []

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(str(video))
    try:
        sampled = [(a, b, _face_area(cap, cv2, cascade, (a + b) / 2.0)) for a, b in scenes]
    finally:
        cap.release()

    faces = [area for _a, _b, area in sampled if area > 0]
    median_face = statistics.median(faces) if faces else 0.0

    raw_hits = [(a, b, area) for a, b, area in sampled
                if b - a >= min_insert_s
                and (area <= 0 or (median_face and area < FACE_SHRINK_RATIO * median_face))]
    merged = _merge_scenes(raw_hits)

    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    out: list[InsertObservation] = []
    cap = cv2.VideoCapture(str(video))
    try:
        for i, (a, b, area) in enumerate(merged):
            # One frame per insert (spec step 4): a quarter in, past the fade.
            frame_path = frames_dir / f"{pair_name}_insert_{i:02d}.png"
            if _save_frame(cap, cv2, a + (b - a) * 0.25, frame_path):
                out.append(InsertObservation(pair=pair_name, start=a, end=b,
                                             frame=str(frame_path), face_area=area,
                                             median_face_area=median_face))
    finally:
        cap.release()
    return out


def _merge_scenes(hits: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    """One visual held across several detected scenes is still one insert."""
    out: list[tuple[float, float, float]] = []
    for a, b, area in hits:
        if out and a - out[-1][1] < 0.2:
            prev = out[-1]
            out[-1] = (prev[0], b, min(prev[2], area))
        else:
            out.append((a, b, area))
    return out


def _face_area(cap, cv2, cascade, t: float) -> float:
    frame = _grab(cap, cv2, t)
    if frame is None:
        return 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    boxes = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return float(max((w * h for _x, _y, w, h in boxes), default=0.0))


def _grab(cap, cv2, t: float):
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
    ok, frame = cap.read()
    return frame if ok else None


def _save_frame(cap, cv2, t: float, dest: Path) -> bool:
    frame = _grab(cap, cv2, t)
    if frame is None:
        return False
    return bool(cv2.imwrite(str(dest), frame))


# -------- step 5: label each insert frame ------------------------------------

VISION_PROMPT = """\
This is one frame from a Darija news reel, at a moment where a visual covers or
shares the frame with the presenter's facecam. Describe what the viewer sees.
Report only what is visible; if the source is not written in the frame, say
"unknown". Do not guess a URL.
"""

VISION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "type": {"type": "string",
                 "enum": ["article_headline", "x_post", "chart", "table", "code",
                          "logo", "screenshot", "other"]},
        "shows": {"type": "string"},
        "likely_source": {"type": "string"},
        "layout": {"type": "string",
                   "enum": ["fullframe", "fullframe_pip", "split", "card", "other"]},
        "pip_corner": {"type": "string",
                       "enum": ["bottom_right", "bottom_left", "top_right",
                                "top_left", "none"]},
        "visible_text": {"type": "string"},
    },
    "required": ["type", "shows", "layout"],
    "propertyOrdering": ["type", "shows", "likely_source", "layout", "pip_corner",
                         "visible_text"],
}


def label_inserts(inserts: Sequence[InsertObservation], *,
                  llm: gemini_client.LLM | None = None) -> list[InsertObservation]:
    """Ask vision what each sampled frame shows. Silent no-op without a model."""
    llm = llm or gemini_client.default_llm()
    if not llm.available:
        return list(inserts)
    for ins in inserts:
        if not ins.frame or not Path(ins.frame).exists():
            continue
        answer = llm.vision_json(VISION_PROMPT, ins.frame, VISION_SCHEMA)
        if isinstance(answer, str):
            answer = json.loads(answer)
        if isinstance(answer, dict):
            ins.label = answer
    return list(inserts)


# -------- step 6: what triggered each insert ---------------------------------

_NUMBER = re.compile(r"\d")
_URL = re.compile(r"(https?://|www\.|\.com|\.ma\b)", re.IGNORECASE)
_QUOTE = re.compile(r"[\"«»“”]")
_DATE = re.compile(r"\b(19|20)\d{2}\b")
_ENTITY = re.compile(r"^[A-Z][A-Za-z0-9&.\-]{1,}$")


def extract_trigger(words: Sequence) -> tuple[int, str, str] | None:
    """First company, person, number, quote, date or URL in the window.

    Ordered by how specific the evidence is: a URL or a figure is unambiguously
    what the screenshot is for, a capitalised Latin token is usually the company
    or person, and everything else is a guess not worth making.
    """
    for i, w in enumerate(words):
        token = w.display or w.word
        if _URL.search(token):
            return i, token, "url"
    for i, w in enumerate(words):
        token = w.display or w.word
        if _DATE.search(token):
            return i, token, "date"
        if _NUMBER.search(token):
            return i, token, "number"
    for i, w in enumerate(words):
        token = w.display or w.word
        if _QUOTE.search(token):
            return i, token, "quote"
    for i, w in enumerate(words):
        token = w.display or w.word
        if textnorm.is_latin(token) and _ENTITY.match(token):
            return i, token, "entity"
    return None


def map_insert_triggers(inserts: Sequence[InsertObservation], published_doc: WordsDoc, *,
                        window_s: float = TRIGGER_WINDOW_S) -> list[InsertObservation]:
    """Attach the spoken trigger to each insert, from -1 s to +1 s around its start."""
    for ins in inserts:
        window = published_doc.in_range(max(0.0, ins.start - window_s), ins.start + window_s)
        if not window:
            continue
        ins.trigger_text = " ".join(w.display or w.word for w in window)
        hit = extract_trigger(window)
        if hit is None:
            continue
        i, token, kind = hit
        ins.trigger_word = token
        ins.trigger_kind = kind
        ins.trigger_time = window[i].start
        # Identity, not equality: two words with the same spelling are the
        # same dataclass value, and only one of them is this trigger.
        ins.trigger_word_index = next(
            (j for j, w in enumerate(published_doc.words) if w is window[i]), -1)
    return list(inserts)


# -------- step 7: no raw take available --------------------------------------

AGENTIC_PROMPT = """\
This is a published Darija news reel. List every edit you can see and hear:
* cuts: moments where the audio or the facecam jumps because material was removed
* inserts: moments where a screenshot, chart, post or other visual takes the frame

Give each one a start and an end in seconds from the beginning of THIS file.
"""

AGENTIC_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "cuts": {"type": "array", "items": {
            "type": "object",
            "properties": {"start": {"type": "number"}, "end": {"type": "number"},
                           "kind": {"type": "string"}, "why": {"type": "string"}},
            "required": ["start", "end"]}},
        "inserts": {"type": "array", "items": {
            "type": "object",
            "properties": {"start": {"type": "number"}, "end": {"type": "number"},
                           "type": {"type": "string"}, "shows": {"type": "string"}},
            "required": ["start", "end"]}},
    },
    "required": ["cuts", "inserts"],
}


def analyse_published(video: str | Path, pair_name: str, *,
                      llm: gemini_client.LLM | None = None
                      ) -> tuple[list[CutObservation], list[InsertObservation]]:
    """Fallback for a published-only pair: ask the model what it sees.

    These ARE model timestamps, and they are the one exception in the project:
    they describe a video that already shipped, they only ever become aggregate
    statistics in the profile, and scene detection overrides them whenever both
    exist. Nothing here reaches an EDL.
    """
    llm = llm or gemini_client.default_llm()
    llm.require()
    answer = llm.generate(AGENTIC_PROMPT, schema=AGENTIC_SCHEMA, files=[str(video)])
    if isinstance(answer, str):
        answer = json.loads(answer)
    answer = answer or {}

    cuts = [CutObservation(pair=pair_name, span=Span(0, 0),
                           start=float(c["start"]), end=float(c["end"]),
                           text="", before="", after="",
                           klass=(c.get("kind") if c.get("kind") in CUT_CLASSES else "other"),
                           why=c.get("why", "reported by video understanding"))
            for c in answer.get("cuts", []) if c.get("end", 0) > c.get("start", 0)]
    inserts = [InsertObservation(pair=pair_name, start=float(i["start"]), end=float(i["end"]),
                                 label={"type": i.get("type", "other"),
                                        "shows": i.get("shows", "")},
                                 detector="gemini_video")
               for i in answer.get("inserts", []) if i.get("end", 0) > i.get("start", 0)]
    return cuts, inserts


# -------- the profile ---------------------------------------------------------


def _median(values: Iterable[float], default: float = 0.0) -> float:
    vals = [v for v in values if v is not None]
    return round(statistics.median(vals), 3) if vals else default


def _display_of(token_norm: str, seen: dict[str, dict[str, int]]) -> str:
    """The spelling of a normalised token that appeared most often."""
    forms = seen.get(token_norm, {})
    return max(forms, key=forms.get) if forms else token_norm


def build_profile(observations: Sequence[PairObservation], *,
                  examples_name: str = "few_shot_examples.md") -> dict:
    """Aggregate every pair into the profile the planner reads."""
    seed = _seed_fillers()
    layout_cfg = _layout_defaults()
    cuts = [c for o in observations for c in o.cuts]
    inserts = [i for o in observations for i in o.inserts]

    published_minutes = sum(o.published_duration for o in observations) / 60.0
    gaps = [g for o in observations for g in o.kept_gaps_ms if g > 0]
    # One long pause is a structural break (a beat before a punchline, or a
    # breath between chapters), not a licence to leave every pause that long.
    beats = [g for g in gaps if g <= LONG_PAUSE_S * 1000.0]

    lead_ins = [i.lead_in_s for i in inserts if i.lead_in_s is not None and i.lead_in_s >= 0]
    pip_scales = [i.pip_scale for i in inserts if i.pip_scale]
    layouts = _mode([i.label.get("layout") for i in inserts if i.label.get("layout")])
    corners = _mode([i.label.get("pip_corner") for i in inserts
                     if i.label.get("pip_corner") not in (None, "none")])

    profile = {
        "cut_ratio": _median([o.cut_ratio for o in observations], 0.0),
        "median_kept_gap_ms": _median(beats, 0.0),
        "max_kept_gap_ms": round(max(beats), 1) if beats else 0.0,
        "filler_policy": {
            "remove": _learned_fillers(cuts, seed),
            "keep": _surviving_keepers(observations, seed),
        },
        "retake_policy": _retake_policy(cuts),
        "intro_trim_s": _median([c.duration for c in cuts if c.at_head], 0.0),
        "outro_trim_s": _median([c.duration for c in cuts if c.at_tail], 0.0),
        "inserts": {
            "per_minute": round(len(inserts) / published_minutes, 2) if published_minutes else 0.0,
            "median_duration_s": _median([i.duration for i in inserts],
                                         layout_cfg["default_duration_s"]),
            "lead_in_s": _median(lead_ins, layout_cfg["lead_in_s"]),
            "layout": _map_layout(layouts, layout_cfg["layout"]),
            "pip_corner": corners or layout_cfg["pip_corner"],
            "pip_scale": _median(pip_scales, layout_cfg["pip_scale"]),
            "triggers": _trigger_order(inserts),
        },
        "examples": examples_name,
        "meta": {
            "pairs": [o.name for o in observations],
            "pairs_with_raw": sum(1 for o in observations if o.has_raw),
            "cuts_observed": len(cuts),
            "inserts_observed": len(inserts),
            "cut_classes": _counts([c.klass for c in cuts]),
            "insert_detectors": _counts([i.detector for i in inserts]),
        },
    }
    return profile


def _seed_fillers() -> dict:
    try:
        return config.load("fillers_darija")
    except (FileNotFoundError, OSError):  # pragma: no cover - config ships with the repo
        return {}


def _layout_defaults() -> dict:
    try:
        cfg = config.load("layout")
    except (FileNotFoundError, OSError):  # pragma: no cover
        cfg = {}
    layout = config.get(cfg, "inserts.default_layout", "fit_card")
    return {
        "layout": layout,
        "lead_in_s": float(config.get(cfg, "inserts.timing.lead_in_s", 0.3)),
        "default_duration_s": float(config.get(cfg, "inserts.timing.default_duration_s", 4.5)),
        "pip_corner": config.get(cfg, f"inserts.layouts.{layout}.facecam_corner",
                                 "bottom_right"),
        "pip_scale": float(config.get(cfg, f"inserts.layouts.{layout}.facecam_scale", 0.30)),
    }


def _learned_fillers(cuts: Sequence[CutObservation], seed: dict, min_hits: int = 2) -> list[str]:
    """Words the editor actually deletes, on top of the seed list."""
    counts: dict[str, int] = {}
    forms: dict[str, dict[str, int]] = {}
    for c in cuts:
        if c.klass != "filler":
            continue
        for token in textnorm.tokenize(c.text):
            norm = textnorm.normalize_token(token)
            if not norm:
                continue
            counts[norm] = counts.get(norm, 0) + 1
            spellings = forms.setdefault(norm, {})
            spellings[token] = spellings.get(token, 0) + 1
    learned = [_display_of(n, forms) for n, k in sorted(counts.items(), key=lambda kv: -kv[1])
               if k >= min_hits]
    seeded = [t for group in (seed.get("remove") or {}).values() for t in (group or [])]
    return _dedup(learned + seeded)


def _surviving_keepers(observations: Sequence[PairObservation], seed: dict) -> list[str]:
    """Seed keepers that really do survive into published episodes."""
    published = {t for o in observations for t in o.published_tokens}
    keepers = [t for group in (seed.get("keep") or {}).values() for t in (group or [])]
    kept = [t for t in keepers
            if published & set(textnorm.normalize_tokens(t))]
    return _dedup(kept or keepers)


def _retake_policy(cuts: Sequence[CutObservation]) -> str:
    """Whether the surviving take is the later one or the earlier one."""
    later = sum(1 for c in cuts if c.klass == "retake"
                and _repeats_in(c.text, c.after))
    earlier = sum(1 for c in cuts if c.klass == "retake"
                  and _repeats_in(c.text, c.before))
    if later == earlier == 0:
        return "keep_last_complete"
    return "keep_last_complete" if later >= earlier else "keep_first_complete"


def _repeats_in(text: str, context: str) -> bool:
    removed = set(textnorm.normalize_tokens(text))
    around = set(textnorm.normalize_tokens(context))
    return bool(removed) and len(removed & around) / len(removed) >= 0.5


def _map_layout(observed: str | None, default: str) -> str:
    return {"fullframe": "fullframe_pip", "fullframe_pip": "fullframe_pip",
            "split": "split", "card": "fit_card"}.get(observed or "", default)


def _trigger_order(inserts: Sequence[InsertObservation]) -> list[str]:
    names = {"entity": "first_mention_company", "number": "number", "quote": "quote",
             "date": "date", "url": "url"}
    counts = _counts([names.get(i.trigger_kind, i.trigger_kind)
                      for i in inserts if i.trigger_kind])
    ordered = [k for k, _v in sorted(counts.items(), key=lambda kv: -kv[1])]
    for i in inserts:
        if i.label.get("type") == "article_headline" and "headline" not in ordered:
            ordered.append("headline")
    return ordered


def _counts(values: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        if v:
            out[v] = out.get(v, 0) + 1
    return out


def _mode(values: Iterable[str]) -> str | None:
    counts = _counts(values)
    return max(counts, key=counts.get) if counts else None


def _dedup(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for i in items:
        key = textnorm.normalize_token(i) or i
        if key not in seen:
            seen.add(key)
            out.append(i)
    return out


# -------- the examples file ---------------------------------------------------


def write_examples(observations: Sequence[PairObservation], dest: str | Path, *,
                   max_cuts: int = 8, max_inserts: int = 5) -> Path:
    """Before/after excerpts and real insert triggers, for the planner prompt.

    This file is injected into every plan prompt, so it stays short and concrete:
    a handful of real holes with the speech either side, and a handful of real
    visuals with the words that summoned them. No rules, no prose -- the rules
    are in the profile.
    """
    cuts = _sample_cuts([c for o in observations for c in o.cuts], max_cuts)
    inserts = [i for o in observations for i in o.inserts if i.trigger_word][:max_inserts]

    lines = [f"# Zeta editing examples ({len(observations)} episode(s))", ""]
    lines += ["## Cuts (what the editor removed, in context)", ""]
    for n, c in enumerate(cuts, 1):
        lines += [
            f"### {n}. {c.klass} - {c.pair} at {_clock(c.start)} ({c.duration:.1f}s)",
            f"- before: {_short(c.before)}",
            f"- REMOVED: {_short(c.text)}",
            f"- after: {_short(c.after)}",
        ]
        if c.why:
            lines.append(f"- why: {_short(c.why, 160)}")
        lines.append("")
    if not cuts:
        lines += ["_No raw takes were available: cut examples could not be extracted._", ""]

    lines += ["## Insert triggers (what put a visual on screen)", ""]
    for n, i in enumerate(inserts, 1):
        label = i.label or {}
        lines += [
            f"### {n}. {i.trigger_word} ({i.trigger_kind}) - {i.pair} at {_clock(i.start)}",
            f"- spoken: {_short(i.trigger_text)}",
            f"- frame: {label.get('type', 'unknown')} - {_short(label.get('shows', ''), 160)}",
            f"- source: {label.get('likely_source', 'unknown')}; "
            f"layout {label.get('layout', 'unknown')}; held {i.duration:.1f}s"
            + (f"; visual is up {i.lead_in_s:.2f}s before the word"
               if i.lead_in_s is not None else ""),
            "",
        ]
    if not inserts:
        lines += ["_No inserts with a resolved trigger were found._", ""]

    p = Path(dest)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def _sample_cuts(cuts: Sequence[CutObservation], limit: int) -> list[CutObservation]:
    """Spread the examples over the classes rather than over the timeline."""
    by_class: dict[str, list[CutObservation]] = {}
    for c in cuts:
        if c.text.strip():
            by_class.setdefault(c.klass, []).append(c)
    out: list[CutObservation] = []
    while len(out) < limit and any(by_class.values()):
        for klass in list(by_class):
            if by_class[klass] and len(out) < limit:
                out.append(by_class[klass].pop(0))
            if not by_class[klass]:
                del by_class[klass]
    return out


def _short(text: str, limit: int = 140) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _clock(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"


# -------- acceptance metric ---------------------------------------------------


def cut_iou(predicted: Sequence[tuple[float, float]],
            truth: Sequence[tuple[float, float]]) -> dict:
    """Phase-3 acceptance: how well recovered cut spans match the real ones.

    Interval IoU over the union of both sets, plus per-span recall at 50 %
    overlap, which is what "insert detection recall" means in the spec table.
    """
    inter = _union_length(_intersections(predicted, truth))
    union = _union_length(list(predicted) + list(truth))
    hits = sum(1 for t in truth
               if _union_length(_intersections([t], predicted)) >= 0.5 * (t[1] - t[0]))
    return {
        "iou": round(inter / union, 4) if union else 0.0,
        "recall": round(hits / len(truth), 4) if truth else 0.0,
        "predicted": len(list(predicted)),
        "truth": len(list(truth)),
    }


def _intersections(a: Sequence[tuple[float, float]],
                   b: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    out = []
    for s1, e1 in a:
        for s2, e2 in b:
            lo, hi = max(s1, s2), min(e1, e2)
            if hi > lo:
                out.append((lo, hi))
    return out


def _union_length(spans: Sequence[tuple[float, float]]) -> float:
    total, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(spans):
        if cur_e is None or s > cur_e:
            total += (cur_e - cur_s) if cur_e is not None else 0.0
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


# -------- orchestration -------------------------------------------------------


def learn_pair(pair: Pair, *, edit_paths: paths.EditPaths | None = None,
               llm: gemini_client.LLM | None = None, profile: dict | None = None,
               visuals: bool = True,
               transcribe: Callable[[Path], WordsDoc] | None = None) -> PairObservation:
    """Everything one `(raw, published)` pair can teach."""
    published_doc = words_for(pair.published, edit_paths=edit_paths, transcribe=transcribe)
    obs = PairObservation(name=pair.name, has_raw=pair.has_raw)
    obs.published_duration = _doc_duration(published_doc)
    obs.kept_gaps_ms = [g * 1000.0 for g in published_doc.gaps()[1:]]
    obs.published_tokens = [w.norm for w in published_doc.words if w.norm]

    if pair.has_raw:
        raw_doc = words_for(pair.raw, edit_paths=edit_paths, transcribe=transcribe)
        obs.raw_duration = _doc_duration(raw_doc)
        obs.cuts, obs.word_cut_ratio = diff_pair(raw_doc, published_doc, pair.name)
        obs.cuts = classify_cuts(obs.cuts, raw_doc, llm=llm, profile=profile)

    if visuals:
        frames = (edit_paths.learn if edit_paths else pair.published.parent / "edit" / "learn")
        try:
            obs.inserts = detect_inserts(pair.published, pair.name, frames)
        except MissingExtra:
            obs.inserts = []
        if obs.inserts:
            obs.inserts = label_inserts(obs.inserts, llm=llm)
        elif not pair.has_raw:
            # Spec step 7: with no raw take and no scene detection there is
            # nothing left but asking the model to watch the episode.
            try:
                fallback_cuts, obs.inserts = analyse_published(pair.published, pair.name,
                                                              llm=llm)
                obs.cuts = obs.cuts or fallback_cuts
            except (gemini_client.LLMError, OSError):
                obs.inserts = []
        obs.inserts = map_insert_triggers(obs.inserts, published_doc)
    return obs


def learn(pairs: Sequence[Pair] | str | Path, out: str | Path | None = None, *,
          edit_paths: paths.EditPaths | None = None,
          llm: gemini_client.LLM | None = None, visuals: bool = True,
          transcribe: Callable[[Path], WordsDoc] | None = None,
          observations_path: str | Path | None = None) -> dict:
    """Learn from every pair; write `style_profile.json` and the examples file.

    `pairs` is either `pairs.csv` or already-loaded `Pair`s. Returns the profile
    itself, because that is what every caller actually wants; the per-pair
    detail is large and goes to `observations_path` when one is asked for.
    """
    if isinstance(pairs, (str, Path)):
        pairs = load_pairs(pairs)
    observations = [learn_pair(p, edit_paths=edit_paths, llm=llm, visuals=visuals,
                               transcribe=transcribe) for p in pairs]

    out_path = Path(out) if out else _default_profile_path(edit_paths, pairs)
    examples = write_examples(observations, out_path.parent / "few_shot_examples.md")
    profile = build_profile(observations, examples_name=examples.name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(profile, ensure_ascii=False, indent=1), encoding="utf-8")
    profile["meta"]["written_to"] = str(out_path)

    if observations_path:
        Path(observations_path).write_text(
            json.dumps([o.to_dict() for o in observations], ensure_ascii=False, indent=1),
            encoding="utf-8")
    return profile


def _default_profile_path(edit_paths: paths.EditPaths | None,
                          pairs: Sequence[Pair]) -> Path:
    """Hard rule 11: under `<videos_dir>/edit/`, next to the episodes it learned from."""
    if edit_paths is not None:
        return edit_paths.edit / "style_profile.json"
    root = pairs[0].published.parent if pairs else Path.cwd()
    return paths.EditPaths.for_videos_dir(root).edit / "style_profile.json"


def _doc_duration(doc: WordsDoc) -> float:
    timed = doc.timed_words()
    end = float(doc.source.get("duration_s") or 0.0)
    return max(end, timed[-1].end if timed else 0.0)


# -------- cli -----------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="learn_style",
        description="Stage 3: learn the editing style from past (raw, published) pairs.")
    ap.add_argument("--pairs", required=True, help="pairs.csv of `raw,published`")
    ap.add_argument("--out", help="style_profile.json (default: <videos-dir>/edit/)")
    ap.add_argument("--videos-dir", help="output root (default: the pairs.csv folder)")
    ap.add_argument("--no-visuals", action="store_true",
                    help="skip scene detection and vision labelling")
    ap.add_argument("--observations", help="also write the per-pair detail as JSON")
    args = ap.parse_args(argv)

    pairs = load_pairs(args.pairs)
    root = args.videos_dir or Path(args.pairs).resolve().parent
    edit_paths = paths.EditPaths.for_videos_dir(root).ensure()
    edit_paths.learn.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else edit_paths.edit / "style_profile.json"

    profile = learn(pairs, out, edit_paths=edit_paths, visuals=not args.no_visuals,
                    observations_path=args.observations)
    print(f"learned from {len(pairs)} pair(s): cut_ratio {profile['cut_ratio']}, "
          f"{profile['meta']['cuts_observed']} cuts, "
          f"{profile['meta']['inserts_observed']} inserts")
    print(f"wrote {out}")
    print(f"wrote {out.parent / profile['examples']}")
    if args.observations:
        print(f"wrote {args.observations}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
