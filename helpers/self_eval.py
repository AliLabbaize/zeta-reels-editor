"""Stage 8: look at what was actually rendered, not at what was planned.

Everything upstream reasons about an EDL. This module reasons about the file
that came out of `render.py`, because that is the only artifact that can be
wrong in ways the plan cannot express: a concat that lost a frame, a fade that
did not take, an overlay that outlives its cut, a caption that lands under an
image.

Three kinds of check, deliberately separated:

  * arithmetic  -- `ffprobe` duration vs `EDL.total_duration_s`, overlay windows
                   vs the cuts and the end of the timeline, caption cues vs
                   overlay windows. Deterministic, no model, no ffmpeg needed
                   for the pure functions.
  * signal      -- a waveform spike inside the 30 ms fade at a boundary (hard
                   rule 3), computed from PCM extracted with ffmpeg.
  * image       -- visual jump at a cut, overlay showing the wrong frames.
                   Only a model can judge these, so they run through
                   `helpers.gemini_client` and report "not checked" when no key
                   is present. A missing key degrades the report; it never
                   fails the run.

The fix-and-re-render loop is capped at 3 fixes (video-use's rule) and always
ends on an evaluation, never on an unverified fix. Whatever is left is flagged
in `edit/verify/self_eval.json` and in the report, because a loop that keeps
re-rendering on its own judgement is how an editor ends up with a file nobody
approved.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

try:  # package import
    from . import config as configs
    from . import gemini_client
    from .edl import EDL
    from .paths import EditPaths
except ImportError:  # `python helpers/self_eval.py ...`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from helpers import config as configs
    from helpers import gemini_client
    from helpers.edl import EDL
    from helpers.paths import EditPaths

MAX_PASSES = 3
BOUNDARY_WINDOW_S = 1.5      # timeline_view context either side of an event
DURATION_TOLERANCE_S = 0.20  # concat + keyframe rounding, not a real drift
FADE_MS = 30.0               # hard rule 3: every boundary carries this fade

TIMELINE_VIEW = Path(__file__).resolve().parent / "timeline_view.py"

SEVERITY_ORDER = {"info": 0, "warning": 1, "error": 2}


@dataclass
class Finding:
    check: str
    severity: str
    message: str
    t_output: float | None = None
    data: dict = field(default_factory=dict)
    image: str | None = None
    checked: bool = True

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"check": self.check, "severity": self.severity,
                             "message": self.message, "checked": self.checked}
        if self.t_output is not None:
            d["t_output"] = round(self.t_output, 3)
        if self.image:
            d["image"] = self.image
        if self.data:
            d["data"] = self.data
        return d


@dataclass
class CueWindow:
    """The minimum a caption cue needs to be checked against an overlay."""

    start: float
    end: float
    text: str = ""


def cue_windows(cues: Iterable[Any]) -> list[CueWindow]:
    """Accept `captions.Cue`, `CueWindow`, or a `{start, end, text}` dict."""
    out: list[CueWindow] = []
    for c in cues or []:
        if isinstance(c, CueWindow):
            out.append(c)
        elif isinstance(c, dict):
            out.append(CueWindow(float(c["start"]), float(c["end"]), c.get("text", "")))
        else:
            out.append(CueWindow(float(c.start), float(c.end), getattr(c, "text", "")))
    return out


def parse_ass(path: str | Path) -> list[CueWindow]:
    """Read back the .ass that was actually burned.

    The render used this file, so the self-eval reads it rather than rebuilding
    cues from words.json: a mismatch between the two is itself a bug worth
    seeing.
    """
    def t(v: str) -> float:
        h, m, s = v.strip().split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)

    out: list[CueWindow] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.startswith("Dialogue:"):
            continue
        fields = line.split(":", 1)[1].split(",", 9)
        if len(fields) < 10:
            continue
        text = fields[9]
        while "{" in text and "}" in text:  # drop override blocks
            a, b = text.index("{"), text.index("}")
            text = text[:a] + text[b + 1:]
        out.append(CueWindow(t(fields[1]), t(fields[2]), text.replace("\\N", " ").replace("\u200f", "").strip()))  # RLM: captions.render_ass
    return out


# -------- event points -------------------------------------------------------


def cut_boundaries(edl: EDL) -> list[float]:
    """Output times where two ranges are joined (the visible cuts)."""
    return edl.offsets()[1:]


def view_points(edl: EDL, window_s: float = BOUNDARY_WINDOW_S) -> list[tuple[str, float, float]]:
    """`(label, start, end)` windows to render with `timeline_view`."""
    total = edl.total_duration_s
    points: list[tuple[str, float]] = [(f"cut_{i:02d}", t) for i, t in enumerate(cut_boundaries(edl))]
    for i, o in enumerate(edl.overlays):
        points.append((f"insert_{i:02d}_in", o.start_in_output))
        points.append((f"insert_{i:02d}_out", o.start_in_output + o.duration))
    out = []
    for label, t in points:
        a = max(0.0, t - window_s)
        b = min(total, t + window_s)
        if b > a:
            out.append((label, a, b))
    return out


# -------- arithmetic checks --------------------------------------------------


def probe_duration(video: str | Path) -> float | None:
    """None when ffprobe is unavailable or the file cannot be read."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
            capture_output=True, text=True, check=True)
        return float(out.stdout.strip())
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


def check_duration(edl: EDL, video: str | Path, *, tol_s: float = DURATION_TOLERANCE_S,
                   probed: float | None = None) -> list[Finding]:
    expected = edl.total_duration_s
    actual = probed if probed is not None else probe_duration(video)
    if actual is None:
        return [Finding("duration", "warning", "ffprobe unavailable: output duration not checked",
                        checked=False, data={"expected_s": round(expected, 3)})]
    delta = actual - expected
    if abs(delta) > tol_s:
        return [Finding("duration", "error",
                        f"output is {actual:.3f}s but the EDL totals {expected:.3f}s "
                        f"({delta:+.3f}s): a segment was lost or double-counted",
                        data={"expected_s": round(expected, 3), "actual_s": round(actual, 3),
                              "delta_s": round(delta, 3), "tolerance_s": tol_s})]
    return [Finding("duration", "info", f"duration matches the EDL ({actual:.3f}s)",
                    data={"expected_s": round(expected, 3), "actual_s": round(actual, 3),
                          "delta_s": round(delta, 3)})]


def check_overlay_windows(edl: EDL) -> list[Finding]:
    """An insert must live inside the cut it was placed in."""
    total = edl.total_duration_s
    offsets = edl.offsets()
    boundaries = offsets[1:]
    findings: list[Finding] = []
    for i, o in enumerate(edl.overlays):
        end = o.start_in_output + o.duration
        if end > total + 1e-3:
            findings.append(Finding(
                "overlay_past_end", "error",
                f"insert {i} runs to {end:.2f}s but the cut ends at {total:.2f}s",
                t_output=o.start_in_output,
                data={"index": i, "file": o.file, "end_s": round(end, 3),
                      "total_s": round(total, 3)}))
            continue
        crossed = [b for b in boundaries if o.start_in_output < b < end]
        if crossed:
            findings.append(Finding(
                "overlay_crosses_cut", "warning",
                f"insert {i} spans the cut at {crossed[0]:.2f}s: the image stays up "
                f"across a hard cut",
                t_output=o.start_in_output,
                data={"index": i, "file": o.file, "boundaries": [round(b, 3) for b in crossed]}))
    return findings


def _covers_caption_band(layout_name: str | None, layout_cfg: dict | None = None) -> bool:
    """Does this insert layout paint over the caption band at the bottom?"""
    cfg = layout_cfg if layout_cfg is not None else configs.load("layout")
    block = ((cfg.get("inserts") or {}).get("layouts") or {}).get(layout_name or "")
    if not block:
        return True  # unknown layout: assume the worst and make a human look
    if block.get("allow_crop"):
        return True  # full-frame: the screenshot reaches the bottom edge
    if block.get("card_anchor") == "upper":
        return False
    if "screenshot_share" in block:
        return False  # split: the screenshot owns the top block only
    if "card_bottom" in block:
        return False  # float: a see-through frame whose card ends above the captions
    return True


def check_captions_under_overlays(cues: Iterable[Any], edl: EDL, *,
                                  layout_cfg: dict | None = None) -> list[Finding]:
    """Hard rule 1: captions are burned last, so nothing may cover them.

    A cue whose whole window sits inside a full-frame insert is the case that
    goes wrong when the filter order slips -- the caption is then painted
    before the image and disappears. Partial overlaps are reported as candidates
    for the image check rather than as failures.
    """
    windows = cue_windows(cues)
    findings: list[Finding] = []
    if edl.overlays and not edl.subtitles:
        findings.append(Finding(
            "captions_missing", "error",
            "the EDL carries overlays but no subtitles file: captions were never "
            "burned (hard rule 1)"))
    for i, o in enumerate(edl.overlays):
        o_start, o_end = o.start_in_output, o.start_in_output + o.duration
        covers = _covers_caption_band((o.meta or {}).get("layout"), layout_cfg)
        for c in windows:
            if c.end <= o_start or c.start >= o_end:
                continue
            inside = c.start >= o_start - 1e-6 and c.end <= o_end + 1e-6
            if inside and covers:
                findings.append(Finding(
                    "caption_under_overlay", "error",
                    f"caption {c.text[:40]!r} at {c.start:.2f}-{c.end:.2f}s is fully "
                    f"inside full-frame insert {i}: it is hidden unless the .ass is "
                    f"burned after the overlay (hard rule 1)",
                    t_output=c.start,
                    data={"overlay": i, "layout": (o.meta or {}).get("layout"),
                          "cue": {"start": round(c.start, 3), "end": round(c.end, 3),
                                  "text": c.text}}))
            else:
                findings.append(Finding(
                    "caption_overlaps_overlay", "info",
                    f"caption at {c.start:.2f}s overlaps insert {i}; check legibility "
                    f"on the rendered frame",
                    t_output=c.start,
                    data={"overlay": i, "layout": (o.meta or {}).get("layout"),
                          "fully_inside": inside, "covers_caption_band": covers}))
    return findings


# -------- audio ---------------------------------------------------------------


def extract_pcm(video: str | Path, start: float = 0.0, duration: float | None = None,
                sample_rate: int = 16000):
    """Mono float PCM around a region. `(samples, sample_rate)` or `(None, sr)`."""
    import numpy as np  # lazy: the arithmetic checks must import without it

    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "a.wav"
        cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-i", str(video)]
        if duration is not None:
            cmd += ["-t", f"{duration:.3f}"]
        cmd += ["-vn", "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(wav)]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.CalledProcessError):
            return None, sample_rate
        if not wav.exists() or wav.stat().st_size == 0:
            return None, sample_rate
        with wave.open(str(wav), "rb") as w:
            frames = w.readframes(w.getnframes())
            sr = w.getframerate()
        return np.frombuffer(frames, dtype=np.int16).astype("float32") / 32768.0, sr


def find_pops(samples, sample_rate: int, boundaries_s: Sequence[float], *,
              offset_s: float = 0.0, fade_ms: float = FADE_MS,
              ratio: float = 2.0, floor: float = 0.05) -> list[dict]:
    """Sample-level discontinuities inside the fade window of each boundary.

    With a 30 ms fade the waveform has to reach the join smoothly, so the
    largest sample-to-sample step there should be no bigger than the material
    around it. A step many times the local median is a click -- the fade was
    not applied, or the concat spliced mid-period. `floor` keeps quiet noise
    from producing huge ratios on silence.
    """
    import numpy as np

    if samples is None or len(samples) < 4:
        return []
    diff = np.abs(np.diff(samples))
    local = float(np.median(diff)) if diff.size else 0.0
    half = max(1, int(sample_rate * fade_ms / 1000.0))
    pops: list[dict] = []
    for t in boundaries_s:
        i = int(round((t - offset_s) * sample_rate))
        a, b = max(0, i - half), min(diff.size, i + half)
        if b <= a:
            continue
        window = diff[a:b]
        peak = float(window.max())
        # "The material around it": 250 ms either side, outside the fade. The
        # file-wide median is mostly silence, so any join near speech read as a
        # pop (6 false errors on a real take, every step smaller than the speech).
        span = int(sample_rate * 0.25)
        near = np.concatenate([diff[max(0, a - span):a], diff[b:b + span]])
        local = float(near.max()) if near.size else local
        r = peak / (local + 1e-6)
        if peak >= floor and r >= ratio:
            pops.append({"t_output": round(float(t), 3), "peak": round(peak, 4),
                         "local_median": round(local, 6), "ratio": round(r, 1)})
    return pops


def check_audio_pops(video: str | Path, edl: EDL, *, samples=None, sample_rate: int = 16000,
                     fade_ms: float = FADE_MS) -> list[Finding]:
    boundaries = cut_boundaries(edl)
    if not boundaries:
        return []
    if samples is None:
        samples, sample_rate = extract_pcm(video, sample_rate=sample_rate)
    if samples is None:
        return [Finding("audio_pop", "warning",
                        "ffmpeg unavailable: boundary audio not checked",
                        checked=False)]
    pops = find_pops(samples, sample_rate, boundaries, fade_ms=fade_ms)
    if not pops:
        return [Finding("audio_pop", "info",
                        f"no waveform spike at {len(boundaries)} boundary/ies "
                        f"(fade {fade_ms:.0f} ms)")]
    return [Finding("audio_pop", "error",
                    f"audio pop at {p['t_output']:.2f}s (step {p['ratio']:.0f}x the local "
                    f"median inside the {fade_ms:.0f} ms fade): hard rule 3 did not hold",
                    t_output=p["t_output"], data=p)
            for p in pops]


# -------- timeline_view ------------------------------------------------------


def run_timeline_view(video: str | Path, start: float, end: float, out_png: Path, *,
                      transcript: str | Path | None = None, n_frames: int = 10) -> Path | None:
    """Subprocess, never an import: timeline_view is vendored and pulls PIL/numpy."""
    cmd = [sys.executable, str(TIMELINE_VIEW), str(video), f"{start:.3f}", f"{end:.3f}",
           "-o", str(out_png), "--n-frames", str(n_frames)]
    if transcript:
        cmd += ["--transcript", str(transcript)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out_png if out_png.exists() else None


def timeline_transcript(doc, edl: EDL, out_path: str | Path) -> Path | None:
    """Write a transcript `timeline_view.py` can label the filmstrip with.

    Two translations are needed and both matter. The vendored tool reads
    ElevenLabs Scribe's shape (`text`, `type`) while words.json carries
    `word`/`display`; and it is pointed at the RENDERED file, whose timeline is
    the output timeline, while words.json is on the source timeline. Labels at
    source times over a cut render would be worse than no labels at all - they
    would send whoever is reviewing a boundary to the wrong place.
    """
    words = getattr(doc, "words", None)
    if not words:
        return None
    out: list[dict] = []
    for w in words:
        if not getattr(w, "timed", False):
            continue
        source = doc.source.get("name") if isinstance(getattr(doc, "source", None), dict) else None
        source = source or (edl.ranges[0].source if edl.ranges else None)
        t = edl.to_output_time(source, w.start)
        if t is None:                       # the word was cut; it has no place here
            continue
        end = edl.to_output_time(source, min(w.end, w.end - 1e-6))
        out.append({"text": w.display or w.word, "type": "word",
                    "start": round(t, 3),
                    "end": round(end if end is not None else t + w.duration, 3)})
    if not out:
        return None
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"words": out}, ensure_ascii=False), encoding="utf-8")
    return path


def render_views(video: str | Path, edl: EDL, out_dir: Path, *,
                 transcript: str | Path | None = None,
                 window_s: float = BOUNDARY_WINDOW_S) -> tuple[dict[str, Path], list[Finding]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    views: dict[str, Path] = {}
    findings: list[Finding] = []
    for label, a, b in view_points(edl, window_s):
        png = run_timeline_view(video, a, b, out_dir / f"{label}.png", transcript=transcript)
        if png is None:
            findings.append(Finding("timeline_view", "warning",
                                    f"could not render the {label} view ({a:.2f}-{b:.2f}s)",
                                    t_output=a, checked=False, data={"label": label}))
            continue
        views[label] = png
    return views, findings


# -------- image checks (model) ------------------------------------------------


_JUMP_SCHEMA = {"type": "object",
                "properties": {"jump": {"type": "boolean"}, "evidence": {"type": "string"}},
                "required": ["jump", "evidence"]}
_INSERT_SCHEMA = {"type": "object",
                  "properties": {"shows_claim": {"type": "boolean"},
                                 "caption_visible": {"type": "boolean"},
                                 "evidence": {"type": "string"}},
                  "required": ["shows_claim", "evidence"]}


def _ask_vision(llm, png: Path, prompt: str, schema: dict) -> dict | None:
    """None means "not checked": no key, no model, or a malformed answer."""
    if llm is None or not getattr(llm, "available", False) or not Path(png).exists():
        return None
    try:
        out = llm.vision_json(prompt, png, schema)
    except Exception:  # LLMUnavailable, transport, bad JSON: never fatal here
        return None
    return out if isinstance(out, dict) else None


def check_visual_jumps(views: dict[str, Path], edl: EDL, llm=None) -> list[Finding]:
    boundaries = cut_boundaries(edl)
    findings: list[Finding] = []
    for i, t in enumerate(boundaries):
        png = views.get(f"cut_{i:02d}")
        if png is None:
            continue
        ans = _ask_vision(
            llm, png,
            "This filmstrip spans a cut in an edited talking-head video. Does the "
            "speaker's position, framing or background jump visibly between the "
            "frames before and after the middle of the strip? Answer JSON "
            '{"jump": bool, "evidence": str}.',
            _JUMP_SCHEMA)
        if ans is None:
            findings.append(Finding("visual_jump", "info",
                                    f"cut at {t:.2f}s: visual continuity not checked "
                                    f"(no vision model available)",
                                    t_output=t, image=str(png), checked=False))
        elif ans.get("jump"):
            findings.append(Finding("visual_jump", "warning",
                                    f"visible jump at the cut at {t:.2f}s: "
                                    f"{ans.get('evidence', '')}",
                                    t_output=t, image=str(png), data=ans))
    return findings


def check_insert_frames(views: dict[str, Path], edl: EDL, llm=None) -> list[Finding]:
    """Does the insert really show its claim, and is the caption still readable?"""
    findings: list[Finding] = []
    for i, o in enumerate(edl.overlays):
        png = views.get(f"insert_{i:02d}_in")
        if png is None:
            continue
        claim = (o.meta or {}).get("claim", "")
        ans = _ask_vision(
            llm, png,
            "This filmstrip spans the start of a screenshot insert in a news reel. "
            f"The insert is a headline card for this story: {claim!r}"
            + (f" (the video's main story: {(o.meta or {}).get('story')!r})"
               if (o.meta or {}).get("story") else "")
            + ". shows_claim is true when a readable headline about that same story is "
            "on screen, whatever its wording. Answer JSON "
            '{"shows_claim": bool, "caption_visible": bool, "evidence": str}.',
            _INSERT_SCHEMA)
        if ans is None:
            findings.append(Finding("insert_frames", "info",
                                    f"insert {i} at {o.start_in_output:.2f}s: frames not "
                                    f"checked (no vision model available)",
                                    t_output=o.start_in_output, image=str(png), checked=False))
            continue
        if not ans.get("shows_claim", True):
            findings.append(Finding("insert_wrong_frames", "error",
                                    f"insert {i} at {o.start_in_output:.2f}s does not show its "
                                    f"claim: {ans.get('evidence', '')}",
                                    t_output=o.start_in_output, image=str(png), data=ans))
        if ans.get("caption_visible") is False:
            findings.append(Finding("caption_under_overlay", "error",
                                    f"caption is not visible over insert {i} at "
                                    f"{o.start_in_output:.2f}s: {ans.get('evidence', '')} "
                                    f"(hard rule 1: captions are burned last)",
                                    t_output=o.start_in_output, image=str(png), data=ans))
    return findings


# -------- the pass -----------------------------------------------------------


@dataclass
class SelfEvalResult:
    passes: int
    findings: list[Finding] = field(default_factory=list)
    flagged: bool = False
    max_passes: int = MAX_PASSES
    fixes: int = 0

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    def to_dict(self) -> dict:
        return {
            "schema": "zeta.self_eval.v1",
            "passes": self.passes,
            "fixes": self.fixes,
            "max_passes": self.max_passes,
            "flagged": self.flagged,
            "counts": {s: sum(1 for f in self.findings if f.severity == s)
                       for s in ("error", "warning", "info")},
            "unchecked": [f.check for f in self.findings if not f.checked],
            "findings": [f.to_dict() for f in
                         sorted(self.findings,
                                key=lambda f: (-SEVERITY_ORDER[f.severity],
                                               f.t_output if f.t_output is not None else 0.0))],
        }


def evaluate(edl: EDL, video: str | Path, *, cues: Iterable[Any] | None = None,
             edit_paths: EditPaths | None = None, llm=None, transcript=None,
             views: dict[str, Path] | None = None, run_views: bool = True,
             layout_cfg: dict | None = None, probed_duration: float | None = None,
             samples=None, sample_rate: int = 16000) -> list[Finding]:
    """One evaluation pass over a rendered file."""
    findings: list[Finding] = []
    findings += check_duration(edl, video, probed=probed_duration)
    findings += check_overlay_windows(edl)

    if cues is None and edl.subtitles and edit_paths is not None:
        # render.py resolves a relative subtitles path against the edit dir;
        # EDLs in the wild also spell it from the videos dir. Try both rather
        # than report "no captions" for a file that was in fact burned.
        ass = Path(edl.subtitles)
        candidates = [ass] if ass.is_absolute() else [edit_paths.edit / ass,
                                                      edit_paths.videos_dir / ass]
        for c in candidates:
            if c.exists():
                cues = parse_ass(c)
                break
    findings += check_captions_under_overlays(cues or [], edl, layout_cfg=layout_cfg)
    findings += check_audio_pops(video, edl, samples=samples, sample_rate=sample_rate)

    if views is None and run_views and edit_paths is not None:
        # A WordsDoc has to be translated and remapped before the vendored tool
        # can read it; a path is passed straight through.
        tl = transcript
        if transcript is not None and not isinstance(transcript, (str, Path)):
            tl = timeline_transcript(transcript, edl,
                                     edit_paths.verify / "_timeline_words.json")
        views, view_findings = render_views(video, edl, edit_paths.verify,
                                            transcript=tl)
        findings += view_findings
    views = views or {}

    if llm is None:
        llm = gemini_client.LLM()
    findings += check_visual_jumps(views, edl, llm)
    findings += check_insert_frames(views, edl, llm)
    return findings


def run_loop(evaluate_pass: Callable[[int], list[Finding]],
             fix: Callable[[list[Finding], int], bool] | None = None, *,
             max_passes: int = MAX_PASSES) -> SelfEvalResult:
    """Evaluate, fix, re-render -- at most `max_passes` fixes, then flag.

    `fix` returns True when it changed something and re-rendered, so the next
    evaluation looks at a different file; the loop always ends on an evaluation,
    never on an unverified fix. Anything still failing when the passes run out
    is flagged rather than fixed again: an unbounded loop is how a render ends
    up silently different from the plan a human approved.
    """
    findings: list[Finding] = []
    passes = 0
    fixes = 0
    while True:
        passes += 1
        findings = list(evaluate_pass(passes - 1))
        errors = [f for f in findings if f.severity == "error"]
        if not errors or fix is None or fixes >= max(0, max_passes):
            break
        if not fix(errors, fixes):
            break
        fixes += 1
    remaining = [f for f in findings if f.severity == "error"]
    for f in remaining:
        f.data.setdefault("unresolved_after_passes", fixes)
    return SelfEvalResult(passes=passes, findings=findings, flagged=bool(remaining),
                          max_passes=max_passes, fixes=fixes)


def write_self_eval(result: SelfEvalResult, edit_paths: EditPaths) -> Path:
    edit_paths.verify.mkdir(parents=True, exist_ok=True)
    p = edit_paths.verify / "self_eval.json"
    p.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 8: evaluate a rendered cut against its EDL")
    ap.add_argument("edl", type=Path)
    ap.add_argument("--video", type=Path, default=None, help="the rendered file (default edit/final.mp4)")
    ap.add_argument("--videos-dir", type=Path, default=None)
    ap.add_argument("--captions", type=Path, default=None, help="the .ass that was burned")
    ap.add_argument("--transcript", type=Path, default=None, help="words.json for the timeline views")
    ap.add_argument("--no-views", action="store_true", help="skip the timeline_view PNGs")
    ap.add_argument("--max-passes", type=int, default=MAX_PASSES)
    args = ap.parse_args(argv)

    edl = EDL.load(args.edl)
    paths = EditPaths.for_videos_dir(args.videos_dir or args.edl.resolve().parent.parent)
    video = args.video or paths.final
    cues = parse_ass(args.captions) if args.captions else None

    result = run_loop(
        lambda i: evaluate(edl, video, cues=cues, edit_paths=paths,
                           transcript=args.transcript, run_views=not args.no_views),
        max_passes=args.max_passes)
    out = write_self_eval(result, paths)
    counts = result.to_dict()["counts"]
    print(f"{counts['error']} error(s), {counts['warning']} warning(s) in {result.passes} pass(es)")
    for f in result.findings:
        if f.severity != "info":
            print(f"  [{f.severity}] {f.check}: {f.message}")
    print(f"  {out}")
    return 1 if result.flagged else 0


if __name__ == "__main__":
    raise SystemExit(main())
