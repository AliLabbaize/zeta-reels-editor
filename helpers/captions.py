"""Burned captions on the OUTPUT timeline (hard rules 1 and 5).

Rule 5: a caption time is an OUTPUT time. Every word goes through
`EDL.to_output_time`, the single place that knows
`output = source - range.start + range.offset`. Nothing here re-derives it.

Rule 1: `render.py` burns the .ass LAST in the filter chain, so this file
writes the final pixel pass and nothing re-times it afterwards.

Three consequences that are easy to get wrong:

  * A word that fell inside a cut has no output time and produces no cue.
  * A cue may never span two ranges. The two sides of a cut are not adjacent
    in the source, so a cue straddling the join sits over the cut and reads as
    a caption that outlives the sentence it belongs to. Chunking therefore
    breaks hard at every range boundary.
  * `render.py` re-styles the burn with `force_style=`, which overrides the
    [V4+ Styles] block but NOT inline override tags. Position, size and colour
    are written inline on every dialogue line as well, so the safe-zone margin
    and the readable size survive the vendored renderer unchanged.

Arabic is written from `display`, never from the normalised matching form, and
never uppercased: `uppercase_latin` applies to Latin-only lines only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

try:  # package import
    from . import config as configs
    from . import textnorm
    from .edl import EDL
    from .paths import EditPaths
    from .words import WordsDoc
except ImportError:  # `python helpers/captions.py ...`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from helpers import config as configs
    from helpers import textnorm
    from helpers.edl import EDL
    from helpers.paths import EditPaths
    from helpers.words import WordsDoc


# A cue ends after these; a comma-class mark only ends a cue that already has
# the minimum number of words, otherwise every list turns into one-word cues.
_STRONG_PUNCT = ".!?؟…"
_WEAK_PUNCT = ",،;:؛"

# The vendored video-use caption style was tuned against libass' default
# PlayResY of 288 (FontSize=18 there is ~6% of frame height). `captions.yaml`
# inherits that scale for `font_size`, while the margins are real pixels in the
# aspect's own grid. Font sizes are therefore rescaled to the PlayRes we write;
# margins are used as-is.
VENDOR_PLAYRES_Y = 288

STYLE_PLAIN = "Zeta"
STYLE_KARAOKE = "ZetaKaraoke"


# -------- model --------------------------------------------------------------


@dataclass
class CueWord:
    """One word of a cue, already on the output timeline."""

    text: str
    start: float | None = None
    end: float | None = None
    lang: str = "ary"

    @property
    def timed(self) -> bool:
        return self.start is not None and self.end is not None


@dataclass
class Cue:
    start: float
    end: float
    words: list[CueWord] = field(default_factory=list)
    range_index: int = 0
    index: int = 0

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def has_untimed(self) -> bool:
        return any(not w.timed for w in self.words)

    @property
    def latin_only(self) -> bool:
        kinds = {w.lang for w in self.words}
        return bool(kinds & {"lat"}) and not (kinds - {"lat", "num", "other"})

    def to_dict(self) -> dict:
        return {"index": self.index, "start": round(self.start, 3),
                "end": round(self.end, 3), "range": self.range_index,
                "text": self.text}


@dataclass
class BidiFinding:
    """A line whose direction libass has to resolve. Eyeball it on a sample."""

    index: int
    start: float
    end: float
    text: str
    kinds: list[str]
    enclosed_run: bool  # a Latin/number run with Arabic on BOTH sides

    def to_dict(self) -> dict:
        return {"index": self.index, "start": round(self.start, 3),
                "end": round(self.end, 3), "text": self.text,
                "kinds": self.kinds, "enclosed_run": self.enclosed_run}


# -------- config -------------------------------------------------------------


@dataclass
class CaptionStyle:
    """Everything the ASS writer needs, resolved for one aspect."""

    aspect: str
    play_res: tuple[int, int]
    font: str
    fallbacks: list[str]
    font_size: int
    margin_v: int
    margin_h: int
    words_per_line: tuple[int, int]
    max_chars_per_line: int
    chunking: dict
    style: dict


def resolve_style(aspect: str | None = None, *, captions_cfg: dict | None = None,
                  layout_cfg: dict | None = None) -> CaptionStyle:
    """Resolve `configs/captions.yaml` + `configs/layout.yaml` for one aspect.

    margin_v is not taste: layout.yaml measures the Instagram UI band at 420 px
    of a 1080x1920 Reel (username, caption, audio ticker, tab bar) plus a right
    action rail, and captions.yaml puts the caption baseline at 480 px, above
    that band. Lower it and the captions render underneath the app chrome.
    """
    caps = captions_cfg if captions_cfg is not None else configs.load("captions")
    layout = layout_cfg if layout_cfg is not None else configs.load("layout")
    key = aspect or layout.get("default_aspect") or "9:16"
    block = (caps.get("aspects") or {}).get(key)
    if block is None:
        raise KeyError(f"unknown aspect {key!r}; known: {sorted(caps.get('aspects') or {})}")

    # PlayRes is the frame the renderer actually produces; the caption grid has
    # to be that same frame or the margins stop meaning pixels.
    res = ((layout.get("aspects") or {}).get(key) or {}).get("resolution") \
        or block.get("resolution") or [1080, 1920]

    wpl = block.get("words_per_line") or [2, 4]
    font = caps.get("font") or {}
    return CaptionStyle(
        aspect=key,
        play_res=(int(res[0]), int(res[1])),
        font=font.get("primary", "Noto Sans Arabic"),
        fallbacks=[v for k, v in font.items() if k != "primary" and v],
        font_size=int(block.get("font_size", 18)),
        margin_v=int(block.get("margin_v", 90)),
        margin_h=int(block.get("margin_h", 90)),
        words_per_line=(int(wpl[0]), int(wpl[-1])),
        max_chars_per_line=int(block.get("max_chars_per_line", 32)),
        chunking=caps.get("chunking") or {},
        style=caps.get("style") or {},
    )


# -------- source time -> output time ----------------------------------------


@dataclass
class _Mapped:
    """A word placed on the output timeline, or parked without timing."""

    text: str
    lang: str
    start: float | None
    end: float | None
    range_index: int
    punct_break: str  # "" | "weak" | "strong"

    @property
    def timed(self) -> bool:
        return self.start is not None and self.end is not None


def _range_index(edl: EDL, source: str, t: float) -> int | None:
    for i, r in enumerate(edl.ranges):
        if r.source == source and r.start <= t < r.end:
            return i
    return None


def _trailing_punct(token: str) -> str:
    stripped = token.rstrip("\"'»)]”’")
    if stripped and stripped[-1] in _STRONG_PUNCT:
        return "strong"
    if stripped and stripped[-1] in _WEAK_PUNCT:
        return "weak"
    return ""


def map_words(doc: WordsDoc, edl: EDL, source: str | None = None) -> list[_Mapped]:
    """Place every surviving word of `doc` on the output timeline.

    Words inside a cut are dropped. A word with no alignment keeps its text but
    no timing, so the line still reads correctly and the karaoke pass can tell
    that it must degrade.
    """
    name = source or (doc.source or {}).get("name")
    if not name:
        # One-source EDLs are the common case; only ambiguity is an error.
        if len(edl.sources) != 1:
            raise ValueError("words doc has no source name and the EDL has several sources")
        name = next(iter(edl.sources))

    out: list[_Mapped] = []
    for w in doc.words:
        text = (w.display or w.word or "").strip()
        if not text:
            continue
        lang = w.lang or textnorm.token_lang(text)
        if not w.timed:
            # Untimed words ride along with the word before them; they cannot
            # be placed on their own and they disable karaoke for their cue.
            if out:
                out.append(_Mapped(text, lang, None, None, out[-1].range_index,
                                   _trailing_punct(text)))
            continue
        out_start = edl.to_output_time(name, w.start)
        if out_start is None:
            continue  # cut: no cue, ever
        idx = _range_index(edl, name, w.start)
        rng = edl.ranges[idx]
        # Clamp the tail to the range. A word whose end fell in the cut would
        # otherwise stretch its cue over material that no longer exists. This
        # is a duration, not a second mapping: the offset came from the EDL.
        tail = min(w.end, rng.end)
        out.append(_Mapped(text, lang, out_start, out_start + max(0.0, tail - w.start),
                           idx, _trailing_punct(text)))
    return out


# -------- chunking -----------------------------------------------------------


def _chunk(mapped: Sequence[_Mapped], style: CaptionStyle) -> list[list[_Mapped]]:
    ch = style.chunking
    gap_s = float(ch.get("break_on_gap_ms", 300)) / 1000.0
    on_punct = bool(ch.get("break_on_punctuation", True))
    max_dur = float(ch.get("max_cue_duration_s", 3.0))
    lo, hi = style.words_per_line

    groups: list[list[_Mapped]] = []
    cur: list[_Mapped] = []
    cur_start: float | None = None
    prev_end: float | None = None

    def flush() -> None:
        nonlocal cur, cur_start, prev_end
        if cur:
            groups.append(cur)
        cur, cur_start, prev_end = [], None, None

    for m in mapped:
        if cur:
            must_break = False
            if m.range_index != cur[-1].range_index:
                must_break = True  # a cue never spans a cut
            elif len(cur) >= hi:
                must_break = True
            elif len(" ".join(w.text for w in cur)) + 1 + len(m.text) > style.max_chars_per_line:
                must_break = True
            elif m.timed and prev_end is not None and (m.start - prev_end) >= gap_s:
                must_break = True
            elif m.timed and cur_start is not None and (m.end - cur_start) > max_dur:
                must_break = True
            elif on_punct and cur[-1].punct_break == "strong":
                must_break = True
            elif on_punct and cur[-1].punct_break == "weak" and len(cur) >= lo:
                must_break = True
            if must_break:
                flush()
        cur.append(m)
        if m.timed:
            if cur_start is None:
                cur_start = m.start
            prev_end = m.end
    flush()
    return groups


def _merge_runts(groups: list[list[_Mapped]], style: CaptionStyle) -> list[list[_Mapped]]:
    """Fold a below-band cue back into its neighbour when that stays legal."""
    lo, hi = style.words_per_line
    max_dur = float(style.chunking.get("max_cue_duration_s", 3.0))
    gap_s = float(style.chunking.get("break_on_gap_ms", 300)) / 1000.0
    out: list[list[_Mapped]] = []
    for g in groups:
        if out and len(g) < lo:
            prev = out[-1]
            same_range = prev[-1].range_index == g[0].range_index
            joined = len(" ".join(w.text for w in prev + g))
            timed = [w for w in prev + g if w.timed]
            dur = (timed[-1].end - timed[0].start) if timed else 0.0
            gap = (g[0].start - prev[-1].end) if (g[0].timed and prev[-1].timed) else 0.0
            if (same_range and len(prev) + len(g) <= hi
                    and joined <= style.max_chars_per_line
                    and dur <= max_dur and gap < gap_s
                    and prev[-1].punct_break != "strong"):
                out[-1] = prev + g
                continue
        out.append(g)
    return out


def _to_cues(groups: list[list[_Mapped]], edl: EDL, style: CaptionStyle) -> list[Cue]:
    min_dur = float(style.chunking.get("min_cue_duration_s", 0.3))
    offsets = edl.offsets()
    cues: list[Cue] = []
    for g in groups:
        timed = [m for m in g if m.timed]
        if not timed:
            continue  # nothing to hang a time on
        cue = Cue(
            start=timed[0].start,
            end=max(m.end for m in timed),
            words=[CueWord(m.text, m.start, m.end, m.lang) for m in g],
            range_index=g[0].range_index,
        )
        cues.append(cue)

    for i, cue in enumerate(cues):
        cue.index = i
        if cue.duration >= min_dur:
            continue
        # Hold a short cue longer, but never past the next cue and never past
        # the end of its own segment -- that would put it over the cut.
        r = edl.ranges[cue.range_index]
        seg_end = offsets[cue.range_index] + r.duration
        limit = seg_end
        if i + 1 < len(cues):
            limit = min(limit, cues[i + 1].start)
        # Only ever extend: clamping must not cut a cue short.
        cue.end = max(cue.end, min(cue.start + min_dur, limit))
    return cues


def build_cues(words: WordsDoc | Mapping[str, WordsDoc], edl: EDL, *,
               aspect: str | None = None, style: CaptionStyle | None = None,
               source: str | None = None) -> list[Cue]:
    """`words.json` (+ EDL) -> cues on the output timeline."""
    st = style or resolve_style(aspect or edl.aspect)
    mapped: list[_Mapped] = []
    if isinstance(words, WordsDoc):
        mapped = map_words(words, edl, source)
    else:
        for name, doc in words.items():
            mapped.extend(map_words(doc, edl, name))
        # Several takes interleave on the output timeline; only their output
        # order matters to a reader.
        mapped.sort(key=lambda m: (m.start if m.start is not None else 0.0))
    groups = _merge_runts(_chunk(mapped, st), st)
    return _to_cues(groups, edl, st)


# -------- ASS ----------------------------------------------------------------


def _ass_time(t: float) -> str:
    t = max(0.0, t)
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def _srt_time(t: float) -> str:
    t = max(0.0, t)
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _inline_colour(ass_colour: str) -> str:
    """`&HAABBGGRR` (style field) -> `&HBBGGRR&` (inline \\1c/\\2c form)."""
    hexpart = (ass_colour or "").upper().replace("&H", "").replace("&", "")
    hexpart = re.sub(r"[^0-9A-F]", "", hexpart) or "FFFFFF"
    if len(hexpart) >= 8:
        hexpart = hexpart[-6:]
    return f"&H{hexpart[-6:].rjust(6, '0')}&"


def scaled_font_size(style: CaptionStyle) -> int:
    return max(8, int(round(style.font_size * style.play_res[1] / VENDOR_PLAYRES_Y)))


def cue_text_for_display(cue: Cue, style: CaptionStyle) -> str:
    """The reader's text. Arabic is never uppercased, whatever the config says."""
    text = cue.text
    if style.style.get("uppercase_latin") and cue.latin_only:
        return text.upper()
    return text


def ass_header(style: CaptionStyle) -> str:
    st = style.style
    size = scaled_font_size(style)
    bold = -1 if st.get("bold", True) else 0
    primary = st.get("primary_colour", "&H00FFFFFF")
    highlight = st.get("highlight_colour", "&H0000D7FF")
    fields = ("Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
              "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, "
              "Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, "
              "Encoding")

    def style_line(name: str, first: str, second: str) -> str:
        return (f"Style: {name},{style.font},{size},{first},{second},"
                f"{st.get('outline_colour', '&H00000000')},{st.get('back_colour', '&H80000000')},"
                f"{bold},0,0,0,100,100,0,0,{st.get('border_style', 1)},"
                f"{st.get('outline', 3)},{st.get('shadow', 0)},{st.get('alignment', 2)},"
                f"{style.margin_h},{style.margin_h},{style.margin_v},1")

    lines = [
        "[Script Info]",
        "; Zeta auto editor - burned LAST in the filter chain (hard rule 1),",
        "; times are OUTPUT-timeline times (hard rule 5).",
        f"; font fallbacks (resolved by fontconfig): {', '.join(style.fallbacks) or 'none'}",
        "ScriptType: v4.00+",
        "WrapStyle: 2",           # we chunk; libass must not re-wrap
        "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: TV.709",
        f"PlayResX: {style.play_res[0]}",
        f"PlayResY: {style.play_res[1]}",
        "",
        "[V4+ Styles]",
        f"Format: {fields}",
        style_line(STYLE_PLAIN, primary, primary),
        # Karaoke flips SecondaryColour -> PrimaryColour as each \k elapses, so
        # the karaoke style carries the highlight as its primary.
        style_line(STYLE_KARAOKE, highlight, primary),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    return "\n".join(lines)


def _inline_prefix(style: CaptionStyle, karaoke: bool) -> str:
    """Overrides that `render.py`'s force_style cannot flatten."""
    st = style.style
    w, h = style.play_res
    align = int(st.get("alignment", 2))
    x = w // 2
    y = h - style.margin_v
    # The font name is the load-bearing one. force_style pins FontName=Helvetica,
    # which has no Arabic coverage at all: without an inline \fn every Darija
    # caption is at the mercy of whatever fontconfig decides to substitute, which
    # differs between Ali's Mac and a Linux batch runner. \bord likewise, because
    # force_style's Outline=2 is thinner than the config asks for at 1920 tall.
    tags = [f"\\an{align}", f"\\pos({x},{y})", f"\\fs{scaled_font_size(style)}",
            f"\\fn{style.font}", f"\\bord{st.get('outline', 3)}",
            f"\\b{1 if st.get('bold', True) else 0}"]
    if karaoke:
        tags.append(f"\\1c{_inline_colour(st.get('highlight_colour', '&H0000D7FF'))}")
        tags.append(f"\\2c{_inline_colour(st.get('primary_colour', '&H00FFFFFF'))}")
    else:
        tags.append(f"\\1c{_inline_colour(st.get('primary_colour', '&H00FFFFFF'))}")
    return "{" + "".join(tags) + "}"


def _karaoke_body(cue: Cue, style: CaptionStyle) -> str:
    """`\\k` timing per word; the gap before a word belongs to the word before it."""
    parts: list[str] = []
    n = len(cue.words)
    for i, w in enumerate(cue.words):
        nxt = cue.words[i + 1].start if i + 1 < n else cue.end
        span = max(0.0, (nxt if nxt is not None else cue.end) - (w.start or cue.start))
        text = w.text.upper() if (style.style.get("uppercase_latin") and cue.latin_only) else w.text
        parts.append("{\\k%d}%s" % (int(round(span * 100)), text))
    return " ".join(parts)


def render_ass(cues: Sequence[Cue], style: CaptionStyle) -> str:
    karaoke_on = bool(style.style.get("highlight_active_word"))
    lines = [ass_header(style)]
    for cue in cues:
        # Karaoke needs a time for every word; one unaligned word and the whole
        # line would drift, so the cue degrades to a plain one instead.
        karaoke = karaoke_on and not cue.has_untimed and len(cue.words) > 1
        name = STYLE_KARAOKE if karaoke else STYLE_PLAIN
        body = _karaoke_body(cue, style) if karaoke else cue_text_for_display(cue, style)
        text = _inline_prefix(style, karaoke) + body.replace("\n", " ")
        lines.append(f"Dialogue: 0,{_ass_time(cue.start)},{_ass_time(cue.end)},{name},,0,0,0,,{text}")
    return "\n".join(lines) + "\n"


def render_srt(cues: Sequence[Cue], style: CaptionStyle | None = None) -> str:
    out: list[str] = []
    for i, cue in enumerate(cues, start=1):
        text = cue_text_for_display(cue, style) if style else cue.text
        out.append(f"{i}\n{_srt_time(cue.start)} --> {_srt_time(cue.end)}\n{text}\n")
    return "\n".join(out)


# -------- bidi ---------------------------------------------------------------


def check_bidi(cues: Sequence[Cue]) -> list[BidiFinding]:
    """Lines where libass has to resolve a direction change.

    Mixed Darija/French lines are the ones that render wrong (a Latin run or a
    number can jump to the far side of the line), and the acceptance test asks
    for three samples to be eyeballed before the style is trusted. This lists
    the candidates instead of guessing; `enclosed_run` marks the hard case,
    a Latin/number run with Arabic on both sides.
    """
    findings: list[BidiFinding] = []
    for cue in cues:
        kinds = [textnorm.token_lang(w.text) for w in cue.words]
        if "ary" not in kinds or not ({"lat", "num"} & set(kinds)):
            continue
        enclosed = False
        for i, k in enumerate(kinds):
            if k in ("lat", "num") and "ary" in kinds[:i] and "ary" in kinds[i + 1:]:
                enclosed = True
                break
        findings.append(BidiFinding(index=cue.index, start=cue.start, end=cue.end,
                                    text=cue.text, kinds=sorted(set(kinds)),
                                    enclosed_run=enclosed))
    return findings


# -------- write --------------------------------------------------------------


def write_captions(words: WordsDoc | Mapping[str, WordsDoc], edl: EDL,
                   edit_paths: EditPaths, *, aspect: str | None = None,
                   style: CaptionStyle | None = None,
                   source: str | None = None, srt: bool | None = None) -> dict:
    st = style or resolve_style(aspect or edl.aspect)
    cues = build_cues(words, edl, style=st, source=source)
    edit_paths.captions.mkdir(parents=True, exist_ok=True)
    ass_path = edit_paths.captions / "final.ass"
    ass_path.write_text(render_ass(cues, st), encoding="utf-8")

    caps = configs.load("captions") if srt is None else {}
    want_srt = caps.get("export_srt", True) if srt is None else srt
    srt_path = edit_paths.captions / "final.srt"
    if want_srt:
        srt_path.write_text(render_srt(cues, st), encoding="utf-8")

    bidi = check_bidi(cues)
    return {
        "ass": ass_path,
        "subtitles_field": subtitles_field(edit_paths, ass_path),
        "srt": srt_path if want_srt else None,
        "cues": cues,
        "aspect": st.aspect,
        "bidi": [f.to_dict() for f in bidi],
    }


def subtitles_field(edit_paths: EditPaths, ass_path: Path) -> str:
    """The value to put in `EDL.subtitles` for this .ass.

    `render.py` resolves a relative subtitles path against the directory the
    edl.json sits in (`<videos>/edit/`), not against `<videos>/`, so the field
    has to be spelled relative to the edit dir or the burn is silently skipped
    with a warning.
    """
    try:
        return str(ass_path.resolve().relative_to(edit_paths.edit.resolve()))
    except ValueError:
        return str(ass_path.resolve())


def cues_to_json(cues: Sequence[Cue]) -> list[dict]:
    return [c.to_dict() for c in cues]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="words.json + edl.json -> burned-caption ASS/SRT on the output timeline")
    ap.add_argument("words", type=Path, nargs="+", help="one or more <name>.words.json")
    ap.add_argument("edl", type=Path, help="edl.json")
    ap.add_argument("--aspect", default=None, help="9:16 | 4:5 | 1:1 | 16:9 (default: EDL/layout)")
    ap.add_argument("--videos-dir", type=Path, default=None,
                    help="output root; default is the parent of the EDL's edit/ dir")
    ap.add_argument("--no-srt", action="store_true")
    ap.add_argument("--bidi", action="store_true", help="print mixed RTL/LTR lines and exit")
    args = ap.parse_args(argv)

    edl = EDL.load(args.edl)
    docs = {}
    for p in args.words:
        doc = WordsDoc.load(p)
        name = (doc.source or {}).get("name") or p.name.split(".")[0]
        docs[name] = doc

    videos_dir = args.videos_dir or args.edl.resolve().parent.parent
    paths = EditPaths.for_videos_dir(videos_dir)
    st = resolve_style(args.aspect or edl.aspect)

    if args.bidi:
        for f in check_bidi(build_cues(docs, edl, style=st)):
            print(json.dumps(f.to_dict(), ensure_ascii=False))
        return 0

    out = write_captions(docs, edl, paths, style=st, srt=not args.no_srt)
    print(f"{len(out['cues'])} cues @ {st.aspect} ({st.play_res[0]}x{st.play_res[1]})")
    print(f"  {out['ass']}")
    if out["srt"]:
        print(f"  {out['srt']}")
    if out["bidi"]:
        print(f"  {len(out['bidi'])} mixed RTL/LTR line(s): check 3 on a render before trusting the style")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
