"""Forced alignment: where the timestamps actually come from.

WhisperX 3.8.6 `load_align_model()` + `align()` against
`jonatasgrosman/wav2vec2-large-xlsr-53-arabic`. The transcriber supplies WHAT
was said, this module supplies WHEN, and nothing else in the pipeline is
allowed to write a `start` or an `end`.

Three things make this work on Darija rather than on paper:

  * `interpolate_method="ignore"`. The alternative ("nearest") invents a
    timestamp for every token the acoustic model could not place, which is
    exactly the token you must not cut next to. A word with no timing is
    honest; QA counts it against coverage and re-aligns its window.
  * An ALIAS MAP. The Arabic wav2vec2 model has an Arabic character vocabulary:
    "Nvidia", "2026" and "milliard" contain nothing it can emit, so they are
    handed to the aligner as Arabic-script approximations of how they sound.
    The alias lives on `Word.alias`; `Word.word`/`display` are untouched, so
    captions and diffs never see it.
  * Latin runs are re-aligned with the English wav2vec2 model inside the window
    bounded by their neighbouring Arabic words, which is both more accurate
    than the transliteration and impossible to run away with -- it cannot
    escape the bracket its neighbours put it in.

Run standalone:
    python helpers/align_whisperx.py edit/transcripts/raw01.words.json edit/audio/raw01.16k.wav
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Sequence

try:
    from . import config as cfgmod
    from . import diff_align, textnorm
    from .words import Word, WordsDoc
except ImportError:  # running as `python helpers/align_whisperx.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from helpers import config as cfgmod
    from helpers import diff_align, textnorm
    from helpers.words import Word, WordsDoc

SAMPLE_RATE = 16000


class AlignError(RuntimeError):
    pass


# -- aliases ------------------------------------------------------------------

# Digits read out loud in Darija. Anything above the table is read digit by
# digit: forced alignment only needs a plausible phoneme sequence of roughly the
# right length, not a grammatical reading.
_DIGIT_WORDS = {
    "0": "صفر", "1": "واحد", "2": "جوج", "3": "تلاتة", "4": "ربعة",
    "5": "خمسة", "6": "ستة", "7": "سبعة", "8": "تمنية", "9": "تسعة",
}
_NUMBER_WORDS = {
    "10": "عشرة", "11": "حداش", "12": "طناش", "13": "تلطاش", "14": "ربعطاش",
    "15": "خمسطاش", "16": "سطاش", "17": "سبعطاش", "18": "تمنطاش", "19": "تسعطاش",
    "20": "عشرين", "30": "تلاتين", "40": "ربعين", "50": "خمسين", "60": "ستين",
    "70": "سبعين", "80": "تمانين", "90": "تسعين", "100": "مية", "1000": "ألف",
}
_SYMBOL_WORDS = {"%": "فالمية", "$": "دولار", "€": "أورو", "+": "زايد", "-": "ناقص"}

# Latin -> Arabic grapheme approximation. Digraphs first; order matters.
_TRANSLIT: tuple[tuple[str, str], ...] = (
    ("tch", "تش"), ("sch", "ش"), ("ch", "ش"), ("sh", "ش"), ("th", "ت"),
    ("ph", "ف"), ("gh", "غ"), ("kh", "خ"), ("qu", "ك"), ("ck", "ك"),
    ("oo", "و"), ("ou", "و"), ("au", "و"), ("eau", "و"), ("ai", "ي"),
    ("ee", "ي"), ("ea", "ي"), ("ey", "ي"), ("ie", "ي"), ("oi", "وا"),
    ("an", "ان"), ("en", "ان"), ("on", "ون"), ("in", "ين"), ("un", "ان"),
    ("a", "ا"), ("b", "ب"), ("c", "ك"), ("d", "د"), ("e", "ي"), ("f", "ف"),
    ("g", "ݣ"), ("h", "ه"), ("i", "ي"), ("j", "ج"), ("k", "ك"), ("l", "ل"),
    ("m", "م"), ("n", "ن"), ("o", "و"), ("p", "ب"), ("q", "ق"), ("r", "ر"),
    ("s", "س"), ("t", "ت"), ("u", "و"), ("v", "ف"), ("w", "و"), ("x", "كس"),
    ("y", "ي"), ("z", "ز"),
)

_LATIN_ONLY = re.compile(r"[^a-z]")


def transliterate_latin(token: str) -> str:
    """Arabic-script approximation of a Latin token, or "" if there is none."""
    t = _LATIN_ONLY.sub("", token.casefold())
    out: list[str] = []
    i = 0
    while i < len(t):
        for src, dst in _TRANSLIT:
            if t.startswith(src, i):
                out.append(dst)
                i += len(src)
                break
        else:
            i += 1
    return "".join(out)


def spell_number(token: str) -> str:
    """Read a numeric token as Darija words, space separated."""
    parts: list[str] = []
    for chunk in re.findall(r"\d+|[%$€+\-]", token):
        if chunk in _SYMBOL_WORDS:
            parts.append(_SYMBOL_WORDS[chunk])
        elif chunk in _NUMBER_WORDS:
            parts.append(_NUMBER_WORDS[chunk])
        elif chunk in _DIGIT_WORDS:
            parts.append(_DIGIT_WORDS[chunk])
        else:
            parts.extend(_DIGIT_WORDS.get(d, "") for d in chunk)
    return " ".join(p for p in parts if p)


def build_alias_map(cfg: dict | None = None) -> dict[str, str]:
    """Normalised token -> Arabic grapheme form handed to the aligner.

    Seeded from `configs/transcribe.yaml: custom_vocabulary`, which is the list
    of terms Ali already knows the ASR mangles, so their alias is computed once
    and stays stable across runs instead of being re-derived per transcript.
    `alignment.aliases` in config overrides any entry.
    """
    cfg = cfg or {}
    out: dict[str, str] = {}
    for term in cfg.get("custom_vocabulary") or []:
        key = textnorm.normalize_token(str(term))
        if not key or textnorm.token_lang(str(term)) == "ary":
            continue  # already in the model's alphabet
        alias = transliterate_latin(str(term))
        if alias:
            out[key] = alias
    for k, v in (cfgmod.get(cfg, "alignment.aliases") or {}).items():
        norm = textnorm.normalize_token(str(k))
        if norm:
            out[norm] = str(v)
    return out


def alias_for(token: str, alias_map: dict[str, str] | None = None) -> str | None:
    """The alias for one token, or None when the model can read it as-is."""
    alias_map = alias_map or {}
    norm = textnorm.normalize_token(token)
    if not norm:
        return None
    if norm in alias_map:
        return alias_map[norm]
    kind = textnorm.token_lang(token)
    if kind == "ary":
        return None
    if kind == "num":
        return spell_number(token) or None
    if kind == "lat":
        return transliterate_latin(token) or None
    return None


def apply_aliases(doc: WordsDoc, cfg: dict | None = None) -> int:
    """Fill `Word.alias` wherever the Arabic model needs one. Returns the count."""
    alias_map = build_alias_map(cfg)
    n = 0
    for w in doc.words:
        if w.alias:
            continue
        a = alias_for(w.word, alias_map)
        if a:
            w.alias = a
            n += 1
    return n


def align_token(w: Word) -> str:
    return w.alias or w.word


# -- segment / token bookkeeping ----------------------------------------------


def build_align_text(words: Sequence[Word]) -> tuple[str, list[tuple[int, int]]]:
    """Alias-substituted text plus each word's `[i, j)` range in that token stream.

    An alias can expand to several tokens ("2026" -> four number words), so the
    mapping back from aligner output to our word list cannot assume 1:1.
    """
    tokens: list[str] = []
    spans: list[tuple[int, int]] = []
    for w in words:
        parts = [p for p in align_token(w).split() if p]
        spans.append((len(tokens), len(tokens) + len(parts)))
        tokens.extend(parts)
    return (" ".join(tokens), spans)


def _returned_tokens(seg: dict) -> list[dict]:
    return [w for w in (seg.get("words") or [])]


def map_returned(alias_tokens: Sequence[str], returned: Sequence[dict]) -> list[dict | None]:
    """Line the aligner's word list back up with the tokens we sent it.

    Equal lengths is the normal case. When whisperx drops a token it could not
    read at all, fall back to the shared diff engine rather than shifting every
    subsequent timestamp by one word.
    """
    if len(returned) == len(alias_tokens):
        return list(returned)
    out: list[dict | None] = [None] * len(alias_tokens)
    a = [textnorm.normalize_token(t) for t in alias_tokens]
    b = [textnorm.normalize_token(str(r.get("word", ""))) for r in returned]
    for a0, b0, size in diff_align.match_spans(a, b):
        for k in range(size):
            out[a0 + k] = returned[b0 + k]
    return out


def apply_aligned(words: Sequence[Word], spans: Sequence[tuple[int, int]],
                  aligned: Sequence[dict | None], *, offset: float = 0.0) -> int:
    """Write timings onto words from the aligner's per-token output.

    A word backed by several alias tokens spans from the first token that got a
    start to the last that got an end; a word whose tokens were all unplaced
    keeps `start is None`, which is what `interpolate_method="ignore"` is for.
    """
    timed = 0
    for w, (i, j) in zip(words, spans):
        starts, ends, scores = [], [], []
        for tok in aligned[i:j]:
            if not tok:
                continue
            s, e = tok.get("start"), tok.get("end")
            if s is None or e is None:
                continue
            starts.append(float(s))
            ends.append(float(e))
            if tok.get("score") is not None:
                scores.append(float(tok["score"]))
        if not starts:
            continue
        w.start = min(starts) + offset
        w.end = max(ends) + offset
        if scores:
            w.score = round(sum(scores) / len(scores), 4)
        timed += 1
    return timed


def _segment_plan(doc: WordsDoc, duration: float, pad: float
                  ) -> list[tuple[int, int, float, float]]:
    """`(word_start, word_end, t_start, t_end)` windows to align independently.

    Uses the transcriber's approximate segment times as anchors -- padded,
    because they are approximate and a segment window that clips real speech
    pushes words outside it into "unalignable". With no hints at all the whole
    file is one window.
    """
    segments = doc.meta.get("segments") or []
    spans = doc.meta.get("segment_word_spans") or []
    if not segments or len(spans) != len(segments):
        return [(0, len(doc.words), 0.0, duration)]

    raw: list[tuple[int, int, float, float]] = []
    for seg, (a, b) in zip(segments, spans):
        if b <= a:
            continue
        s, e = seg.get("start"), seg.get("end")
        if s is None or e is None:
            continue
        raw.append((a, b, float(s), float(e)))
    if not raw:
        return [(0, len(doc.words), 0.0, duration)]

    # Padding must not let two windows claim the same audio: both would place a
    # word there and the result is an overlap QA then has to repair. Each side
    # of a gap gets at most half of it.
    out: list[tuple[int, int, float, float]] = []
    for k, (a, b, s, e) in enumerate(raw):
        lo, hi = max(0.0, s - pad), min(duration, e + pad)
        if k:
            lo = max(lo, (raw[k - 1][3] + s) / 2.0)
        if k + 1 < len(raw):
            hi = min(hi, (e + raw[k + 1][2]) / 2.0)
        if hi <= lo:  # hints that overlap each other: trust them unpadded
            lo, hi = s, max(e, s + 1e-3)
        out.append((a, b, lo, hi))
    return out


def latin_runs(words: Sequence[Word], start: int = 0) -> list[tuple[int, int]]:
    """Index runs of consecutive Latin-script words, as `[i, j)` absolute spans."""
    runs: list[tuple[int, int]] = []
    i = 0
    n = len(words)
    while i < n:
        if textnorm.is_latin(words[i].word):
            j = i
            while j < n and textnorm.is_latin(words[j].word):
                j += 1
            runs.append((start + i, start + j))
            i = j
        else:
            i += 1
    return runs


def bound_run(words: Sequence[Word], i: int, j: int, *,
              lo: float, hi: float) -> tuple[float, float]:
    """Time bracket for a Latin run: the neighbouring words' timings.

    The English model is only allowed to place the run inside the silence its
    Arabic neighbours leave for it, so a mis-read cannot drag a caption or a cut
    edge outside the span the word really occupies.
    """
    before = next((w for w in reversed(words[:i]) if w.timed), None)
    after = next((w for w in words[j:] if w.timed), None)
    a = before.end if before else lo
    b = after.start if after else hi
    return (max(lo, min(a, hi)), min(hi, max(b, lo)))


# -- whisperx -----------------------------------------------------------------


def resolve_device(cfg: dict | None = None) -> str:
    """`auto` -> cuda when torch sees a GPU, else cpu. Alignment is fine on CPU."""
    want = str(cfgmod.get(cfg or {}, "alignment.device", "auto") or "auto").lower()
    if want != "auto":
        return want
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _whisperx():
    try:
        import whisperx
    except ImportError as exc:
        raise AlignError(
            "whisperx is not installed: uv pip install -e '.[align]' "
            "(whisperx==3.8.6 + torch)") from exc
    return whisperx


def load_audio(wav: str | Path):
    """16 kHz float32 mono array, via whisperx (ffmpeg under the hood)."""
    return _whisperx().load_audio(str(wav))


class _ModelCache:
    """Loading a wav2vec2 model costs seconds; QA re-aligns call back in."""

    def __init__(self) -> None:
        self._models: dict[tuple[str, str, str], tuple[Any, Any]] = {}

    def get(self, language: str, device: str, model_name: str | None) -> tuple[Any, Any]:
        key = (language, device, model_name or "")
        if key not in self._models:
            wx = _whisperx()
            kw = {"language_code": language, "device": device}
            if model_name:
                kw["model_name"] = model_name
            self._models[key] = wx.load_align_model(**kw)
        return self._models[key]


_MODELS = _ModelCache()


def _run_align(segments: list[dict], audio, *, language: str, device: str,
               model_name: str | None, interpolate: str) -> list[dict]:
    wx = _whisperx()
    model, metadata = _MODELS.get(language, device, model_name)
    result = wx.align(segments, model, metadata, audio, device,
                      return_char_alignments=False,
                      interpolate_method=interpolate)
    return result.get("segments") or []


def align(
    doc: WordsDoc,
    wav: str | Path,
    cfg: dict | None = None,
    *,
    device: str | None = None,
    audio: Any = None,
) -> WordsDoc:
    """Fill `start`, `end` and `score` on every word of `doc`. Returns `doc`."""
    cfg = cfg or {}
    acfg = cfgmod.get(cfg, "alignment") or {}
    model_name = acfg.get("model")
    interpolate = acfg.get("interpolate_method", "ignore")
    device = device or resolve_device(cfg)
    apply_aliases(doc, cfg)

    if audio is None:
        audio = load_audio(wav)
    duration = len(audio) / float(SAMPLE_RATE)
    pad = float(acfg.get("segment_pad_s", 0.5))

    plan = _segment_plan(doc, duration, pad)
    payload: list[dict] = []
    token_spans: list[list[tuple[int, int]]] = []
    for a, b, t0, t1 in plan:
        text, spans = build_align_text(doc.words[a:b])
        payload.append({"start": t0, "end": t1, "text": text})
        token_spans.append(spans)

    segments = _run_align(payload, audio, language=doc.language or "ar", device=device,
                          model_name=model_name, interpolate=interpolate)

    for (a, b, _t0, _t1), spans, seg in zip(plan, token_spans, segments):
        alias_tokens = [t for w in doc.words[a:b] for t in align_token(w).split()]
        apply_aligned(doc.words[a:b], spans,
                      map_returned(alias_tokens, _returned_tokens(seg)))

    if acfg.get("align_latin_spans", True):
        align_latin_spans(doc, audio, cfg, device=device, duration=duration)

    doc.aligner = f"whisperx:{model_name}"
    doc.meta["alignment"] = {
        "device": device,
        "interpolate_method": interpolate,
        "english_model": acfg.get("english_model") if acfg.get("align_latin_spans", True)
        else None,
        "coverage": round(doc.coverage(), 4),
    }
    return doc


def align_latin_spans(doc: WordsDoc, audio: Any, cfg: dict | None = None, *,
                      device: str | None = None, duration: float | None = None) -> int:
    """Re-align Latin runs with the English model, bounded by their neighbours."""
    cfg = cfg or {}
    acfg = cfgmod.get(cfg, "alignment") or {}
    english = acfg.get("english_model", "WAV2VEC2_ASR_BASE_960H")
    interpolate = acfg.get("interpolate_method", "ignore")
    device = device or resolve_device(cfg)
    duration = duration if duration is not None else len(audio) / float(SAMPLE_RATE)

    fixed = 0
    for i, j in latin_runs(doc.words):
        lo, hi = bound_run(doc.words, i, j, lo=0.0, hi=duration)
        if hi - lo < 0.05:
            continue
        clip = audio[int(lo * SAMPLE_RATE):int(hi * SAMPLE_RATE)]
        text = " ".join(w.word for w in doc.words[i:j])
        try:
            segments = _run_align([{"start": 0.0, "end": hi - lo, "text": text}], clip,
                                  language="en", device=device, model_name=english,
                                  interpolate=interpolate)
        except Exception:
            # A failed English pass must not lose the Arabic-model timings that
            # are already on these words; leave them and let QA judge.
            continue
        if not segments:
            continue
        spans = [(k, k + 1) for k in range(j - i)]
        returned = map_returned([w.word for w in doc.words[i:j]], _returned_tokens(segments[0]))
        fixed += apply_aligned(doc.words[i:j], spans, returned, offset=lo)
    return fixed


def realign_window(doc: WordsDoc, wav: str | Path, i: int, j: int,
                   cfg: dict | None = None, *, pad_s: float = 2.0,
                   device: str | None = None, audio: Any = None) -> int:
    """Re-align word span `[i, j)` from its own audio window. Returns words timed.

    QA's one automatic repair pass: a window that failed inside a long segment
    gets aligned on its own, where the acoustic model is not competing with ten
    minutes of context.
    """
    cfg = cfg or {}
    acfg = cfgmod.get(cfg, "alignment") or {}
    device = device or resolve_device(cfg)
    if audio is None:
        audio = load_audio(wav)
    duration = len(audio) / float(SAMPLE_RATE)

    lo, hi = bound_run(doc.words, i, j, lo=0.0, hi=duration)
    lo, hi = max(0.0, lo - pad_s), min(duration, hi + pad_s)
    if hi - lo < 0.05:
        return 0

    clip = audio[int(lo * SAMPLE_RATE):int(hi * SAMPLE_RATE)]
    text, spans = build_align_text(doc.words[i:j])
    segments = _run_align([{"start": 0.0, "end": hi - lo, "text": text}], clip,
                          language=doc.language or "ar", device=device,
                          model_name=acfg.get("model"),
                          interpolate=acfg.get("interpolate_method", "ignore"))
    if not segments:
        return 0
    alias_tokens = [t for w in doc.words[i:j] for t in align_token(w).split()]
    return apply_aligned(doc.words[i:j], spans,
                         map_returned(alias_tokens, _returned_tokens(segments[0])),
                         offset=lo)


# -- cli ----------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Forced-align a words.json against its WAV (Stage 1 timing).")
    ap.add_argument("words_json")
    ap.add_argument("wav")
    ap.add_argument("--config", default="transcribe")
    ap.add_argument("--device", default=None, help="cpu | cuda (default: config/auto)")
    ap.add_argument("-o", "--out", default=None, help="defaults to overwriting words_json")
    args = ap.parse_args(argv)

    cfg = cfgmod.load(args.config)
    doc = WordsDoc.load(args.words_json)
    try:
        align(doc, args.wav, cfg, device=args.device)
    except AlignError as exc:
        print(f"alignment failed: {exc}", file=sys.stderr)
        return 1
    out = Path(args.out or args.words_json)
    doc.save(out)
    print(f"{out}: {len(doc.timed_words())}/{len(doc.words)} words timed "
          f"({doc.coverage() * 100:.1f}% coverage), aligner={doc.aligner}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
