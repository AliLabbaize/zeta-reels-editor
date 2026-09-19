"""Stage 1: transcription. Text only -- never timing.

The freecut pattern: several ASR backends, one contract.

    transcribe(wav: Path, cfg: dict, llm: LLM, **kw) -> WordsDoc

`BACKENDS` maps a name from `configs/transcribe.yaml` to a callable with that
signature, so swapping the transcriber is a config edit and nothing downstream
notices.

WHAT A BACKEND MAY AND MAY NOT RETURN
-------------------------------------
A backend returns words with NO start/end. Segment-level times that a model
hands back are kept in `WordsDoc.meta["segments"]` as hints for the aligner and
for locating audio excerpts, and that is all they are ever used for: an LLM's
sense of time is a guess and a guess in `words.json` becomes a cut inside a
syllable. Timing comes from `align_whisperx.align()`.

The single exception is `gemini_transcribe` with `use_backend_timings: true` in
config, which exists so the Stage 1 benchmark can measure variant (c) -- native
word timestamps -- against WhisperX. Those words are labelled `src=
"gemini_transcribe"` so that any artifact carrying model-emitted time is
identifiable after the fact.

Run standalone:
    python helpers/transcribe_gemini.py edit/audio/raw01.16k.wav --backend gemini_flash_lite
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import wave
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

try:
    from . import config as cfgmod
    from . import diff_align, textnorm, words as wordsmod
    from . import env
    from .gemini_client import LLM, LLMError, default_llm
    from .paths import EditPaths
    from .words import Word, WordsDoc
except ImportError:  # running as `python helpers/transcribe_gemini.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from helpers import config as cfgmod
    from helpers import diff_align, textnorm, words as wordsmod
    from helpers import env
    from helpers.gemini_client import LLM, LLMError, default_llm
    from helpers.paths import EditPaths
    from helpers.words import Word, WordsDoc

# Darija news delivery runs fast. Six tokens per second is the budget used to
# size the overlap comparison window; it only has to be an over-estimate.
TOKENS_PER_SECOND = 6


class TranscribeError(RuntimeError):
    pass


# -- structured output --------------------------------------------------------
#
# A JSON *schema* rather than "reply with JSON" in the prompt: with prose
# instructions a lite model eventually returns a code fence, an apology, or a
# truncated object, and the failure lands in the middle of a 10-minute batch.

SEGMENTS_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "text": {"type": "string"},
                },
                "required": ["start", "end", "text"],
            },
        }
    },
    "required": ["segments"],
}

# gemini_transcribe can return word timestamps; the schema allows them so the
# benchmark can score variant (c). They are discarded unless the config opts in.
WORDS_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "text": {"type": "string"},
                    "words": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "word": {"type": "string"},
                                "start": {"type": "number"},
                                "end": {"type": "number"},
                            },
                            "required": ["word", "start", "end"],
                        },
                    },
                },
                "required": ["start", "end", "text"],
            },
        }
    },
    "required": ["segments"],
}

SYSTEM_PROMPT = """You are a VERBATIM transcriber for Zeta, a Moroccan news channel.
The speaker talks Moroccan Darija and code-switches into French and English.

Transcribe exactly what is said:
- Darija and Arabic: Arabic script. Never Arabizi: not "l9aw", "m3a", "3la".
- French and English words: Latin script, spelled normally (Nvidia, startup, marché).
- Example of the required mix: "واحد الشركة سميتها Nvidia ربحات بزاف ديال الفلوس، donc
  الأرباح ديالها طلعو". The rule alone is ignored on real speech; the example is not.
- Keep every filler, hesitation, repetition, stutter and false start. They are the
  editorial signal this pipeline exists to find. Never clean up, never summarise,
  never translate, never reorder.
- Numbers: digits (12, 2026, 3.5). Percent as "%".
- No speaker labels, no bracketed sound events, no punctuation you did not hear.

Split the audio into short segments at natural pauses."""

USER_PROMPT = """Transcribe this audio chunk verbatim.

Timestamps are SECONDS FROM THE START OF THIS CHUNK and are only approximate
hints; forced alignment fixes them afterwards, so never stretch a segment to
make the numbers look tidy.
{vocab}"""


def _vocab_block(vocabulary: Sequence[str]) -> str:
    if not vocabulary:
        return ""
    terms = ", ".join(str(v) for v in vocabulary)
    return ("\nSpell these terms exactly this way when you hear them: " + terms + "\n")


# -- wav utilities (stdlib only: ingest always writes 16 kHz mono PCM) --------


def _open_wav(path: str | Path) -> wave.Wave_read:
    try:
        return wave.open(str(path), "rb")
    except (wave.Error, EOFError) as exc:
        raise TranscribeError(
            f"{Path(path).name} is not a readable PCM WAV. Transcription reads the "
            f"file produced by helpers/ingest.py (16 kHz mono pcm_s16le).") from exc


def wav_duration(path: str | Path) -> float:
    with _open_wav(path) as w:
        return w.getnframes() / float(w.getframerate() or 1)


def wav_bytes_per_second(path: str | Path) -> int:
    with _open_wav(path) as w:
        return w.getframerate() * w.getsampwidth() * w.getnchannels()


def slice_wav(src: str | Path, start: float, end: float, out: str | Path) -> Path:
    """Copy `[start, end)` of a PCM WAV into a new file, sample-exact."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with _open_wav(src) as w:
        rate, n = w.getframerate(), w.getnframes()
        i0 = max(0, min(n, int(round(start * rate))))
        i1 = max(i0, min(n, int(round(end * rate))))
        w.setpos(i0)
        frames = w.readframes(i1 - i0)
        nch, width = w.getnchannels(), w.getsampwidth()
    with wave.open(str(out), "wb") as o:
        o.setnchannels(nch)
        o.setsampwidth(width)
        o.setframerate(rate)
        o.writeframes(frames)
    return out


def plan_chunks(duration_s: float, chunk_s: float, overlap_s: float = 0.0
                ) -> list[tuple[float, float]]:
    """Chunk windows `[start, end)` that advance by `chunk_s - overlap_s`.

    Overlap exists so no word falls on a chunk boundary: a model handed audio
    that starts mid-word transcribes the fragment or drops it. The duplicated
    speech is removed again in `stitch_chunks`.
    """
    if duration_s <= 0:
        return []
    if chunk_s <= 0 or duration_s <= chunk_s:
        return [(0.0, duration_s)]
    step = chunk_s - max(0.0, overlap_s)
    if step <= 0:
        raise TranscribeError(f"chunk_overlap_s ({overlap_s}) must be smaller "
                              f"than chunk_s ({chunk_s})")
    out: list[tuple[float, float]] = []
    start = 0.0
    while start < duration_s:
        end = min(start + chunk_s, duration_s)
        out.append((start, end))
        if end >= duration_s:
            break
        start += step
    return out


# -- segment plumbing ---------------------------------------------------------


def _shift(segments: Iterable[dict], offset: float) -> list[dict]:
    out = []
    for s in segments:
        d = dict(s)
        d["start"] = float(s.get("start") or 0.0) + offset
        d["end"] = float(s.get("end") or 0.0) + offset
        if s.get("words"):
            d["words"] = [{**w,
                           "start": float(w.get("start") or 0.0) + offset,
                           "end": float(w.get("end") or 0.0) + offset}
                          for w in s["words"]]
        out.append(d)
    return out


def _segment_tokens(segments: Sequence[dict]) -> list[str]:
    out: list[str] = []
    for s in segments:
        out.extend(textnorm.tokenize(s.get("text", "")))
    return out


def _drop_leading_tokens(segments: Sequence[dict], n: int) -> list[dict]:
    """Remove the first `n` tokens, dropping segments that empty out."""
    out: list[dict] = []
    remaining = n
    for s in segments:
        toks = textnorm.tokenize(s.get("text", ""))
        if remaining >= len(toks):
            remaining -= len(toks)
            continue
        if remaining:
            toks = toks[remaining:]
            remaining = 0
        out.append({**s, "text": " ".join(toks)})
    return out


def overlap_token_count(tail: Sequence[str], head: Sequence[str]) -> int:
    """How many leading tokens of `head` repeat the end of `tail`.

    Exact suffix/prefix first, because two passes over identical audio usually
    do produce identical text. When they differ by a word (the model heard the
    truncated edge differently), fall back to the longest common block that
    starts near the head and reaches the end of the tail -- that block IS the
    duplicated speech, and everything before its end in `head` is duplicate too.
    """
    tail, head = list(tail), list(head)
    for k in range(min(len(tail), len(head)), 0, -1):
        if tail[-k:] == head[:k]:
            return k

    best = 0
    for a, b, size in diff_align.match_spans(tail, head):
        if size < 2:
            continue
        near_head_start = b <= 2
        reaches_tail_end = a + size >= len(tail) - 2
        if near_head_start and reaches_tail_end:
            best = max(best, b + size)
    return best


def stitch_chunks(chunks: Sequence[tuple[float, Sequence[dict]]],
                  overlap_s: float) -> list[dict]:
    """Concatenate per-chunk segments, removing the re-transcribed overlap."""
    out: list[dict] = []
    window = max(8, int(math.ceil(max(0.0, overlap_s) * TOKENS_PER_SECOND)))
    for offset, segments in chunks:
        segs = _shift(segments, offset)
        if not out:
            out.extend(segs)
            continue
        tail = _segment_tokens(out)[-window:]
        head = _segment_tokens(segs)[:window]
        drop = overlap_token_count(tail, head)
        if drop:
            segs = _drop_leading_tokens(segs, drop)
        out.extend(s for s in segs if s.get("text", "").strip())
    return out


def doc_from_segments(segments: Sequence[dict], *, src: str, **kw: Any) -> WordsDoc:
    """`words.from_segments` plus the word-index span of each segment.

    The span table is what lets `recheck_uncertain` turn "words 412..418 look
    wrong" back into "cut audio from 71.2 s to 78.9 s" without any word ever
    carrying a model timestamp.
    """
    doc = wordsmod.from_segments(segments, src=src, **kw)
    spans, i = [], 0
    for s in segments:
        n = len(textnorm.tokenize(s.get("text", "")))
        spans.append([i, i + n])
        i += n
    doc.meta["segment_word_spans"] = spans
    return doc


def approx_span_time(doc: WordsDoc, i: int, j: int) -> tuple[float, float] | None:
    """Approximate wall-clock window of word span `[i, j)` from segment hints."""
    segments = doc.meta.get("segments") or []
    spans = doc.meta.get("segment_word_spans") or []
    if not segments or len(spans) != len(segments):
        return None
    lo, hi = None, None
    for seg, (a, b) in zip(segments, spans):
        if b <= i or a >= j:
            continue
        s, e = seg.get("start"), seg.get("end")
        if s is None or e is None:
            continue
        lo = s if lo is None else min(lo, s)
        hi = e if hi is None else max(hi, e)
    return None if lo is None or hi is None or hi <= lo else (float(lo), float(hi))


# -- backends -----------------------------------------------------------------


def backend_cfg(cfg: dict, name: str) -> dict:
    block = cfgmod.get(cfg, f"backends.{name}")
    if not isinstance(block, dict):
        raise TranscribeError(
            f"no config block for backend {name!r}; known: "
            f"{sorted((cfg.get('backends') or {}))}")
    return block


def _parse_segments(payload: Any) -> list[dict]:
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict) or "segments" not in payload:
        raise TranscribeError(f"backend returned no segments: {str(payload)[:200]}")
    segs = []
    for s in payload["segments"]:
        text = (s.get("text") or "").strip()
        if not text:
            continue
        segs.append({"start": float(s.get("start") or 0.0),
                     "end": float(s.get("end") or 0.0),
                     "text": text,
                     **({"words": s["words"]} if s.get("words") else {})})
    return segs


def _gemini_chunked(
    wav: Path,
    cfg: dict,
    llm: LLM,
    *,
    backend: str,
    schema: dict,
    work_dir: Path | None = None,
) -> list[dict]:
    """Shared driver for both Gemini backends: chunk, call, stitch."""
    bcfg = backend_cfg(cfg, backend)
    chunk_s = float(bcfg.get("chunk_s") or 0)
    overlap_s = float(bcfg.get("chunk_overlap_s") or 0.0)
    duration = wav_duration(wav)
    vocab = _vocab_block(cfg.get("custom_vocabulary") or [])
    prompt = USER_PROMPT.format(vocab=vocab)
    work_dir = Path(work_dir or wav.parent / "chunks")

    results: list[tuple[float, list[dict]]] = []
    windows = plan_chunks(duration, chunk_s, overlap_s)
    for start, end in windows:
        if len(windows) == 1 and start == 0.0:
            part = wav
        else:
            # The LLM cache keys on file NAME, not content, so chunk files must
            # be named by their window or two chunks of one take collide.
            part = slice_wav(wav, start, end,
                             work_dir / f"{wav.stem}.{int(start * 1000):09d}"
                                        f"-{int(end * 1000):09d}.wav")
        payload = llm.generate(
            prompt,
            system=SYSTEM_PROMPT,
            schema=schema,
            model=bcfg.get("model"),
            files=[part],
            temperature=float(bcfg.get("temperature") or 0.0),
        )
        results.append((start, _parse_segments(payload)))

    return stitch_chunks(results, overlap_s)


def gemini_flash_lite(wav: Path, cfg: dict, llm: LLM, **kw: Any) -> WordsDoc:
    """Default text source: a lite multimodal model reading the whole WAV."""
    segments = _gemini_chunked(wav, cfg, llm, backend="gemini_flash_lite",
                               schema=SEGMENTS_SCHEMA, work_dir=kw.get("work_dir"))
    doc = doc_from_segments(segments, src="gemini", backend="gemini_flash_lite",
                            language=cfg.get("language", "ary"))
    doc.meta["model"] = backend_cfg(cfg, "gemini_flash_lite").get("model")
    return doc


def gemini_transcribe(wav: Path, cfg: dict, llm: LLM, **kw: Any) -> WordsDoc:
    """Google's dedicated ASR model, VERBATIM mode.

    Word timestamps are requested but thrown away unless `use_backend_timings`
    is set: Google's own documentation says enabling them degrades accuracy, and
    Darija coverage is unverified. This backend is a benchmark candidate.
    """
    bcfg = backend_cfg(cfg, "gemini_transcribe")
    if str(bcfg.get("mode", "VERBATIM")).upper() != "VERBATIM":
        # Hard rule 8. SMART mode strips exactly the fillers and false starts
        # that the planner uses to decide where to cut.
        raise TranscribeError("gemini_transcribe must run in VERBATIM mode")

    segments = _gemini_chunked(wav, cfg, llm, backend="gemini_transcribe",
                               schema=WORDS_SCHEMA, work_dir=kw.get("work_dir"))
    use_timings = bool(cfg.get("use_backend_timings"))
    doc = doc_from_segments(segments, src="gemini_transcribe",
                            backend="gemini_transcribe",
                            language=cfg.get("language", "ary"))
    doc.meta["model"] = bcfg.get("model")

    if use_timings:
        timed = _words_from_segment_words(segments)
        if timed:
            doc.words = timed
            doc.aligner = "gemini_transcribe:word_timestamp"
            doc.meta["timing_source"] = "backend"
            doc.meta.pop("segment_word_spans", None)  # word list no longer matches
    return doc


def _words_from_segment_words(segments: Sequence[dict]) -> list[Word]:
    out: list[Word] = []
    for s in segments:
        for w in s.get("words") or []:
            token = (w.get("word") or "").strip()
            if not token:
                continue
            out.append(Word(word=token,
                            start=float(w["start"]), end=float(w["end"]),
                            # Labelled so that a words.json carrying model-emitted
                            # time can be spotted anywhere downstream.
                            src="gemini_transcribe"))
    return out


def cohere_arabic(wav: Path, cfg: dict, llm: LLM | None = None, **kw: Any) -> WordsDoc:
    """Second-opinion ASR: `cohere-transcribe-arabic-07-2026` (open weights).

    Used to find passages where two independent models disagree, not to produce
    the shipped transcript and never to produce timing.
    """
    bcfg = backend_cfg(cfg, "cohere_arabic")
    if bcfg.get("local_weights"):
        raise TranscribeError(
            "local cohere weights are not wired up yet; unset "
            "backends.cohere_arabic.local_weights to use the API")

    try:
        import cohere  # noqa: F401  (lazy: optional extra)
    except ImportError as exc:
        raise TranscribeError(
            "cohere is not installed: uv pip install -e '.[second-opinion]'") from exc

    env.load_env()
    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        raise TranscribeError("COHERE_API_KEY is not set: put it in the "
                              "environment or in .env at the repo root "
                              "(see .env.example)")

    client = cohere.ClientV2(api_key=api_key)
    duration = wav_duration(wav)
    chunk_s = cohere_chunk_seconds(wav, bcfg)
    work_dir = Path(kw.get("work_dir") or wav.parent / "chunks")

    results: list[tuple[float, list[dict]]] = []
    for start, end in plan_chunks(duration, chunk_s, 0.0):
        part = slice_wav(wav, start, end,
                         work_dir / f"{wav.stem}.cohere.{int(start * 1000):09d}.wav")
        results.append((start, _cohere_chunk(client, part, bcfg.get("model"), end - start)))

    segments = stitch_chunks(results, 0.0)
    doc = doc_from_segments(segments, src="cohere", backend="cohere_arabic",
                            language=cfg.get("language", "ary"))
    doc.meta["model"] = bcfg.get("model")
    return doc


def cohere_chunk_seconds(wav: str | Path, bcfg: dict) -> float:
    """Chunk length that keeps every upload under the 25 MB API cap."""
    wanted = float(bcfg.get("chunk_s") or 240)
    cap_bytes = float(bcfg.get("api_max_bytes") or 26214400)
    bps = wav_bytes_per_second(wav)
    # 64 KB of headroom for the WAV header and multipart framing.
    allowed = max(1.0, (cap_bytes - 65536) / max(1, bps))
    return min(wanted, allowed)


def _cohere_chunk(client: Any, part: Path, model: str | None, duration: float) -> list[dict]:
    """Call the API and normalise whatever segment shape it returns."""
    with part.open("rb") as fh:
        resp = client.transcribe(model=model, audio=fh)
    data = resp if isinstance(resp, dict) else getattr(resp, "__dict__", {}) or {}
    segments = data.get("segments") or getattr(resp, "segments", None)
    if segments:
        return _parse_segments({"segments": [
            (s if isinstance(s, dict) else getattr(s, "__dict__", {})) for s in segments]})
    text = data.get("text") or getattr(resp, "text", "") or ""
    if not text.strip():
        raise TranscribeError(f"cohere returned nothing for {part.name}")
    return [{"start": 0.0, "end": duration, "text": text.strip()}]


BACKENDS: dict[str, Callable[..., WordsDoc]] = {
    "gemini_flash_lite": gemini_flash_lite,
    "gemini_transcribe": gemini_transcribe,
    "cohere_arabic": cohere_arabic,
}


# -- cache + dispatch ---------------------------------------------------------


def _slug(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "none"))


def cache_file(cache_dir: str | Path, sha256: str, backend: str, model: Any) -> Path:
    """One transcript per (source bytes, backend, model). Hard rule 9."""
    return Path(cache_dir) / f"{sha256[:16]}.{_slug(backend)}.{_slug(model)}.words.json"


def transcribe(
    wav: str | Path,
    cfg: dict,
    llm: LLM | None = None,
    *,
    backend: str | None = None,
    sha256: str | None = None,
    cache_dir: str | Path | None = None,
    source: dict | None = None,
    force: bool = False,
    **kw: Any,
) -> WordsDoc:
    """Run one backend, with the per-source-hash transcript cache in front."""
    wav = Path(wav)
    name = backend or cfg.get("default_backend") or "gemini_flash_lite"
    if name not in BACKENDS:
        raise TranscribeError(f"unknown backend {name!r}; known: {sorted(BACKENDS)}")
    model = cfgmod.get(cfg, f"backends.{name}.model")

    cached = None
    if sha256 and cache_dir:
        cached = cache_file(cache_dir, sha256, name, model)
        if cached.exists() and not force:
            doc = WordsDoc.load(cached)
            doc.meta["cache"] = str(cached)
            return doc

    if llm is None:
        llm = default_llm(cfgmod.get(cfg, f"backends.{name}") or {})
    doc = BACKENDS[name](wav, cfg, llm, **kw)
    if source:
        doc.source = dict(source)

    if cached is not None:
        cached.parent.mkdir(parents=True, exist_ok=True)
        doc.save(cached)
    return doc


def transcribe_source(
    source: Any,
    cfg: dict,
    edit_paths: EditPaths,
    llm: LLM | None = None,
    *,
    backend: str | None = None,
    force: bool = False,
    second_opinion: bool | None = None,
) -> WordsDoc:
    """Transcribe an `ingest.Source` and write `edit/transcripts/<name>.words.json`."""
    edit_paths.ensure()
    cache_dir = edit_paths.transcripts / ".cache"
    wav = Path(source.wav) if getattr(source, "wav", None) else None
    if wav is None:
        raise TranscribeError(f"source {source.name!r} has no extracted WAV; run ingest first")

    want_so = cfgmod.get(cfg, "second_opinion.enabled", False) if second_opinion is None \
        else second_opinion
    runner = transcribe_with_second_opinion if want_so else transcribe
    doc = runner(wav, cfg, llm, backend=backend, sha256=source.sha256,
                 cache_dir=cache_dir, source=source.words_source_block(), force=force)
    doc.save(edit_paths.words_json(source.name))
    return doc


# -- second opinion -----------------------------------------------------------


def _levenshtein(a: Sequence[str], b: Sequence[str]) -> int:
    """Token-level edit distance (rows only: spans are short)."""
    if not a:
        return len(b)
    prev = list(range(len(b) + 1))
    for i, ta in enumerate(a, 1):
        cur = [i]
        for j, tb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ta != tb)))
        prev = cur
    return prev[-1]


def token_distance(a: Sequence[str], b: Sequence[str]) -> float:
    """Normalised token distance in [0, 1]. Empty vs non-empty is total."""
    if not a and not b:
        return 0.0
    return _levenshtein(a, b) / max(len(a), len(b))


def disagreement_spans(primary: WordsDoc, other: WordsDoc,
                       threshold: float = 0.35) -> list[dict]:
    """Word spans of `primary` where the two transcripts tell different stories.

    The matching blocks between the two normalised token streams are the places
    the models agree; the gaps between consecutive blocks are candidate
    disagreements, scored by normalised edit distance so that a one-word
    spelling difference inside a long span does not flag the span.
    """
    a = [w.norm for w in primary.words]
    b = [w.norm for w in other.words]
    blocks = [blk for blk in diff_align.match_spans(a, b) if blk[2] > 0]

    out: list[dict] = []
    ai = bi = 0
    for a0, b0, size in blocks + [(len(a), len(b), 0)]:
        ga, gb = a[ai:a0], b[bi:b0]
        if ga or gb:
            dist = token_distance(ga, gb)
            if dist >= threshold:
                out.append({
                    "start_index": ai,
                    "end_index": a0,
                    "distance": round(dist, 4),
                    "primary_text": " ".join(
                        (w.display or w.word) for w in primary.words[ai:a0]),
                    "other_text": " ".join(
                        (w.display or w.word) for w in other.words[bi:b0]),
                })
        ai, bi = a0 + size, b0 + size
    return out


def mark_confirmed(primary: WordsDoc, other: WordsDoc, label: str) -> int:
    """Stamp `confirmed_by` on every primary word a second model also heard."""
    a = [w.norm for w in primary.words]
    b = [w.norm for w in other.words]
    n = 0
    for a0, _b0, size in diff_align.match_spans(a, b):
        for w in primary.words[a0:a0 + size]:
            if label not in w.confirmed_by:
                w.confirmed_by.append(label)
                n += 1
    return n


def excerpt_window(start: float, end: float, *, duration: float, margin: float,
                   min_s: float, max_s: float) -> tuple[float, float]:
    """Audio window for re-transcribing a flagged span, clamped to the file."""
    a, b = max(0.0, start - margin), min(duration, end + margin)
    length = b - a
    if length < min_s:
        grow = (min_s - length) / 2.0
        a, b = max(0.0, a - grow), min(duration, b + grow)
    if b - a > max_s:
        centre = (a + b) / 2.0
        a, b = max(0.0, centre - max_s / 2.0), min(duration, centre + max_s / 2.0)
    return (a, b)


def _replace_words(doc: WordsDoc, i: int, j: int, new: Sequence[Word]) -> None:
    """Swap word span `[i, j)` and keep the segment index table tiling.

    A segment boundary that fell INSIDE the replaced span has no meaningful
    position any more -- the words it separated are gone. It is collapsed to the
    end of the replacement so the spans still tile the word list, which is all
    `approx_span_time` needs from them.
    """
    delta = len(new) - (j - i)
    doc.words[i:j] = list(new)
    spans = doc.meta.get("segment_word_spans")
    if not spans:
        return

    def moved(index: int) -> int:
        if index <= i:
            return index
        return index + delta if index >= j else i + len(new)

    for span in spans:
        span[0], span[1] = moved(span[0]), max(moved(span[0]), moved(span[1]))


def recheck_uncertain(
    doc: WordsDoc,
    wav: str | Path,
    cfg: dict,
    llm: LLM,
    *,
    backend: str | None = None,
    work_dir: str | Path | None = None,
) -> int:
    """Re-transcribe each flagged span from a short excerpt and merge it back.

    Spans are processed last-first so that earlier indices stay valid while the
    word list changes length underneath.
    """
    spans = doc.meta.get("uncertain") or []
    if not spans:
        return 0

    so = cfg.get("second_opinion") or {}
    name = backend or cfg.get("default_backend") or "gemini_flash_lite"
    bcfg = backend_cfg(cfg, name)
    wav = Path(wav)
    duration = wav_duration(wav)
    work_dir = Path(work_dir or wav.parent / "excerpts")
    prompt = USER_PROMPT.format(vocab=_vocab_block(cfg.get("custom_vocabulary") or []))

    fixed = 0
    for span in sorted(spans, key=lambda s: s["start_index"], reverse=True):
        i, j = int(span["start_index"]), int(span["end_index"])
        approx = approx_span_time(doc, i, max(j, i + 1))
        if approx is None:
            # No time hints (e.g. backend timings were used): nothing to cut.
            span["rechecked"] = False
            span["reason"] = "no segment time hints"
            continue
        a, b = excerpt_window(
            approx[0], approx[1], duration=duration,
            margin=float(so.get("excerpt_margin_s", 1.0)),
            min_s=float(so.get("excerpt_min_s", 5.0)),
            max_s=float(so.get("excerpt_max_s", 15.0)))
        part = slice_wav(wav, a, b, work_dir / f"{wav.stem}.unc{i:06d}.wav")
        try:
            payload = llm.generate(prompt, system=SYSTEM_PROMPT, schema=SEGMENTS_SCHEMA,
                                   model=bcfg.get("model"), files=[part],
                                   temperature=float(bcfg.get("temperature") or 0.0))
            segments = _parse_segments(payload)
        except (LLMError, TranscribeError) as exc:
            span["rechecked"] = False
            span["reason"] = f"recheck failed: {exc}"
            continue

        # The excerpt carries margin on both sides, so its transcript covers more
        # than the flagged span. Keep only the part that the surrounding words do
        # not already account for by diffing against the excerpt token stream.
        tokens = _segment_tokens(segments)
        new_tokens = _trim_to_span(doc, i, j, tokens)
        replacement = [Word(word=t, src=f"{name}:recheck") for t in new_tokens]
        span["rechecked"] = True
        span["excerpt_s"] = [round(a, 3), round(b, 3)]
        span["recheck_text"] = " ".join(new_tokens)
        _replace_words(doc, i, j, replacement)
        fixed += 1

    doc.meta["uncertain_rechecked"] = fixed
    return fixed


def _trim_to_span(doc: WordsDoc, i: int, j: int, excerpt_tokens: Sequence[str],
                  context: int = 6) -> list[str]:
    """Cut the excerpt transcript down to what belongs between the neighbours.

    The words just before `i` and just after `j` are trusted (they matched both
    models). Finding them inside the excerpt gives the boundaries of the part
    that replaces the flagged span.
    """
    if not excerpt_tokens:
        return []
    norm = [textnorm.normalize_token(t) for t in excerpt_tokens]
    before = [w.norm for w in doc.words[max(0, i - context):i]]
    after = [w.norm for w in doc.words[j:j + context]]

    lo = 0
    if before:
        for a0, b0, size in diff_align.match_spans(norm, before):
            if size and b0 + size >= len(before):  # block ends at the neighbour's end
                lo = max(lo, a0 + size)
    hi = len(excerpt_tokens)
    if after:
        best = None
        for a0, b0, size in diff_align.match_spans(norm, after):
            if size and b0 == 0 and a0 >= lo:  # block starts the following context
                best = a0 if best is None else min(best, a0)
        if best is not None:
            hi = best
    return list(excerpt_tokens[lo:hi]) if hi > lo else []


def transcribe_with_second_opinion(
    wav: str | Path,
    cfg: dict,
    llm: LLM | None = None,
    *,
    backend: str | None = None,
    sha256: str | None = None,
    cache_dir: str | Path | None = None,
    source: dict | None = None,
    force: bool = False,
    **kw: Any,
) -> WordsDoc:
    """Stage 1 second-opinion flow: two backends, flag, re-transcribe, merge.

    Returns the primary transcript with `meta["uncertain"]` listing every span
    the two models disagreed on, whether it was re-checked, and what came back.
    The report reads that list; nothing else acts on it.
    """
    wav = Path(wav)
    primary = transcribe(wav, cfg, llm, backend=backend, sha256=sha256,
                         cache_dir=cache_dir, source=source, force=force, **kw)
    so = cfg.get("second_opinion") or {}
    if not so.get("enabled"):
        return primary

    other_name = so.get("backend") or "cohere_arabic"
    other = transcribe(wav, cfg, llm, backend=other_name, sha256=sha256,
                       cache_dir=cache_dir, force=force, **kw)

    mark_confirmed(primary, other, other_name.split("_")[0])
    spans = disagreement_spans(primary, other,
                               float(so.get("disagreement_threshold", 0.35)))
    primary.meta["uncertain"] = spans
    primary.meta["second_opinion"] = {
        "backend": other_name,
        "model": other.meta.get("model"),
        "spans": len(spans),
    }

    if spans and so.get("recheck_uncertain", True):
        if llm is None:
            llm = default_llm(backend_cfg(cfg, backend or cfg.get("default_backend")
                                          or "gemini_flash_lite"))
        recheck_uncertain(primary, wav, cfg, llm, backend=backend,
                          work_dir=kw.get("work_dir"))

    if cache_dir and sha256:
        # Overwrite the primary cache entry: the merged transcript, not the raw
        # first pass, is what a re-run should get back.
        primary.save(cache_file(cache_dir, sha256, backend or cfg.get("default_backend")
                                or "gemini_flash_lite",
                                primary.meta.get("model")))
    return primary


# -- cli ----------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Transcribe a 16 kHz WAV with a pluggable backend (Stage 1). "
                    "Text only: word timings come from helpers/align_whisperx.py.")
    ap.add_argument("wav", help="16 kHz mono WAV from helpers/ingest.py")
    ap.add_argument("--backend", choices=sorted(BACKENDS), default=None)
    ap.add_argument("--config", default="transcribe", help="config name under configs/")
    ap.add_argument("--second-opinion", action="store_true",
                    help="force the two-backend disagreement flow on")
    ap.add_argument("-o", "--out", default=None, help="write words.json here")
    args = ap.parse_args(argv)

    cfg = cfgmod.load(args.config)
    if args.second_opinion:
        cfg.setdefault("second_opinion", {})["enabled"] = True
    runner = transcribe_with_second_opinion if cfgmod.get(cfg, "second_opinion.enabled") \
        else transcribe
    try:
        doc = runner(Path(args.wav), cfg, None, backend=args.backend)
    except (TranscribeError, LLMError) as exc:
        print(f"transcription failed: {exc}", file=sys.stderr)
        return 1

    if args.out:
        doc.save(args.out)
        print(f"wrote {args.out}")
    print(f"{doc.backend}: {len(doc.words)} words, "
          f"{len(doc.meta.get('segments') or [])} segment hints, "
          f"{len(doc.meta.get('uncertain') or [])} uncertain spans")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


GAP_PROMPT = """This is a short cut of a Moroccan news video, taken from between two
words the transcript already has. Transcribe ONLY what is spoken in it, VERBATIM:
Darija in Arabic script, French and English in Latin, every repetition, false
start and filler kept. If nothing is spoken, return an empty text."""


def fill_loud_gaps(doc: WordsDoc, wav: str | Path, llm, *, min_gap_s: float = 0.7,
                   work_dir: str | Path | None = None) -> list[tuple[int, int]]:
    """Transcribe speech the first pass left out, and splice it into the words.

    Gemini collapses retakes: on a 6 min take, "the agents don't want to kill
    humanity..." was said twice and written once, so the second take sat in a
    word gap as untranscribed speech. Nobody could cut it, and alignment
    squeezed the words around it. Every gap the audio says is mostly speech is
    transcribed on its own; the new words go in untimed (hard rule 8: verbatim),
    and the returned spans are for the caller to align.
    """
    sil = doc.meta.get("silences")
    if sil is None:
        return []
    wav = Path(wav)
    work = Path(work_dir or wav.parent / "gaps")
    inserts: list[tuple[int, list[str]]] = []
    timed = [i for i, w in enumerate(doc.words) if w.timed]
    for i, j in zip(timed, timed[1:]):
        a, b = doc.words[i].end, doc.words[j].start
        if b - a < min_gap_s:
            continue
        quiet = sum(max(0.0, min(b, e) - max(a, s0)) for s0, e in sil)
        if quiet > 0.6 * (b - a):
            continue
        part = slice_wav(wav, a, b, work / f"gap_{a:08.2f}.wav")
        try:
            text = str(llm.generate(GAP_PROMPT, files=[part], temperature=0.0) or "")
        except Exception:
            continue
        tokens = [t for t in text.replace("\n", " ").split() if t.strip()]
        if tokens:
            inserts.append((j, tokens))
    spans: list[tuple[int, int]] = []
    shift = 0
    for j, tokens in inserts:
        at = j + shift
        doc.words[at:at] = [Word(word=t, src="gap") for t in tokens]
        spans.append((at, at + len(tokens)))
        shift += len(tokens)
    if spans:
        doc.meta["gap_fills"] = [list(s) for s in spans]
    return spans