"""Stage 1 backends: chunking, stitching, disagreement, cache, WER maths.

No network, no ffmpeg, no torch: audio is stdlib `wave`, the model is
`gemini_client.register_mock` under ZETA_LLM_MOCK=1.
"""

from __future__ import annotations

import json
import re
import wave

import pytest

from helpers import benchmark
from helpers import config as cfgmod
from helpers import gemini_client
from helpers import transcribe_gemini as tg
from helpers.gemini_client import LLM, LLMUnavailable
from helpers.paths import EditPaths
from helpers.words import Word, WordsDoc


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Every test in this file must fail loudly rather than reach the network."""
    monkeypatch.setenv("ZETA_LLM_MOCK", "1")
    monkeypatch.delenv("ZETA_LLM_FIXTURES", raising=False)
    yield
    gemini_client.register_mock(None)


def make_wav(path, seconds: float, rate: int = 16000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))
    return path


def chunked_cfg(chunk_s=4, overlap_s=1.0) -> dict:
    """The real configs/transcribe.yaml, with tiny chunks so a 10 s WAV splits."""
    cfg = cfgmod.load("transcribe")
    cfg["backends"]["gemini_flash_lite"].update(
        {"chunk_s": chunk_s, "chunk_overlap_s": overlap_s})
    cfg["backends"]["gemini_transcribe"].update({"chunk_s": chunk_s})
    return cfg


# -- chunk planning -----------------------------------------------------------


def test_plan_chunks_advances_by_chunk_minus_overlap():
    assert tg.plan_chunks(10.0, 4.0, 1.0) == [(0.0, 4.0), (3.0, 7.0), (6.0, 10.0)]


def test_plan_chunks_leaves_short_audio_whole():
    assert tg.plan_chunks(30.0, 600.0, 2.0) == [(0.0, 30.0)]
    assert tg.plan_chunks(0.0, 600.0, 2.0) == []


def test_plan_chunks_refuses_an_overlap_that_never_advances():
    with pytest.raises(tg.TranscribeError, match="chunk_overlap_s"):
        tg.plan_chunks(100.0, 5.0, 5.0)


def test_slice_wav_is_sample_exact(tmp_path):
    src = make_wav(tmp_path / "take.wav", 10.0)
    assert tg.wav_duration(src) == pytest.approx(10.0)
    part = tg.slice_wav(src, 3.0, 7.0, tmp_path / "part.wav")
    assert tg.wav_duration(part) == pytest.approx(4.0)
    assert tg.wav_bytes_per_second(src) == 32000


def test_a_non_pcm_file_names_the_module_that_produces_one(tmp_path):
    bad = tmp_path / "take.wav"
    bad.write_bytes(b"not a wav")
    with pytest.raises(tg.TranscribeError, match="helpers/ingest.py"):
        tg.wav_duration(bad)


# -- overlap de-duplication ---------------------------------------------------


def test_overlap_token_count_finds_an_exact_repeat():
    tail = ["واحد", "جوج", "تلاتة"]
    head = ["جوج", "تلاتة", "ربعة"]
    assert tg.overlap_token_count(tail, head) == 2


def test_overlap_token_count_tolerates_one_misheard_word():
    # The second pass heard the truncated edge differently; the block that
    # reaches the end of the tail is still the duplicated speech.
    tail = ["واحد", "جوج", "تلاتة", "ربعة"]
    head = ["جوجة", "تلاتة", "ربعة", "خمسة"]
    assert tg.overlap_token_count(tail, head) == 3


def test_overlap_token_count_is_zero_when_nothing_repeats():
    assert tg.overlap_token_count(["واحد", "جوج"], ["خمسة", "ستة"]) == 0


def test_stitch_chunks_drops_the_duplicated_speech():
    chunks = [
        (0.0, [{"start": 0.0, "end": 4.0, "text": "واحد جوج تلاتة ربعة"}]),
        (3.0, [{"start": 0.0, "end": 4.0, "text": "ربعة خمسة ستة سبعة"}]),
        (6.0, [{"start": 0.0, "end": 4.0, "text": "سبعة تمنية تسعة عشرة"}]),
    ]
    segments = tg.stitch_chunks(chunks, overlap_s=1.0)
    text = " ".join(s["text"] for s in segments)
    assert text == "واحد جوج تلاتة ربعة خمسة ستة سبعة تمنية تسعة عشرة"
    # Chunk-relative times became absolute on the way through.
    assert [s["start"] for s in segments] == [0.0, 3.0, 6.0]


def test_stitch_chunks_keeps_a_legitimate_repetition_in_one_chunk():
    # Zeta stutters on purpose; a repeat INSIDE a chunk is editorial signal.
    chunks = [(0.0, [{"start": 0.0, "end": 4.0, "text": "هاد هاد الشركة"}])]
    assert tg.stitch_chunks(chunks, 1.0)[0]["text"] == "هاد هاد الشركة"


# -- the gemini backends ------------------------------------------------------


CHUNK_TEXT = {
    0: "واحد جوج تلاتة ربعة",
    3000: "ربعة خمسة ستة سبعة",
    6000: "سبعة تمنية تسعة عشرة",
}
STITCHED = "واحد جوج تلاتة ربعة خمسة ستة سبعة تمنية تسعة عشرة"


def segments_for(req: dict) -> dict:
    """Answer a chunk request by reading the window out of the file name."""
    name = req["files"][0]
    m = re.search(r"\.(\d{9})-(\d{9})\.wav$", name)
    key = int(m.group(1)) if m else 0
    return {"segments": [{"start": 0.0, "end": 4.0, "text": CHUNK_TEXT[key]}]}


def test_gemini_flash_lite_chunks_calls_and_stitches(tmp_path):
    wav = make_wav(tmp_path / "raw01.wav", 10.0)
    seen = []

    def handler(req):
        seen.append(req)
        return segments_for(req)

    gemini_client.register_mock(handler)
    doc = tg.gemini_flash_lite(wav, chunked_cfg(), LLM(model="test"))

    assert len(seen) == 3, "one call per chunk window"
    assert doc.text() == STITCHED
    assert doc.backend == "gemini_flash_lite"
    assert doc.meta["model"] == "gemini-3.1-flash-lite"


def test_no_model_timestamp_reaches_a_word(tmp_path):
    wav = make_wav(tmp_path / "raw01.wav", 10.0)
    gemini_client.register_mock(segments_for)
    doc = tg.gemini_flash_lite(wav, chunked_cfg(), LLM(model="test"))

    assert all(not w.timed for w in doc.words), "timing comes from WhisperX only"
    assert doc.aligner is None
    # The model's sense of time survives as a hint, and only as a hint.
    assert doc.meta["segments"][0]["start"] == 0.0
    assert doc.meta["segments"][1]["start"] == 3.0


def test_segment_word_spans_index_every_word(tmp_path):
    wav = make_wav(tmp_path / "raw01.wav", 10.0)
    gemini_client.register_mock(segments_for)
    doc = tg.gemini_flash_lite(wav, chunked_cfg(), LLM(model="test"))

    spans = doc.meta["segment_word_spans"]
    assert spans[0][0] == 0 and spans[-1][1] == len(doc.words)
    assert tg.approx_span_time(doc, 0, 2) == (0.0, 4.0)
    assert tg.approx_span_time(doc, 5, 6) == (3.0, 7.0)


def test_the_prompt_carries_the_custom_vocabulary_and_verbatim_rule(tmp_path):
    wav = make_wav(tmp_path / "raw01.wav", 3.0)
    captured = {}

    def handler(req):
        captured.update(req)
        return {"segments": [{"start": 0.0, "end": 3.0, "text": "واحد"}]}

    gemini_client.register_mock(handler)
    tg.gemini_flash_lite(wav, chunked_cfg(chunk_s=600), LLM(model="test"))

    assert "VERBATIM" in captured["system"]
    assert "Nvidia" in captured["prompt"]
    assert captured["temperature"] == 0.0
    assert captured["schema"] == tg.SEGMENTS_SCHEMA, "schema, not JSON-in-prose"


def test_gemini_transcribe_word_timings_are_opt_in_and_labelled(tmp_path):
    wav = make_wav(tmp_path / "raw01.wav", 3.0)
    payload = {"segments": [{
        "start": 0.0, "end": 3.0, "text": "واحد جوج",
        "words": [{"word": "واحد", "start": 0.1, "end": 0.6},
                  {"word": "جوج", "start": 0.7, "end": 1.2}],
    }]}
    gemini_client.register_mock(lambda req: payload)

    cfg = chunked_cfg(chunk_s=600)
    cfg["use_backend_timings"] = False
    off = tg.gemini_transcribe(wav, cfg, LLM(model="test"))
    assert all(not w.timed for w in off.words)
    assert off.aligner is None

    cfg["use_backend_timings"] = True
    on = tg.gemini_transcribe(wav, cfg, LLM(model="test"))
    assert [w.start for w in on.words] == [0.1, 0.7]
    # Labelled, so an artifact carrying model-emitted time is identifiable.
    assert all(w.src == "gemini_transcribe" for w in on.words)
    assert on.aligner == "gemini_transcribe:word_timestamp"
    assert on.meta["timing_source"] == "backend"


def test_gemini_transcribe_refuses_smart_mode(tmp_path):
    wav = make_wav(tmp_path / "raw01.wav", 3.0)
    cfg = chunked_cfg()
    cfg["backends"]["gemini_transcribe"]["mode"] = "SMART"
    with pytest.raises(tg.TranscribeError, match="VERBATIM"):
        tg.gemini_transcribe(wav, cfg, LLM(model="test"))


def test_unknown_backend_names_the_known_ones(tmp_path):
    wav = make_wav(tmp_path / "raw01.wav", 1.0)
    with pytest.raises(tg.TranscribeError, match="cohere_arabic"):
        tg.transcribe(wav, chunked_cfg(), LLM(model="test"), backend="whisper")


def test_cohere_chunking_stays_under_the_api_cap(tmp_path):
    wav = make_wav(tmp_path / "raw01.wav", 5.0)
    assert tg.cohere_chunk_seconds(wav, {"chunk_s": 240, "api_max_bytes": 26214400}) == 240
    # 1 MB cap at 32 kB/s: ~30 s, and never the configured 240.
    tight = tg.cohere_chunk_seconds(wav, {"chunk_s": 240, "api_max_bytes": 1_048_576})
    assert 20 < tight < 31
    assert tight * tg.wav_bytes_per_second(wav) < 1_048_576


def test_cohere_backend_names_the_extra_when_the_package_is_missing(tmp_path):
    pytest.importorskip("builtins")
    wav = make_wav(tmp_path / "raw01.wav", 1.0)
    try:
        import cohere  # noqa: F401
    except ImportError:
        with pytest.raises(tg.TranscribeError, match=r"second-opinion"):
            tg.cohere_arabic(wav, chunked_cfg(), None)


# -- cache --------------------------------------------------------------------


def test_a_transcript_is_cached_per_source_hash(tmp_path):
    wav = make_wav(tmp_path / "raw01.wav", 10.0)
    cache = tmp_path / "cache"
    gemini_client.register_mock(segments_for)

    first = tg.transcribe(wav, chunked_cfg(), LLM(model="test"),
                          backend="gemini_flash_lite", sha256="deadbeef" * 8,
                          cache_dir=cache, source={"name": "raw01"})
    assert first.source == {"name": "raw01"}
    assert list(cache.glob("*.words.json"))

    # Hard rule 9: with the mock removed, a re-run that still succeeds can only
    # have come from the cache.
    gemini_client.register_mock(None)
    again = tg.transcribe(wav, chunked_cfg(), LLM(model="test"),
                          backend="gemini_flash_lite", sha256="deadbeef" * 8,
                          cache_dir=cache)
    assert again.text() == first.text()
    assert again.meta["cache"]

    with pytest.raises(LLMUnavailable):
        tg.transcribe(wav, chunked_cfg(), LLM(model="test"),
                      backend="gemini_flash_lite", sha256="0" * 64, cache_dir=cache)


def test_cache_key_separates_backends_and_models():
    a = tg.cache_file("/c", "a" * 64, "gemini_flash_lite", "gemini-3.5-flash-lite")
    b = tg.cache_file("/c", "a" * 64, "gemini_flash_lite", "gemini-3.1-flash-lite")
    c = tg.cache_file("/c", "a" * 64, "gemini_transcribe", "gemini-3.5-flash-lite")
    assert len({a, b, c}) == 3


# -- second opinion -----------------------------------------------------------


def two_docs() -> tuple[WordsDoc, WordsDoc]:
    primary = tg.doc_from_segments(
        [{"start": 0.0, "end": 3.0, "text": "واحد جوج تلاتة"},
         {"start": 3.0, "end": 6.0, "text": "ربعة خمسة ستة"}],
        src="gemini", backend="gemini_flash_lite")
    other = tg.doc_from_segments(
        [{"start": 0.0, "end": 6.0, "text": "واحد جوج سبعة تمنية خمسة ستة"}],
        src="cohere", backend="cohere_arabic")
    return primary, other


def test_disagreement_flags_only_the_span_that_differs():
    primary, other = two_docs()
    spans = tg.disagreement_spans(primary, other, threshold=0.35)
    assert len(spans) == 1
    span = spans[0]
    assert (span["start_index"], span["end_index"]) == (2, 4)
    assert span["primary_text"] == "تلاتة ربعة"
    assert span["other_text"] == "سبعة تمنية"
    assert span["distance"] == 1.0


def test_a_spelling_difference_is_not_a_disagreement():
    primary = tg.doc_from_segments(
        [{"start": 0.0, "end": 3.0, "text": "الشركة ديال البورصة"}], src="gemini")
    other = tg.doc_from_segments(
        [{"start": 0.0, "end": 3.0, "text": "الشركه ديال البورصه"}], src="cohere")
    # Normalisation folds taa marbuta, so these transcripts agree.
    assert tg.disagreement_spans(primary, other, 0.35) == []


def test_a_high_threshold_ignores_a_small_divergence():
    primary, other = two_docs()
    assert tg.disagreement_spans(primary, other, threshold=1.5) == []


def test_mark_confirmed_stamps_the_words_both_models_heard():
    primary, other = two_docs()
    assert tg.mark_confirmed(primary, other, "cohere") == 4
    assert [i for i, w in enumerate(primary.words) if w.confirmed_by] == [0, 1, 4, 5]
    assert primary.words[0].confirmed_by == ["cohere"]
    assert primary.words[2].confirmed_by == []
    # Idempotent: a second pass must not duplicate the label.
    tg.mark_confirmed(primary, other, "cohere")
    assert primary.words[0].confirmed_by == ["cohere"]


def test_excerpt_window_respects_margin_min_max_and_the_file_bounds():
    # Short span grown to the 5 s minimum.
    assert tg.excerpt_window(10.0, 10.4, duration=60.0, margin=1.0,
                             min_s=5.0, max_s=15.0) == pytest.approx((7.7, 12.7))
    # Long span clamped to the 15 s maximum, centred.
    a, b = tg.excerpt_window(10.0, 40.0, duration=60.0, margin=1.0,
                             min_s=5.0, max_s=15.0)
    assert b - a == pytest.approx(15.0)
    # Never past the head of the file.
    assert tg.excerpt_window(0.2, 0.6, duration=3.0, margin=1.0,
                             min_s=5.0, max_s=15.0) == pytest.approx((0.0, 3.0))


def test_recheck_replaces_only_the_flagged_words(tmp_path):
    wav = make_wav(tmp_path / "raw01.wav", 10.0)
    primary, other = two_docs()
    primary.meta["uncertain"] = tg.disagreement_spans(primary, other, 0.35)
    gemini_client.register_mock(lambda req: {"segments": [
        {"start": 0.0, "end": 7.0, "text": "واحد جوج سبعة تمنية خمسة ستة"}]})

    fixed = tg.recheck_uncertain(primary, wav, chunked_cfg(), LLM(model="test"))

    assert fixed == 1
    # The excerpt covered the neighbours too; only the flagged span moved.
    assert primary.text() == "واحد جوج سبعة تمنية خمسة ستة"
    assert primary.words[0].src == "gemini"
    assert primary.words[2].src == "gemini_flash_lite:recheck"
    span = primary.meta["uncertain"][0]
    assert span["rechecked"] is True
    assert span["recheck_text"] == "سبعة تمنية"
    assert span["excerpt_s"][0] == 0.0


def test_recheck_without_time_hints_is_reported_not_guessed(tmp_path):
    wav = make_wav(tmp_path / "raw01.wav", 10.0)
    doc = WordsDoc(words=[Word(word=t) for t in "واحد جوج تلاتة".split()])
    doc.meta["uncertain"] = [{"start_index": 1, "end_index": 2}]
    gemini_client.register_mock(lambda req: {"segments": []})

    assert tg.recheck_uncertain(doc, wav, chunked_cfg(), LLM(model="test")) == 0
    assert doc.meta["uncertain"][0]["rechecked"] is False
    assert doc.text() == "واحد جوج تلاتة"


def test_replace_words_keeps_the_segment_index_table_consistent():
    doc, _ = two_docs()
    tg._replace_words(doc, 2, 4, [Word(word=t) for t in ("سبعة", "تمنية", "تسعة")])
    spans = doc.meta["segment_word_spans"]
    assert len(doc.words) == 7
    assert spans[0][0] == 0 and spans[-1][1] == len(doc.words)
    assert spans[0][1] == spans[1][0], "segments still tile the word list"


def test_second_opinion_flow_marks_flags_and_merges(tmp_path, monkeypatch):
    wav = make_wav(tmp_path / "raw01.wav", 10.0)
    cfg = chunked_cfg(chunk_s=600)
    cfg["second_opinion"].update({"enabled": True, "backend": "cohere_arabic",
                                  "disagreement_threshold": 0.35,
                                  "recheck_uncertain": False})

    primary, other = two_docs()
    calls = []

    def fake_backend(doc):
        def run(wav_, cfg_, llm_, **kw):
            calls.append(doc.backend)
            return doc
        return run

    monkeypatch.setitem(tg.BACKENDS, "gemini_flash_lite", fake_backend(primary))
    monkeypatch.setitem(tg.BACKENDS, "cohere_arabic", fake_backend(other))

    out = tg.transcribe_with_second_opinion(wav, cfg, LLM(model="test"))

    assert calls == ["gemini_flash_lite", "cohere_arabic"]
    assert out is primary
    assert len(out.meta["uncertain"]) == 1
    assert out.meta["second_opinion"]["backend"] == "cohere_arabic"
    assert out.words[0].confirmed_by == ["cohere"]


def test_second_opinion_is_off_by_default(tmp_path, monkeypatch):
    wav = make_wav(tmp_path / "raw01.wav", 10.0)
    primary, _ = two_docs()
    monkeypatch.setitem(tg.BACKENDS, "gemini_flash_lite",
                        lambda *a, **kw: primary)
    out = tg.transcribe_with_second_opinion(wav, chunked_cfg(chunk_s=600),
                                            LLM(model="test"))
    assert "uncertain" not in out.meta


# -- benchmark maths ----------------------------------------------------------


def test_levenshtein_counts_splits_the_error_classes():
    assert benchmark.levenshtein_counts(["a", "b", "c"], ["a", "x", "c", "d"]) == (1, 0, 1)
    assert benchmark.levenshtein_counts(["a", "b", "c"], []) == (0, 3, 0)
    assert benchmark.levenshtein_counts([], ["a"]) == (0, 0, 1)
    assert benchmark.levenshtein_counts(["a"], ["a"]) == (0, 0, 0)


def test_wer_is_computed_on_normalised_tokens():
    ref = "الشركة ديال Nvidia طلعات بزاف"
    hyp = "الشركه ديال Nvidia طلعات"
    out = benchmark.wer(ref, hyp)
    # Only "بزاف" is missing: the taa-marbuta spelling is not an error.
    assert out == pytest.approx({"wer": 0.2, "substitutions": 0, "deletions": 1,
                                 "insertions": 0, "ref_words": 5, "hyp_words": 4,
                                 "scorer": out["scorer"]}, rel=1e-9)


def test_wer_agrees_with_the_builtin_scorer():
    from helpers import textnorm
    ref = "واحد جوج تلاتة ربعة خمسة"
    hyp = "واحد جوجة تلاتة خمسة ستة"
    out = benchmark.wer(ref, hyp)
    s, d, i = benchmark.levenshtein_counts(textnorm.normalize_tokens(ref),
                                           textnorm.normalize_tokens(hyp))
    assert (out["substitutions"], out["deletions"], out["insertions"]) == (s, d, i)
    assert out["wer"] == pytest.approx((s + d + i) / 5)


def test_empty_reference_does_not_divide_by_zero():
    assert benchmark.wer("", "واحد")["wer"] == 0.0


def test_benchmark_table_ranks_variants_and_writes_both_artifacts(tmp_path):
    paths = EditPaths.for_videos_dir(tmp_path).ensure()
    samples = [benchmark.Sample(name="clip1", wav=tmp_path / "clip1.wav",
                                reference="واحد جوج تلاتة ربعة")]

    texts = {"a": "واحد جوج تلاتة خمسة",      # 1 substitution
             "b": "واحد جوج تلاتة ربعة",      # perfect
             "c": "واحد جوج تلاتة ربعة",      # perfect but badly aligned
             "d": "واحد جوج ربعة"}            # 1 deletion

    def fake_transcribe(wav, cfg, backend):
        vid = fake_transcribe.variant
        doc = WordsDoc(words=[Word(word=t, start=i * 0.5, end=i * 0.5 + 0.4)
                              for i, t in enumerate(texts[vid].split())])
        if vid == "c":
            doc.words[0].start = doc.words[0].end = None   # coverage 0.75
        return doc

    results = []
    for v in benchmark.DEFAULT_VARIANTS:
        fake_transcribe.variant = v.id
        results.append(benchmark.run_variant(
            v, samples, cfgmod.load("transcribe"),
            transcribe_fn=fake_transcribe, align_fn=lambda d, w, c: d))

    assert [round(r["wer"], 3) for r in results] == [0.25, 0.0, 0.0, 0.25]
    assert results[2]["coverage"] == pytest.approx(0.75)

    # (c) ties (b) on WER but cannot clear the coverage gate.
    winner = benchmark.pick_winner(results, min_coverage=0.95)
    assert winner["id"] == "b"

    payload = benchmark.run_benchmark(
        samples, benchmark.DEFAULT_VARIANTS, cfgmod.load("transcribe"), paths,
        transcribe_fn=lambda wav, cfg, backend: fake_transcribe(wav, cfg, backend),
        align_fn=lambda d, w, c: d)
    md = (paths.edit / "benchmark.md").read_text(encoding="utf-8")
    assert "| a |" in md and "| d |" in md and "Winner" in md
    assert "default_backend:" in md
    assert json.loads((paths.edit / "benchmark.json").read_text())["winner"] == payload["winner"]


def test_manifest_reads_reference_text_from_a_file(tmp_path):
    (tmp_path / "clip1.txt").write_text("واحد جوج", encoding="utf-8")
    (tmp_path / "m.csv").write_text(
        "name,wav,reference\nclip1,clip1.wav,clip1.txt\n", encoding="utf-8")
    samples = benchmark.load_manifest(tmp_path / "m.csv")
    assert samples[0].reference.strip() == "واحد جوج"
    assert samples[0].wav == tmp_path / "clip1.wav"


# -- alignment ----------------------------------------------------------------
#
# The other half of Stage 1. whisperx/torch are not installed here, so the
# library itself is stubbed and what is under test is OUR part: the alias map,
# the segment plan, and lining the aligner's output back up with our word list.


class FakeWhisperX:
    """Places every token evenly inside its segment window."""

    def __init__(self):
        self.calls = []

    def load_align_model(self, language_code, device, model_name=None):
        return (f"model:{model_name}", {"language": language_code})

    def align(self, segments, model, metadata, audio, device,
              return_char_alignments=False, interpolate_method="nearest"):
        self.calls.append({"segments": segments, "model": model, "device": device,
                           "interpolate_method": interpolate_method,
                           "audio_len": len(audio)})
        out = []
        for seg in segments:
            toks = seg["text"].split()
            step = (seg["end"] - seg["start"]) / max(1, len(toks))
            out.append({"words": [
                {"word": t, "start": seg["start"] + k * step,
                 "end": seg["start"] + (k + 1) * step, "score": 0.9}
                for k, t in enumerate(toks)]})
        return {"segments": out}

    def load_audio(self, path):
        return [0.0] * (16000 * 10)


@pytest.fixture
def fake_whisperx(monkeypatch):
    from helpers import align_whisperx

    wx = FakeWhisperX()
    monkeypatch.setattr(align_whisperx, "_whisperx", lambda: wx)
    monkeypatch.setattr(align_whisperx, "_MODELS", align_whisperx._ModelCache())
    return wx


def test_alias_map_covers_digits_latin_and_the_custom_vocabulary():
    from helpers import align_whisperx

    m = align_whisperx.build_alias_map(cfgmod.load("transcribe"))
    assert m["nvidia"] == align_whisperx.alias_for("Nvidia", m)
    # Arabic needs no alias: the model can already emit those characters.
    assert align_whisperx.alias_for("الشركة", m) is None
    # Digits and Latin do.
    assert align_whisperx.alias_for("2026", m) == "جوج صفر جوج ستة"
    assert align_whisperx.alias_for("50%", m) == "خمسين فالمية"
    assert all(ch.isspace() or "؀" <= ch <= "ࣿ"
               for ch in align_whisperx.alias_for("startup", m))


def test_config_can_override_a_generated_alias():
    from helpers import align_whisperx

    cfg = cfgmod.load("transcribe")
    cfg["alignment"]["aliases"] = {"Nvidia": "انفيديا"}
    assert align_whisperx.alias_for("Nvidia", align_whisperx.build_alias_map(cfg)) == "انفيديا"


def test_apply_aliases_leaves_the_display_text_alone():
    from helpers import align_whisperx

    doc = WordsDoc(words=[Word(word="Nvidia"), Word(word="الشركة")])
    assert align_whisperx.apply_aliases(doc, cfgmod.load("transcribe")) == 1
    assert doc.words[0].alias and doc.words[0].display == "Nvidia"
    assert doc.words[1].alias is None


def test_map_returned_survives_a_dropped_token():
    from helpers import align_whisperx

    sent = ["واحد", "جوج", "تلاتة"]
    returned = [{"word": "واحد", "start": 0.0, "end": 0.4},
                {"word": "تلاتة", "start": 1.0, "end": 1.4}]
    mapped = align_whisperx.map_returned(sent, returned)
    assert [m["word"] if m else None for m in mapped] == ["واحد", None, "تلاتة"]


def test_a_multi_token_alias_produces_one_word_timing():
    from helpers import align_whisperx

    words = [Word(word="2026", alias="جوج صفر جوج ستة")]
    _text, spans = align_whisperx.build_align_text(words)
    aligned = [{"start": 1.0, "end": 1.2, "score": 0.8},
               {"start": 1.2, "end": 1.4, "score": 0.9},
               {"start": 1.4, "end": 1.6, "score": 0.9},
               {"start": 1.6, "end": 2.0, "score": 1.0}]
    assert align_whisperx.apply_aligned(words, spans, aligned) == 1
    assert (words[0].start, words[0].end) == (1.0, 2.0)
    assert words[0].score == pytest.approx(0.9)


def test_a_word_the_model_could_not_place_stays_untimed():
    from helpers import align_whisperx

    words = [Word(word="هاد"), Word(word="الشركة")]
    _text, spans = align_whisperx.build_align_text(words)
    align_whisperx.apply_aligned(words, spans, [{"start": 0.1, "end": 0.4}, None])
    assert words[0].timed and not words[1].timed, "no invented timestamp"


def test_align_uses_ignore_and_fills_word_timings(tmp_path, fake_whisperx):
    from helpers import align_whisperx

    doc = tg.doc_from_segments(
        [{"start": 0.0, "end": 2.0, "text": "واحد جوج"},
         {"start": 3.0, "end": 5.0, "text": "تلاتة ربعة"}],
        src="gemini", backend="gemini_flash_lite")

    out = align_whisperx.align(doc, tmp_path / "raw01.wav", cfgmod.load("transcribe"),
                               device="cpu")

    assert all(w.timed for w in out.words)
    assert out.aligner == "whisperx:jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
    assert out.meta["alignment"]["coverage"] == 1.0
    call = fake_whisperx.calls[0]
    assert call["interpolate_method"] == "ignore"
    # Segment hints anchor the windows, padded because they are approximate.
    assert call["segments"][0]["start"] == 0.0
    assert call["segments"][1]["end"] == pytest.approx(5.5)


def test_latin_spans_are_re_aligned_inside_their_neighbours_bracket(tmp_path,
                                                                    fake_whisperx):
    from helpers import align_whisperx

    doc = tg.doc_from_segments(
        [{"start": 0.0, "end": 6.0, "text": "شركة Nvidia طلعات"}],
        src="gemini", backend="gemini_flash_lite")
    align_whisperx.align(doc, tmp_path / "raw01.wav", cfgmod.load("transcribe"),
                         device="cpu")

    english = [c for c in fake_whisperx.calls if c["model"] == "model:WAV2VEC2_ASR_BASE_960H"]
    assert english, "the Latin run went to the English model"
    # The English pass only ever saw the audio between the Arabic neighbours.
    assert english[0]["audio_len"] < 16000 * 6
    nvidia = doc.words[1]
    assert doc.words[0].end <= nvidia.start and nvidia.end <= doc.words[2].start + 1e-6


def test_a_failing_english_pass_keeps_the_arabic_timings(tmp_path, fake_whisperx,
                                                         monkeypatch):
    from helpers import align_whisperx

    doc = tg.doc_from_segments(
        [{"start": 0.0, "end": 6.0, "text": "شركة Nvidia طلعات"}], src="gemini")
    real_align = fake_whisperx.align

    def flaky(segments, model, *a, **kw):
        if model == "model:WAV2VEC2_ASR_BASE_960H":
            raise RuntimeError("english model unavailable")
        return real_align(segments, model, *a, **kw)

    monkeypatch.setattr(fake_whisperx, "align", flaky)
    align_whisperx.align(doc, tmp_path / "raw01.wav", cfgmod.load("transcribe"),
                         device="cpu")
    assert all(w.timed for w in doc.words)


def test_realign_window_offsets_back_onto_the_source_timeline(tmp_path, fake_whisperx):
    from helpers import align_whisperx

    doc = tg.doc_from_segments(
        [{"start": 0.0, "end": 9.0,
          "text": "واحد جوج تلاتة ربعة خمسة ستة"}], src="gemini")
    for i, w in enumerate(doc.words):
        w.start, w.end = i * 1.5, i * 1.5 + 1.0
    doc.words[3].start = doc.words[3].end = None

    timed = align_whisperx.realign_window(doc, tmp_path / "raw01.wav", 3, 4,
                                          cfgmod.load("transcribe"), device="cpu")
    assert timed == 1
    # Inside the bracket its neighbours leave, not back at t=0.
    assert doc.words[3].start >= doc.words[2].end - 2.0


def test_segment_windows_never_claim_the_same_audio(tmp_path, fake_whisperx):
    from helpers import align_whisperx

    # 0.6 s between the segments, less than twice the 0.5 s pad: without the
    # midpoint clamp both windows would cover 2.1-2.5 and both would place a
    # word there, which QA sees as an overlap.
    doc = tg.doc_from_segments(
        [{"start": 0.0, "end": 2.0, "text": "واحد جوج"},
         {"start": 2.6, "end": 5.0, "text": "تلاتة ربعة"}], src="gemini")
    align_whisperx.align(doc, tmp_path / "raw01.wav", cfgmod.load("transcribe"),
                         device="cpu")

    windows = fake_whisperx.calls[0]["segments"]
    assert windows[0]["end"] <= windows[1]["start"] + 1e-9
    assert doc.overlaps() == []


def test_speech_in_a_word_gap_is_transcribed_and_spliced_in(tmp_path):
    # A retake the first pass collapsed sits in a gap as untranscribed speech.
    from helpers.words import Word, WordsDoc
    wav = make_wav(tmp_path / "raw01.wav", 10.0)
    doc = WordsDoc(words=[Word(word="واحد", start=1.0, end=1.4), Word(word="جوج", start=4.0, end=4.4)])
    doc.meta["silences"] = [[0.0, 0.9]]                 # the gap 1.4-4.0 is loud
    gemini_client.register_mock(lambda req: "واحد تاني")
    spans = tg.fill_loud_gaps(doc, wav, LLM(model="test"), work_dir=tmp_path / "g")
    assert spans == [(1, 3)]
    assert [w.word for w in doc.words] == ["واحد", "واحد", "تاني", "جوج"]
    assert not doc.words[1].timed                       # timing is the aligner's job
    doc.meta["silences"] = [[1.4, 4.0]]                 # a real pause: left alone
    assert tg.fill_loud_gaps(WordsDoc(words=[Word(word="a", start=1.0, end=1.4),
                                             Word(word="b", start=4.0, end=4.4)],
                                      meta={"silences": [[1.4, 4.0]]}),
                             wav, LLM(model="test"), work_dir=tmp_path / "g") == []
