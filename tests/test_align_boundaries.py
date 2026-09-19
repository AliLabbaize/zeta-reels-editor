"""Alignment windows split in real silence, not at the transcriber's timestamp."""
import pytest

from helpers.align_whisperx import snap_boundaries


def test_a_boundary_moves_into_the_silence_near_it():
    np = pytest.importorskip("numpy")
    sr = 16000
    # speech 0-3.2 s, silence 3.2-3.6 s, speech 3.6-6 s; the model said 4.0 s.
    audio = np.concatenate([np.ones(int(3.2 * sr)), np.zeros(int(0.4 * sr)),
                            np.ones(int(2.4 * sr))]).astype("float32")
    plan = [(0, 5, 0.0, 4.0), (5, 9, 4.0, 6.0)]
    (_, _, _, cut), (_, _, start, _) = snap_boundaries(plan, audio, 1.5, sr=sr)
    assert cut == start and 3.2 <= cut <= 3.6


def test_a_transcriber_clock_that_overruns_the_audio_is_rescaled():
    from helpers.align_whisperx import _segment_plan
    from helpers.words import Word, WordsDoc
    doc = WordsDoc(words=[Word(word=str(i)) for i in range(4)])
    doc.meta["segments"] = [{"start": 0.0, "end": 55.0}, {"start": 55.0, "end": 110.0}]
    doc.meta["segment_word_spans"] = [[0, 2], [2, 4]]
    plan = _segment_plan(doc, 100.0, 0.5)          # audio is 100 s, clock said 110
    assert plan[-1][3] <= 100.0 and plan[1][2] < 55.0


def test_auto_device_prefers_a_gpu_then_apple_then_cpu(monkeypatch):
    import types
    from helpers import align_whisperx as aw
    fake = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True)))
    monkeypatch.setitem(__import__("sys").modules, "torch", fake)
    assert aw.resolve_device({}) == "mps"
    fake.cuda.is_available = lambda: True
    assert aw.resolve_device({}) == "cuda"
    fake.cuda.is_available = lambda: False
    fake.backends.mps.is_available = lambda: False
    assert aw.resolve_device({}) == "cpu"
    assert aw.resolve_device({"alignment": {"device": "cpu"}}) == "cpu"
