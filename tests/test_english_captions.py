"""English line under the Darija one: placement, and one line per cue or nothing."""
import pytest

from helpers import captions, gemini_client
from helpers.captions import Cue, CueWord
from helpers.gemini_client import LLM

CUES = [Cue(start=0, end=1, words=[CueWord("السلام", 0, .5), CueWord("عليكم", .5, 1)]),
        Cue(start=1, end=2, words=[CueWord("Hugging", 1, 1.5), CueWord("Face", 1.5, 2)])]


def test_english_takes_the_baseline_and_darija_moves_up(monkeypatch):
    monkeypatch.setattr(captions, "_english_cfg", lambda: {"mode": "under"})
    st = captions.resolve_style("9:16")
    ass = captions.render_ass(CUES[:1], st, ["Peace be upon you"])
    base = st.play_res[1] - st.margin_v
    en = next(l for l in ass.splitlines() if "Peace be upon you" in l)
    ar = next(l for l in ass.splitlines() if "السلام" in l)
    assert f"\\pos(540,{base})" in en and f"\\pos(540,{base})" not in ar


def test_a_translation_with_the_wrong_line_count_is_refused():
    gemini_client.register_mock(lambda req: {"lines": ["only one"]})
    with pytest.raises(ValueError, match="1 lines for 2 cues"):
        captions.translate_cues(CUES, LLM())


def test_english_only_drops_the_darija_burn_but_keeps_it_if_translation_failed():
    st = captions.resolve_style("9:16")
    only = captions.render_ass(CUES, st, ["Peace be upon you", "Hugging Face"])
    if captions._english_cfg().get("mode") == "only":
        assert "السلام" not in only and "Peace be upon you" in only
    assert "السلام" in captions.render_ass(CUES, st, None)   # no English: Darija ships


def test_a_long_script_is_translated_in_batches_and_a_miscount_is_retried():
    cues = [Cue(start=i, end=i + 1, words=[CueWord(f"w{i}", i, i + 1)]) for i in range(65)]
    calls = []

    def answer(req):
        n = int(req["prompt"].split("Fragments (")[1].split(")")[0])
        calls.append(n)
        short = len(calls) == 2                       # the second batch miscounts once
        return {"lines": [f"en{i}" for i in range(n - (1 if short else 0))]}

    gemini_client.register_mock(answer)
    out = captions.translate_cues(cues, LLM())
    assert len(out) == 65 and calls == [30, 30, 30, 5]


def test_a_translation_shifted_by_one_cue_is_caught_by_its_echo():
    # 30 for 30 but each line one cue late: the count passes, the echo does not.
    gemini_client.register_mock(lambda req: {"lines": [
        {"src": "Hugging Face", "en": "Hugging Face"}, {"src": "", "en": ""}]})
    with pytest.raises(ValueError, match="echoes"):
        captions.translate_cues(CUES, LLM())
    gemini_client.register_mock(lambda req: {"lines": [
        {"src": "السلام عليكم", "en": "Peace be upon you"},
        {"src": "Hugging Face.", "en": "Hugging Face"}]})
    assert captions.translate_cues(CUES, LLM()) == ["Peace be upon you", "Hugging Face"]


def test_numbering_the_model_adds_is_stripped():
    gemini_client.register_mock(lambda req: {"lines": [
        {"src": "السلام عليكم", "en": "1. Peace be upon you"},
        {"src": "Hugging Face", "en": "2) Hugging Face"}]})
    assert captions.translate_cues(CUES, LLM()) == ["Peace be upon you", "Hugging Face"]


def test_english_is_regrouped_into_readable_cues():
    from types import SimpleNamespace as N
    # The IMG_7487 failure: 0.1-0.5 s fragments, a word split across two, a line ending on "of a".
    frags = [(0.0, 0.4, "When Marx was"), (0.4, 0.9, "writing the Communist Mani-"),
             (0.9, 1.3, "-festo or Rousseau"), (1.3, 1.8, "was writing an inspection of a"),
             (1.8, 2.6, "building in New York."), (5.0, 5.2, "I mean")]
    out = captions.readable_english([N(start=s, end=e) for s, e, _ in frags], [t for *_, t in frags])
    text = " ".join(t for *_, t in out)
    assert "Manifesto" in text and "Mani-" not in text
    assert not any(captions._dangles(t) for *_, t in out)
    assert all(e - s >= 1.0 for s, e, _ in out)                     # nothing flashes by
    assert all(a[1] <= b[0] for a, b in zip(out, out[1:]))         # never overlapping
    assert out[-1][2] == "I mean"                                  # a 2.4 s pause is never merged across
    assert "\\N" in captions.break_line("they want normal people like me. Trump, folks,", 33)
    assert captions.break_line("they want normal people like me. Trump, folks,", 33).startswith(
        "they want normal people like me.")
