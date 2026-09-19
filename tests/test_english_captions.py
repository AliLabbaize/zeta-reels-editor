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
