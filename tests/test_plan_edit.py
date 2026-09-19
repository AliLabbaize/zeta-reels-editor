"""Stage 4 tests: the planner's answer is never trusted, only validated.

Every LLM call here goes through `gemini_client.register_mock`, with
`ZETA_LLM_MOCK=1` so that a request no mock answers fails loudly instead of
reaching the network.
"""

from __future__ import annotations

import json

import pytest

from helpers import gemini_client, plan_edit
from helpers.plan_edit import EditPlan, PlanError, confirm, locate_anchor, plan, validate
from helpers.words import Word, WordsDoc

RAW = ["Zeta", "today", "SpaceX", "raised", "two", "billion", "dollars",
       "euh", "euh", "and", "the", "filing", "is", "public", "bye"]


def mkdoc(tokens=RAW, *, word_s: float = 0.30, gap_s: float = 0.50) -> WordsDoc:
    """Widely spaced words: every boundary is a safe cut point, so the tests
    are about validation rather than about silence geometry."""
    words, t = [], 0.5
    for tok in tokens:
        words.append(Word(word=tok, start=round(t, 3), end=round(t + word_s, 3)))
        t += word_s + gap_s
    return WordsDoc(words=words,
                    source={"name": "raw01", "path": "/tmp/raw01.mp4",
                            "duration_s": round(t + 0.5, 3)})


KEPT = "Zeta today SpaceX raised two billion dollars and the filing is public"

GOOD = {
    "kept_text": KEPT,
    "inserts": [{"after_text": "SpaceX raised two", "claim": "SpaceX raised $2B",
                 "entity": "SpaceX", "prefer_source": "https://spacex.com/updates"}],
    "strategy": "Keep the hook, drop the fillers and the sign-off.",
}


@pytest.fixture(autouse=True)
def mock_llm(monkeypatch):
    monkeypatch.setenv("ZETA_LLM_MOCK", "1")
    yield
    gemini_client.register_mock(None)


def answer(payloads):
    """Serve payloads in order; record the prompts they answered."""
    seen: list[str] = []

    def handler(req):
        seen.append(req["prompt"])
        return payloads[min(len(seen) - 1, len(payloads) - 1)]

    gemini_client.register_mock(handler)
    return seen


# -------- happy path ----------------------------------------------------------


def test_plan_resolves_an_insert_to_the_right_word_index():
    doc = mkdoc()
    answer([GOOD])
    p = plan(doc, "packed", {"cut_ratio": 0.25}, source_name="raw01")

    assert p.attempts == 1
    ins = p.inserts[0]
    assert ins.trigger_word_index == RAW.index("SpaceX")
    assert ins.trigger_word == "SpaceX"
    assert ins.anchor_span == (2, 5)
    assert not ins.ambiguous


def test_the_plan_never_carries_a_timestamp():
    doc = mkdoc()
    answer([GOOD])
    blob = json.dumps(plan(doc, "packed", {"cut_ratio": 0.25},
                           source_name="raw01").to_dict()["inserts"])

    assert "start" not in blob and "time" not in blob and "second" not in blob


def test_plan_round_trips_through_edit_plan_json(tmp_path):
    doc = mkdoc()
    answer([GOOD])
    p = plan(doc, "packed", {"cut_ratio": 0.25}, source_name="raw01")
    path = p.save(tmp_path / "edit" / "edit_plan.json")
    back = EditPlan.load(path)

    assert back.kept_text == p.kept_text
    assert back.inserts[0].trigger_word_index == p.inserts[0].trigger_word_index
    assert json.loads(path.read_text(encoding="utf-8"))["schema"] == plan_edit.SCHEMA


def test_ambiguous_anchor_is_warned_not_rejected():
    doc = mkdoc(["one", "two", "the", "filing", "three", "the", "filing", "four"])
    answer([{"kept_text": "one two the filing three the filing four",
             "inserts": [{"after_text": "the filing", "claim": "c", "entity": "e"}],
             "strategy": "s"}])
    p = plan(doc, "packed", {}, source_name="raw01")

    assert p.inserts[0].trigger_word_index == 2 and p.inserts[0].ambiguous
    assert any("more than once" in w for w in p.warnings)


# -------- paraphrase ----------------------------------------------------------


def test_paraphrase_is_rejected_and_retried_once_with_the_reason():
    doc = mkdoc()
    bad = dict(GOOD, kept_text="Zeta today SpaceX secured two billion dollars")
    seen = answer([bad, GOOD])
    p = plan(doc, "packed", {"cut_ratio": 0.25}, source_name="raw01")

    assert p.attempts == 2
    assert "REJECTED" in seen[1] and "secured" in seen[1]


def test_a_planner_that_keeps_paraphrasing_fails_loudly():
    doc = mkdoc()
    bad = dict(GOOD, kept_text="Zeta today SpaceX secured two billion dollars")
    answer([bad])
    with pytest.raises(PlanError) as exc:
        plan(doc, "packed", {"cut_ratio": 0.25}, source_name="raw01")

    assert "paraphrased" in str(exc.value).lower()
    assert len(exc.value.failures) == 1


# -------- cut ratio -----------------------------------------------------------


def test_cut_ratio_outside_the_band_is_a_failure():
    doc = mkdoc()
    _, failures = validate(doc, dict(GOOD, kept_text="Zeta today SpaceX"),
                           {"cut_ratio": 0.20}, "raw01")

    assert any("cut_ratio" in f for f in failures)
    assert any("Keep more material" in f for f in failures)


def test_cut_ratio_inside_the_band_passes():
    doc = mkdoc()
    result, failures = validate(doc, GOOD, {"cut_ratio": 0.25}, "raw01")

    assert failures == [] and result is not None
    assert abs(result.cut_ratio - 0.25) <= 0.10


# -------- insert anchors ------------------------------------------------------


def test_anchor_absent_from_the_transcript_is_a_failure():
    doc = mkdoc()
    payload = dict(GOOD, inserts=[{"after_text": "the Series B round",
                                   "claim": "c", "entity": "e"}])
    _, failures = validate(doc, payload, {"cut_ratio": 0.25}, "raw01")

    assert any("not in the text you kept" in f for f in failures)


def test_anchor_inside_deleted_material_is_a_failure():
    doc = mkdoc()
    # "euh euh" was cut, so it cannot anchor a visual.
    payload = dict(GOOD, inserts=[{"after_text": "euh euh", "claim": "c", "entity": "e"}])
    _, failures = validate(doc, payload, {"cut_ratio": 0.25}, "raw01")

    assert any("not in the text you kept" in f for f in failures)


def test_one_word_anchor_is_rejected_as_too_short():
    doc = mkdoc()
    payload = dict(GOOD, inserts=[{"after_text": "SpaceX", "claim": "c", "entity": "e"}])
    _, failures = validate(doc, payload, {"cut_ratio": 0.25}, "raw01")

    assert any("too short to anchor" in f for f in failures)


def test_locate_anchor_matches_across_darija_spelling_variants():
    # The same word with and without the hamza on the alef.
    doc = mkdoc(["أول", "شركة", "قالت"])
    hits = locate_anchor(doc, [type("S", (), {"start": 0, "end": 3})()],
                         "اول شركة")

    assert hits == [(0, 2)]


def test_missing_strategy_is_a_failure():
    doc = mkdoc()
    _, failures = validate(doc, dict(GOOD, strategy=""), {"cut_ratio": 0.25}, "raw01")

    assert any("strategy" in f for f in failures)


# -------- confirmation (hard rule 10) -----------------------------------------


def _plan_obj() -> EditPlan:
    doc = mkdoc()
    result, _ = validate(doc, GOOD, {"cut_ratio": 0.25}, "raw01")
    return result


def test_auto_skips_the_confirmation(capsys):
    assert confirm(_plan_obj(), auto=True) is True
    assert "--auto" in capsys.readouterr().out


def test_interactive_confirmation_prints_the_strategy_and_waits(capsys):
    p = _plan_obj()
    assert confirm(p, input_fn=lambda _prompt: "y") is True
    assert p.strategy in capsys.readouterr().out
    assert p.confirmed is True


def test_interactive_refusal_stops_the_run():
    p = _plan_obj()
    assert confirm(p, input_fn=lambda _prompt: "") is False
    assert p.confirmed is False


# -------- prompt --------------------------------------------------------------


def test_prompt_carries_the_profile_and_the_examples():
    prompt = plan_edit.build_prompt("packed lines", {"cut_ratio": 0.31},
                                    few_shot="BEFORE/AFTER example")
    assert "0.31" in prompt and "BEFORE/AFTER example" in prompt
    assert "packed lines" in prompt


def test_packed_view_is_the_one_packed_format():
    """The prompt promises gap durations and language tags, so the planner has
    to read the same view `zeta transcribe` writes, not a lookalike."""
    from helpers import pack_transcripts

    doc = mkdoc(["one", "two"], gap_s=0.8)
    view = plan_edit.packed_view(doc)

    assert view == pack_transcripts.pack_doc(doc, silence_s=0.5)
    assert "gap=800ms" in view
    assert "[en]" in view, "the language tag has to survive into the planner's view"


def test_a_cold_start_may_cut_fillers_but_never_a_sentence():
    from helpers.diff_align import Span
    from helpers.plan_edit import restore_long_deletions
    from helpers.words import Word, WordsDoc
    doc = WordsDoc(words=[Word(word=w) for w in
                          "يعني OpenAI دارت واحد الغلط كبير بزاف euh صافي".split()])
    kept = restore_long_deletions(doc, [Span(0, 1), Span(1, 7), Span(7, 8)])
    assert kept == "OpenAI دارت واحد الغلط كبير بزاف صافي"   # fillers out, sentence back
