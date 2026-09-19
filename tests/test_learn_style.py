"""Stage 3 tests: what a published episode teaches about the style.

The pair in `synthetic_pair` is a raw take with three known holes cut out of it,
re-timed the way a real published file would be. Recovering those holes from the
two transcripts alone is the phase-3 acceptance bar (IoU >= 0.7).
"""

from __future__ import annotations

import json

import sys

import pytest

from helpers import gemini_client, learn_style
from helpers.diff_align import Span
from helpers.learn_style import (
    CutObservation, InsertObservation, MissingExtra, Pair, PairObservation,
    build_profile, classify_cuts, cut_iou, detect_inserts, diff_pair, extract_trigger,
    label_inserts, learn, load_pairs, map_insert_triggers, write_examples,
)
from helpers.words import Word, WordsDoc

RAW_TOKENS = [
    # 0-4   hook
    "Zeta", "today", "SpaceX", "raised", "money",
    # 5-8   false start + filler, planted cut A
    "the", "fil", "yaani", "yaani",
    # 9-14  body
    "the", "filing", "says", "two", "billion", "dollars",
    # 15-19 tangent, planted cut B
    "yesterday", "I", "was", "drinking", "coffee",
    # 20-24 body
    "and", "Musk", "confirmed", "it", "yesterday",
    # 25-28 outro, planted cut C
    "ok", "that", "is", "all",
]
PLANTED = [(5, 9), (15, 20), (25, 29)]


def timed(tokens, *, start: float = 0.5, word_s: float = 0.35,
          gap_s: float = 0.25, name: str = "raw01") -> WordsDoc:
    words, t = [], start
    for tok in tokens:
        words.append(Word(word=tok, start=round(t, 3), end=round(t + word_s, 3)))
        t += word_s + gap_s
    return WordsDoc(words=words,
                    source={"name": name, "path": f"/tmp/{name}.mp4",
                            "duration_s": round(t + 0.4, 3)})


def synthetic_pair():
    """A raw doc, the published doc cut out of it, and the planted cut times."""
    raw = timed(RAW_TOKENS)
    cut_indices = {i for a, b in PLANTED for i in range(a, b)}
    kept = [w for i, w in enumerate(raw.words) if i not in cut_indices]

    published, t = [], 0.4
    for w in kept:
        published.append(Word(word=w.word, start=round(t, 3), end=round(t + w.duration, 3)))
        t += w.duration + 0.25
    pub = WordsDoc(words=published,
                   source={"name": "pub01", "path": "/tmp/pub01.mp4",
                           "duration_s": round(t + 0.4, 3)})
    truth = [raw.span_time(a, b) for a, b in PLANTED]
    return raw, pub, truth


@pytest.fixture(autouse=True)
def mock_llm(monkeypatch):
    monkeypatch.setenv("ZETA_LLM_MOCK", "1")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    yield
    gemini_client.register_mock(None)


def offline(monkeypatch):
    """No key and no mock: the learner must still work, with weaker signal."""
    monkeypatch.delenv("ZETA_LLM_MOCK", raising=False)
    gemini_client.register_mock(None)


# -------- the diff recovers the real cuts -------------------------------------


def test_diff_recovers_the_planted_cuts(monkeypatch):
    offline(monkeypatch)
    raw, pub, truth = synthetic_pair()
    cuts, word_ratio = diff_pair(raw, pub, "ep01")

    score = cut_iou([(c.start, c.end) for c in cuts], truth)
    assert score["iou"] >= 0.7, score          # spec phase-3 acceptance bar
    assert score["recall"] == 1.0
    assert word_ratio == pytest.approx(13 / len(RAW_TOKENS))


def test_recovered_cuts_carry_their_text_and_context(monkeypatch):
    offline(monkeypatch)
    raw, pub, _truth = synthetic_pair()
    cuts, _ = diff_pair(raw, pub, "ep01")
    tangent = [c for c in cuts if "coffee" in c.text][0]

    assert "drinking" in tangent.text
    assert "billion" in tangent.before and "Musk" in tangent.after
    assert not tangent.at_head and not tangent.at_tail
    assert cuts[-1].at_tail


def test_cut_iou_is_a_real_interval_metric():
    assert cut_iou([(0.0, 1.0)], [(0.0, 1.0)])["iou"] == 1.0
    assert cut_iou([(0.0, 1.0)], [(2.0, 3.0)])["iou"] == 0.0
    assert cut_iou([(0.0, 2.0)], [(1.0, 3.0)])["iou"] == pytest.approx(1 / 3, abs=1e-3)
    assert cut_iou([(0.0, 1.0)], [(0.0, 1.0), (5.0, 6.0)])["recall"] == 0.5


# -------- classification ------------------------------------------------------


def test_classification_runs_through_the_llm():
    raw, pub, _ = synthetic_pair()
    cuts, _ = diff_pair(raw, pub, "ep01")
    seen: list[dict] = []

    def handler(req):
        seen.append(req)
        return {"cuts": [{"index": i, "class": "tangent", "why": "off the story"}
                         for i in range(len(cuts))]}

    gemini_client.register_mock(handler)
    labelled = classify_cuts(cuts, raw, llm=gemini_client.LLM())

    assert [c.klass for c in labelled] == ["tangent"] * len(cuts)
    assert labelled[0].why == "off the story"
    assert "REMOVED" not in seen[0]["prompt"] and seen[0]["schema"]["required"] == ["cuts"]


def test_classification_falls_back_to_the_plan_time_heuristic(monkeypatch):
    offline(monkeypatch)
    raw, pub, _ = synthetic_pair()
    cuts, _ = diff_pair(raw, pub, "ep01")
    labelled = classify_cuts(cuts, raw, llm=gemini_client.LLM())

    assert labelled[-1].klass == "outro_trim"
    assert all(c.klass in learn_style.CUT_CLASSES for c in labelled)


def test_an_unknown_class_from_the_model_is_ignored():
    raw, pub, _ = synthetic_pair()
    cuts, _ = diff_pair(raw, pub, "ep01")
    gemini_client.register_mock(
        lambda req: {"cuts": [{"index": 0, "class": "vibes", "why": "nope"}]})
    labelled = classify_cuts(cuts, raw, llm=gemini_client.LLM())

    assert labelled[0].klass in learn_style.CUT_CLASSES and labelled[0].klass != "vibes"


# -------- inserts: vision, triggers, detection --------------------------------


def test_insert_frames_are_labelled_through_vision(tmp_path):
    frame = tmp_path / "ep01_insert_00.png"
    frame.write_bytes(b"\x89PNG\r\n")
    ins = InsertObservation(pair="ep01", start=10.0, end=14.5, frame=str(frame))
    seen: list[dict] = []

    def handler(req):
        seen.append(req)
        return {"type": "article_headline", "shows": "SpaceX raises $2B",
                "likely_source": "spacex.com", "layout": "fullframe",
                "pip_corner": "bottom_right", "visible_text": "SpaceX raises"}

    gemini_client.register_mock(handler)
    out = label_inserts([ins], llm=gemini_client.LLM())

    assert out[0].label["type"] == "article_headline"
    assert seen[0]["images"] == ["ep01_insert_00.png"]


def test_trigger_is_the_first_hard_evidence_in_the_window():
    doc = timed(["and", "SpaceX", "raised", "2", "billion"], start=9.0)
    ins = InsertObservation(pair="ep01", start=9.9, end=14.0)
    out = map_insert_triggers([ins], doc)[0]

    # A figure beats a name: the screenshot exists to show the number.
    assert out.trigger_word == "2" and out.trigger_kind == "number"
    assert out.trigger_word_index == 3
    assert out.lead_in_s == pytest.approx(0.9)


def test_trigger_window_ignores_words_more_than_a_second_away():
    doc = timed(["and", "SpaceX", "raised", "2", "billion"], start=9.0)
    ins = InsertObservation(pair="ep01", start=9.7, end=14.0)
    out = map_insert_triggers([ins], doc)[0]

    # "2" starts at 10.8, outside the -1s..+1s window around 9.7.
    assert "2" not in out.trigger_text and out.trigger_kind == "entity"


def test_trigger_falls_back_to_the_entity_when_there_is_no_number():
    doc = timed(["and", "SpaceX", "confirmed", "it"], start=0.0)
    ins = InsertObservation(pair="ep01", start=0.5, end=4.0)
    out = map_insert_triggers([ins], doc)[0]

    assert out.trigger_kind == "entity" and out.trigger_word == "SpaceX"


def test_extract_trigger_priority_order():
    mk = lambda toks: [Word(word=t, start=i, end=i + 0.2) for i, t in enumerate(toks)]
    assert extract_trigger(mk(["see", "spacex.com", "for", "2026"]))[2] == "url"
    assert extract_trigger(mk(["in", "2026", "they", "grew"]))[2] == "date"
    assert extract_trigger(mk(["just", "words", "here"])) is None


def test_scene_detection_names_the_extra_it_needs(tmp_path, monkeypatch):
    # A None entry makes the import fail, so this holds with [learn] installed too.
    monkeypatch.setitem(sys.modules, "scenedetect", None)
    with pytest.raises(MissingExtra) as exc:
        detect_inserts(tmp_path / "pub.mp4", "ep01", tmp_path / "frames")

    assert ".[learn]" in str(exc.value)


def test_published_only_fallback_uses_video_understanding(tmp_path):
    gemini_client.register_mock(lambda req: {
        "cuts": [{"start": 1.0, "end": 2.0, "kind": "filler", "why": "hesitation"}],
        "inserts": [{"start": 5.0, "end": 9.0, "type": "chart", "shows": "revenue"}],
    })
    cuts, inserts = learn_style.analyse_published(tmp_path / "pub.mp4", "ep01",
                                                  llm=gemini_client.LLM())

    assert cuts[0].klass == "filler" and inserts[0].detector == "gemini_video"
    assert inserts[0].duration == pytest.approx(4.0)


# -------- the profile ---------------------------------------------------------


def observation_with(monkeypatch) -> PairObservation:
    offline(monkeypatch)
    raw, pub, _ = synthetic_pair()
    cuts, ratio = diff_pair(raw, pub, "ep01")
    cuts = classify_cuts(cuts, raw)
    obs = PairObservation(name="ep01", has_raw=True,
                          raw_duration=raw.source["duration_s"],
                          published_duration=pub.source["duration_s"],
                          cuts=cuts, word_cut_ratio=ratio,
                          kept_gaps_ms=[g * 1000 for g in pub.gaps()[1:]],
                          published_tokens=[w.norm for w in pub.words])
    obs.inserts = [InsertObservation(
        pair="ep01", start=10.0, end=14.5, face_area=2500.0, median_face_area=25000.0,
        label={"type": "article_headline", "shows": "SpaceX raises $2B",
               "likely_source": "spacex.com", "layout": "fullframe",
               "pip_corner": "bottom_right"},
        trigger_word="two", trigger_kind="number", trigger_time=10.3,
        trigger_text="raised two billion")]
    return obs


def test_learn_defaults_to_the_edit_directory(monkeypatch, tmp_path):
    offline(monkeypatch)
    _raw, pub, _ = synthetic_pair()
    pub.save(tmp_path / "pub01.words.json")
    (tmp_path / "pub01.mp4").write_bytes(b"")

    profile = learn([Pair(published=tmp_path / "pub01.mp4")], visuals=False)

    # Hard rule 11: nothing is written outside <videos_dir>/edit/.
    assert profile["meta"]["written_to"] == str(tmp_path / "edit" / "style_profile.json")


def test_profile_has_the_shape_the_spec_defines(monkeypatch):
    profile = build_profile([observation_with(monkeypatch)])

    assert set(profile) >= {
        "cut_ratio", "median_kept_gap_ms", "max_kept_gap_ms", "filler_policy",
        "retake_policy", "intro_trim_s", "outro_trim_s", "inserts", "examples"}
    assert set(profile["filler_policy"]) == {"remove", "keep"}
    assert set(profile["inserts"]) >= {"per_minute", "median_duration_s", "lead_in_s",
                                       "layout", "pip_corner", "pip_scale", "triggers"}
    assert profile["examples"] == "few_shot_examples.md"


def test_profile_numbers_come_from_the_observations(monkeypatch):
    obs = observation_with(monkeypatch)
    profile = build_profile([obs])

    assert profile["cut_ratio"] == pytest.approx(obs.cut_ratio, abs=1e-3)
    assert profile["outro_trim_s"] > 0
    assert profile["inserts"]["median_duration_s"] == pytest.approx(4.5)
    assert profile["inserts"]["lead_in_s"] == pytest.approx(0.3)
    assert profile["inserts"]["layout"] == "fullframe_pip"
    assert profile["inserts"]["pip_scale"] == pytest.approx(0.316, abs=0.01)
    assert profile["inserts"]["triggers"][0] == "number"
    assert "headline" in profile["inserts"]["triggers"]


def test_profile_keeps_the_seed_fillers_and_adds_learned_ones(monkeypatch):
    obs = observation_with(monkeypatch)
    obs.cuts = [CutObservation(pair="ep01", span=Span(0, 1), start=1.0, end=1.4,
                               text="yaani", before="", after="", klass="filler")
                for _ in range(3)]
    profile = build_profile([obs])

    assert "yaani" in profile["filler_policy"]["remove"]
    assert "euh" in profile["filler_policy"]["remove"]      # from configs/fillers_darija.yaml


def test_a_long_pause_does_not_become_the_max_kept_gap(monkeypatch):
    obs = observation_with(monkeypatch)
    obs.kept_gaps_ms = [200.0, 380.0, 900.0, 7000.0]
    profile = build_profile([obs])

    assert profile["max_kept_gap_ms"] == 900.0
    assert profile["median_kept_gap_ms"] == 380.0


def test_the_profile_drives_the_cut_engine(monkeypatch):
    profile = build_profile([observation_with(monkeypatch)])
    doc = timed(["one", "two", "three", "four"], gap_s=0.6)
    plan = __import__("helpers.derive_cuts", fromlist=["derive"]).derive(
        doc, "one two three", profile, "raw01")

    assert plan.target_cut_ratio == profile["cut_ratio"]


# -------- examples file -------------------------------------------------------


def test_examples_file_is_small_and_concrete(monkeypatch, tmp_path):
    obs = observation_with(monkeypatch)
    path = write_examples([obs], tmp_path / "few_shot_examples.md")
    text = path.read_text(encoding="utf-8")

    assert "REMOVED:" in text and "coffee" in text
    assert "## Insert triggers" in text and "two (number)" in text
    assert "spacex.com" in text
    assert len(text) < 6000, "this file goes into every planner prompt"


# -------- pairs.csv and the end-to-end run ------------------------------------


def test_load_pairs_accepts_a_header_and_published_only_rows(tmp_path):
    (tmp_path / "pairs.csv").write_text(
        "raw,published\nraw01.mp4,pub01.mp4\n,pub02.mp4\npub03.mp4\n", encoding="utf-8")
    pairs = load_pairs(tmp_path / "pairs.csv")

    assert [p.has_raw for p in pairs] == [True, False, False]
    assert pairs[0].raw == tmp_path / "raw01.mp4"
    assert pairs[1].published == tmp_path / "pub02.mp4"
    assert pairs[2].name == "pub03"


def test_learn_writes_the_profile_and_the_examples(monkeypatch, tmp_path):
    offline(monkeypatch)
    raw, pub, _ = synthetic_pair()
    raw.save(tmp_path / "raw01.words.json")
    pub.save(tmp_path / "pub01.words.json")
    (tmp_path / "raw01.mp4").write_bytes(b"")
    (tmp_path / "pub01.mp4").write_bytes(b"")

    (tmp_path / "pairs.csv").write_text("raw01.mp4,pub01.mp4\n", encoding="utf-8")
    path = tmp_path / "edit" / "style_profile.json"
    profile = learn(tmp_path / "pairs.csv", path, visuals=False,
                    observations_path=tmp_path / "observations.json")
    observations = json.loads((tmp_path / "observations.json").read_text(encoding="utf-8"))

    assert json.loads(path.read_text(encoding="utf-8"))["cut_ratio"] == profile["cut_ratio"]
    assert (path.parent / "few_shot_examples.md").exists()
    assert observations[0]["has_raw"] and observations[0]["cuts"]
    assert profile["meta"]["pairs_with_raw"] == 1


def test_published_only_pair_still_yields_a_profile(monkeypatch, tmp_path):
    offline(monkeypatch)
    _raw, pub, _ = synthetic_pair()
    pub.save(tmp_path / "pub01.words.json")
    (tmp_path / "pub01.mp4").write_bytes(b"")

    profile = learn([Pair(published=tmp_path / "pub01.mp4")],
                    tmp_path / "style_profile.json", visuals=False)

    # No raw take: the cut ratio is unknown, not zero (None keeps a new edit to
    # fillers and repeats).
    assert profile["meta"]["cuts_observed"] == 0 and profile["cut_ratio"] is None
    assert profile["median_kept_gap_ms"] > 0      # pacing still comes through
    assert "could not be extracted" in (tmp_path / "few_shot_examples.md").read_text(
        encoding="utf-8")
