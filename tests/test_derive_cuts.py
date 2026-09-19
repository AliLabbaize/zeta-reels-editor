"""Behaviour tests for the cut engine.

Every case here is a rule from section 8 of the spec expressed as geometry: the
docs are synthetic so that the silence between two words is exactly the number
under test.
"""

from __future__ import annotations

import pytest

from helpers import edl
from helpers.derive_cuts import (
    MERGE_GAP_MS, CHECK_GAP_MS, ParaphraseError, _merge_close, classify_cut, derive,
)
from helpers.diff_align import Span
from helpers.words import Word, WordsDoc


def mkdoc(spec: list[tuple[str, float, float]], duration: float | None = None) -> WordsDoc:
    """`[(word, start, end)]` -> a timed WordsDoc named raw01."""
    words = [Word(word=w, start=s, end=e) for w, s, e in spec]
    dur = duration if duration is not None else words[-1].end + 0.5
    return WordsDoc(words=words,
                    source={"name": "raw01", "path": "/tmp/raw01.mp4", "duration_s": dur})


def gapped(tokens: list[str], *, word_s: float = 0.40, gap_s: float = 0.50,
           start: float = 0.50) -> list[tuple[str, float, float]]:
    """Evenly spaced words with a fixed silence between each pair."""
    out, t = [], start
    for tok in tokens:
        out.append((tok, round(t, 3), round(t + word_s, 3)))
        t += word_s + gap_s
    return out


# -------- hard rule 6: never cut inside a word -------------------------------


def test_no_range_boundary_lands_inside_a_word():
    doc = mkdoc(gapped(["one", "two", "three", "four", "five", "six"]))
    plan = derive(doc, "one two five six", {}, "raw01")

    for w in doc.words:
        covered = sum(min(r.end, w.end) - max(r.start, w.start)
                      for r in plan.ranges
                      if r.end > w.start and r.start < w.end)
        # A word is either wholly in the cut or wholly on screen; a partial
        # overlap is a cut landing mid-word.
        assert (covered == pytest.approx(0.0, abs=1e-9)
                or covered == pytest.approx(w.duration, abs=1e-9)), w.word


def test_kept_words_survive_and_cut_words_do_not():
    doc = mkdoc(gapped(["alpha", "beta", "gamma", "delta"]))
    plan = derive(doc, "alpha delta", {}, "raw01")
    inside = lambda w: any(r.start <= w.start and r.end >= w.end for r in plan.ranges)

    assert inside(doc.words[0]) and inside(doc.words[3])
    assert not inside(doc.words[1]) and not inside(doc.words[2])


# -------- hard rule 7: padding ------------------------------------------------


def test_default_padding_is_140_before_and_100_after():
    doc = mkdoc(gapped(["one", "two", "three", "four"]))
    plan = derive(doc, "one four", {}, "raw01")

    assert plan.ranges[0].start == pytest.approx(doc.words[0].start - 0.140)
    assert plan.ranges[0].end == pytest.approx(doc.words[0].end + 0.100)
    assert plan.ranges[1].start == pytest.approx(doc.words[3].start - 0.140)


def test_padding_never_swallows_a_neighbouring_word():
    # 150 ms of silence at both cut edges: usable, but tighter than the 130 ms
    # of pad the two sides want between them.
    doc = mkdoc(gapped(["one", "two", "three", "four"], gap_s=0.150))
    plan = derive(doc, "one four", {}, "raw01")
    removed = doc.words[1:3]

    for r in plan.ranges:
        for w in removed:
            assert r.end <= w.start + 1e-9 or r.start >= w.end - 1e-9
    assert plan.ranges[0].end <= removed[0].start
    assert plan.ranges[1].start >= removed[-1].end
    # 150-400 ms is allowed but never silently: the caller runs timeline_view.
    assert plan.needs_visual_check and plan.cuts[0].needs_visual_check


def test_padding_stays_inside_the_source_bounds():
    doc = mkdoc([("one", 0.02, 0.40), ("two", 0.90, 1.30), ("three", 1.80, 2.20)],
                duration=2.24)
    plan = derive(doc, "one three", {}, "raw01")

    assert plan.ranges[0].start == pytest.approx(0.0)      # clamped at the head
    assert plan.ranges[-1].end == pytest.approx(2.24)      # clamped at the tail


def test_profile_padding_is_clamped_to_the_30_200_ms_window():
    doc = mkdoc(gapped(["one", "two", "three"], gap_s=1.0))
    plan = derive(doc, "one three", {"cuts": {"pad_before_ms": 5, "pad_after_ms": 900}},
                  "raw01")

    assert plan.ranges[0].end == pytest.approx(doc.words[0].end + 0.200)
    assert plan.ranges[1].start == pytest.approx(doc.words[2].start - 0.030)


# -------- silence rule --------------------------------------------------------


def test_a_filler_with_no_silence_around_it_stays_rather_than_eating_neighbours():
    doc = mkdoc([("wa", 0.50, 0.90), ("hello", 1.40, 1.80), ("umm", 1.90, 2.20),
                 ("world", 2.30, 2.70), ("ok", 3.20, 3.60)], duration=4.0)
    plan = derive(doc, "wa hello world ok", {}, "raw01")

    # The old rule walked out into the 500 ms gaps and deleted "hello" and
    # "world" too. Ali's words outrank a filler: nothing is cut.
    assert plan.cuts == []
    for r in plan.ranges:
        for edge in (r.start, r.end):
            assert not (1.80 < edge < 1.90) and not (2.20 < edge < 2.30)


def test_removal_with_no_usable_silence_anywhere_is_restored():
    doc = mkdoc([("a", 0.00, 0.30), ("b", 0.32, 0.60), ("c", 0.62, 0.70),
                 ("d", 0.72, 1.00), ("e", 1.02, 1.30)], duration=1.40)
    plan = derive(doc, "a b d e", {}, "raw01")

    assert plan.cuts == []
    assert len(plan.ranges) == 1
    assert plan.dropped and "c" in plan.dropped[0].text


# -------- merging -------------------------------------------------------------


def test_ranges_closer_than_120ms_are_merged():
    padded = [(0.0, 1.0), (1.05, 2.0), (2.5, 3.0)]
    assert _merge_close(padded) == [[0, 1], [2]]


def test_ranges_120ms_apart_are_left_alone():
    padded = [(0.0, 1.0), (1.12, 2.0)]
    assert _merge_close(padded) == [[0], [1]]
    assert (1.12 - 1.0) * 1000 >= MERGE_GAP_MS


def test_merged_ranges_keep_the_material_between_them():
    doc = mkdoc([("a", 0.00, 0.30), ("b", 0.32, 0.60), ("c", 0.62, 0.70),
                 ("d", 0.72, 1.00)], duration=1.10)
    plan = derive(doc, "a b d", {}, "raw01")

    assert len(plan.ranges) == 1
    assert plan.ranges[0].start <= doc.words[2].start
    assert plan.ranges[0].end >= doc.words[2].end


# -------- cut ratio -----------------------------------------------------------


def test_cut_ratio_is_reported_against_the_profile_band():
    doc = mkdoc(gapped(["one", "two", "three", "four"], word_s=0.40, gap_s=0.50,
                       start=0.0), duration=3.10)
    plan = derive(doc, "one two", {"cut_ratio": 0.30}, "raw01")

    kept = plan.ranges[0].duration
    assert plan.cut_ratio == pytest.approx(1.0 - kept / 3.10)
    assert plan.cut_ratio_delta == pytest.approx(plan.cut_ratio - 0.30)
    assert plan.within_band is False          # reported, never forced
    assert plan.word_cut_ratio == pytest.approx(0.5)


def test_cut_ratio_inside_the_band_passes():
    doc = mkdoc(gapped(["one", "two", "three", "four", "five", "six"], start=0.0),
                duration=5.30)
    plan = derive(doc, "one two three four five", {"cut_ratio": 0.15}, "raw01")

    assert abs(plan.cut_ratio_delta) <= 0.10
    assert plan.within_band is True


# -------- paraphrase ----------------------------------------------------------


def test_paraphrase_raises_instead_of_cutting():
    doc = mkdoc(gapped(["the", "company", "raised", "money"]))
    with pytest.raises(ParaphraseError) as exc:
        derive(doc, "the company secured money", {}, "raw01")

    assert "secured" in exc.value.invented


def test_reordering_kept_words_is_treated_as_paraphrase():
    doc = mkdoc(gapped(["alpha", "beta", "gamma"]))
    with pytest.raises(ParaphraseError):
        derive(doc, "gamma beta alpha", {}, "raw01")


# -------- classification and downstream contract ------------------------------


def test_head_and_tail_removals_are_trims():
    doc = mkdoc(gapped(["intro", "hook", "body", "bye"]))
    plan = derive(doc, "hook body", {}, "raw01")

    assert [c.klass for c in plan.cuts] == ["intro_trim", "outro_trim"]


def test_filler_only_removal_is_classified_from_the_seed_config():
    doc = mkdoc([("الشركة", 0.50, 0.90),
                 ("يعني", 1.40, 1.80),
                 ("قالت", 2.30, 2.70)], duration=3.2)
    plan = derive(doc, "الشركة قالت",
                  {}, "raw01")

    assert plan.cuts[0].klass == "filler"


def test_false_start_is_recognised():
    doc = mkdoc(gapped(["so", "the", "comp", "the", "company", "grew"]))
    # "so the comp- the company grew": the removal is the aborted restart.
    assert classify_cut(doc, Span(1, 3), set())[0] == "false_start"


def test_plan_ranges_build_a_valid_edl():
    doc = mkdoc(gapped(["one", "two", "three", "four", "five"]))
    plan = derive(doc, "one three five", {}, "raw01")

    built = edl.build({"raw01": "/tmp/raw01.mp4"}, plan.ranges).validate()
    assert built.total_duration_s == pytest.approx(plan.kept_duration)
    assert plan.range_of_word(0) == 0 and plan.range_of_word(1) is None


def test_plan_serialises_for_the_report():
    doc = mkdoc(gapped(["one", "two", "three"]))
    d = derive(doc, "one three", {"cut_ratio": 0.2}, "raw01").to_dict()

    assert d["source"] == "raw01" and d["cuts"][0]["class"]
    assert set(d) >= {"ranges", "cuts", "cut_ratio", "cut_ratio_delta", "within_band"}


def test_a_long_pause_inside_kept_speech_is_cut_but_its_words_stay():
    doc = mkdoc([("واحد", 0.5, 0.9), ("جوج", 1.0, 1.4), ("تلاتة", 3.4, 3.8), ("ربعة", 3.9, 4.3)])
    doc.meta["silences"] = [[1.4, 3.4]]
    plan = derive(doc, "واحد جوج تلاتة ربعة", {}, "raw01")
    assert [(r.start, r.end) for r in plan.ranges] == [
        pytest.approx((0.36, 1.5)), pytest.approx((3.26, 4.4))]   # 2 s of air gone
    assert len(derive(doc, "واحد جوج تلاتة ربعة", {"cuts": {"max_pause_ms": 0}},
                      "raw01").ranges) == 1                      # 0 disables it


def test_a_gap_the_audio_says_is_speech_is_never_cut():
    # Misaligned words leave a timing gap over continuous speech.
    doc = mkdoc([("واحد", 0.5, 0.9), ("جوج", 1.0, 1.4), ("تلاتة", 3.4, 3.8), ("ربعة", 3.9, 4.3)])
    doc.meta["silences"] = [[1.4, 1.6]]
    assert len(derive(doc, "واحد جوج تلاتة ربعة", {}, "raw01").ranges) == 1
