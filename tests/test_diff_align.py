"""The one diff engine: cuts in plan mode, learned cuts in learn mode."""

from __future__ import annotations

from helpers import diff_align as D
from helpers.words import Word, WordsDoc
from conftest import make_doc


def test_kept_spans_and_cut_text():
    doc = make_doc("salam a khoya euh bghit ngoulikom chi haja mzyana")
    r = D.diff_text_against_words(doc, "salam a khoya bghit ngoulikom chi haja mzyana")
    assert [D.span_text(doc, s) for s in r.cut] == ["euh"]
    assert r.invented == []
    assert round(r.cut_ratio, 3) == round(1 / 9, 3)   # one filler out of nine words


def test_paraphrase_is_reported_not_silently_cut():
    doc = make_doc("nvidia rbhat had l3am bezzaf")
    r = D.diff_text_against_words(doc, "nvidia hit a record this year")
    assert r.invented, "a model that rewrote instead of deleting must be caught"


def test_retake_keeps_the_later_complete_attempt():
    # Ali restarts the sentence: the second attempt is the one that survives.
    doc = make_doc("bghit ngoul bghit ngoulikom chi haja")
    r = D.diff_text_against_words(doc, "bghit ngoulikom chi haja")
    kept_first_index = r.kept[0].start
    assert kept_first_index == 2, "the diff should keep the last complete take"


def test_spans_to_time_uses_word_timings_only():
    doc = make_doc("wahed jouj tlata rbaa", word_s=0.4, gap_s=0.1)
    r = D.diff_text_against_words(doc, "wahed tlata rbaa")
    times = D.spans_to_time(doc, r.kept)
    assert times[0] == (0.0, 0.4)
    assert times[1] == (1.0, 1.9)


def test_untimed_words_do_not_produce_a_range():
    doc = WordsDoc(words=[Word("wahed"), Word("jouj")])
    assert D.spans_to_time(doc, [D.Span(0, 2)]) == []


def test_learn_mode_recovers_a_planted_cut():
    raw = make_doc("wahed jouj tlata rbaa khamsa setta sebaa tmnya")
    published = make_doc("wahed jouj sebaa tmnya")
    r = D.diff_words(raw, published)
    assert [(s.start, s.end) for s in r.cut] == [(2, 6)]
    assert round(r.cut_ratio, 2) == 0.5


def test_context_around_reads_five_seconds_either_side():
    doc = make_doc("a b c d e f g", word_s=0.4, gap_s=0.1)
    before, after = D.context_around(doc, D.Span(3, 4), seconds=1.0)
    assert before.split() == ["c"] or "c" in before
    assert "e" in after


def test_long_transcripts_still_align():
    # autojunk would drop the most common tokens above 200 items and silently
    # mis-align a real 10 minute take. This is the regression guard.
    words = ["dyal" if i % 3 else f"w{i}" for i in range(600)]
    doc = make_doc(" ".join(words))
    edited = " ".join(words[:100] + words[200:])
    r = D.diff_text_against_words(doc, edited)
    assert r.invented == []
    assert sum(len(s) for s in r.cut) == 100
