"""Stage 2: the packed view the planner reads. Columns are a contract."""

from __future__ import annotations

import pytest

from helpers import pack_transcripts as pack
from helpers.paths import EditPaths
from helpers.words import Word, WordsDoc


def W(word, start, end):
    return Word(word=word, start=start, end=end)


def sample_doc() -> WordsDoc:
    """One take: Darija, a mixed phrase, a French aside, an English aside."""
    return WordsDoc(
        source={"name": "raw01"},
        words=[
            # phrase 1: starts at 1.00, so 1000 ms of lead-in silence
            W("واش", 1.00, 1.20),
            W("هاد", 1.20, 1.40),
            W("الشركة", 1.40, 1.80),
            # 620 ms pause -> break
            W("Nvidia", 2.42, 2.90),
            W("ديال", 2.90, 3.10),
            # 380 ms pause -> NOT a break (below 0.5 s)
            W("startup", 3.48, 3.90),
            # exactly 500 ms -> break (the threshold is inclusive)
            W("donc", 4.40, 4.70),
            W("le", 4.70, 4.90),
            W("marché", 4.90, 5.30),
            # 900 ms pause -> break
            W("so", 6.20, 6.40),
            W("look", 6.40, 6.70),
        ],
    )


def test_phrases_break_on_half_a_second_of_silence():
    ph = pack.phrases(sample_doc())
    assert [(round(p.start, 2), round(p.end, 2)) for p in ph] == [
        (1.00, 1.80), (2.42, 3.90), (4.40, 5.30), (6.20, 6.70)]


def test_gap_column_is_the_silence_before_the_phrase_in_ms():
    ph = pack.phrases(sample_doc())
    # First phrase: silence measured from t=0, which is what intro_trim learns from.
    assert [p.gap_before_ms for p in ph] == [1000, 620, 500, 900]


def test_lang_column_tags_code_switching():
    ph = pack.phrases(sample_doc())
    assert [p.lang for p in ph] == ["ary", "mixed", "fr", "en"]


def test_line_format_is_exactly_the_documented_one():
    ph = pack.phrases(sample_doc())
    assert pack.format_phrase(ph[0]) == "[1.00-1.80]  gap=1000ms  [ary]  واش هاد الشركة"
    assert pack.format_phrase(ph[2]) == "[4.40-5.30]  gap=500ms  [fr]  donc le marché"


def test_untimed_words_stay_in_their_phrase_without_creating_a_boundary():
    doc = sample_doc()
    doc.words.insert(2, Word(word="زعما"))  # aligner could not place it
    ph = pack.phrases(doc)
    assert len(ph) == 4
    assert "زعما" in ph[0].text
    assert ph[0].word_end == 4  # the untimed word is inside the first phrase


def test_max_words_can_force_a_break():
    ph = pack.phrases(sample_doc(), max_words=2)
    assert len(ph) > 4
    assert all(p.word_end - p.word_start <= 2 for p in ph)


def test_phrase_spans_point_back_at_the_word_list():
    doc = sample_doc()
    for p in pack.phrases(doc):
        assert doc.words[p.word_start].start == pytest.approx(p.start)
        assert doc.words[p.word_end - 1].end == pytest.approx(p.end)


def test_pack_reads_every_transcript_and_writes_takes_packed(tmp_path):
    paths = EditPaths.for_videos_dir(tmp_path).ensure()
    sample_doc().save(paths.words_json("raw01"))
    second = sample_doc()
    second.source = {"name": "raw02"}
    second.save(paths.words_json("raw02"))
    # A QA report next door must not be mistaken for a transcript.
    (paths.transcripts / "raw01.qa.json").write_text("{}", encoding="utf-8")

    text = pack.pack(paths)
    assert paths.packed.read_text(encoding="utf-8") == text
    assert "## raw01" in text and "## raw02" in text
    assert text.count("gap=620ms") == 2
    assert "gap=" in text.splitlines()[-1]


def test_pack_without_transcripts_is_an_error_not_an_empty_file(tmp_path):
    paths = EditPaths.for_videos_dir(tmp_path).ensure()
    with pytest.raises(FileNotFoundError):
        pack.pack(paths)
    assert not paths.packed.exists()
