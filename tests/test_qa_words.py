"""The QA gate. If these pass wrongly, `--auto` renders a broken cut."""

from __future__ import annotations

import json

import pytest

from helpers import qa_words
from helpers.paths import EditPaths
from helpers.words import Word, WordsDoc

CFG = {"qa": {"min_coverage": 0.95, "max_word_duration_s": 2.0,
              "allow_overlaps": False, "realign_flagged_windows": True,
              "window_pad_s": 2.0}}


def clean_doc(n: int = 40) -> WordsDoc:
    """`n` half-second words, 100 ms apart. Nothing wrong with it."""
    return WordsDoc(
        source={"name": "raw01"},
        aligner="whisperx:test",
        words=[Word(word=f"كلمة{i}", start=i * 0.6, end=i * 0.6 + 0.5, score=0.9)
               for i in range(n)],
    )


def test_a_clean_transcript_passes():
    report = qa_words.check(clean_doc(), CFG)
    assert report["ok"] and report["failures"] == []
    assert report["coverage"] == 1.0
    assert report["overlap_count"] == 0 and report["long_word_count"] == 0


def test_overlapping_words_fail():
    doc = clean_doc()
    doc.words[10].end = doc.words[11].start + 0.2   # alignment slipped
    report = qa_words.check(doc, CFG)
    assert not report["ok"]
    assert report["overlap_count"] == 1
    assert report["overlaps"][0]["a"] == 10 and report["overlaps"][0]["b"] == 11
    assert any("overlapping" in f for f in report["failures"])


def test_a_word_longer_than_the_threshold_fails():
    doc = clean_doc()
    doc.words[5].end = doc.words[5].start + 2.4     # swallowed a pause
    report = qa_words.check(doc, CFG)
    assert not report["ok"]
    assert report["long_word_count"] == 1
    assert report["long_words"][0]["index"] == 5
    assert report["long_words"][0]["duration"] == pytest.approx(2.4)


def test_coverage_below_the_minimum_fails():
    doc = clean_doc(20)
    for w in doc.words[:3]:                          # 17/20 = 0.85
        w.start = w.end = None
    report = qa_words.check(doc, CFG)
    assert not report["ok"]
    assert report["coverage"] == pytest.approx(0.85)
    assert report["untimed_count"] == 3
    assert any("coverage" in f for f in report["failures"])


def test_an_overlap_just_inside_the_slack_is_not_a_failure():
    doc = clean_doc()
    # Aligners emit exactly-touching boundaries; 1 ms of slack is deliberate.
    doc.words[7].end = doc.words[8].start + 0.0005
    assert qa_words.check(doc, CFG)["ok"]


def test_flagged_indices_cover_every_defect_class():
    doc = clean_doc()
    doc.words[2].start = doc.words[2].end = None
    doc.words[10].end = doc.words[11].start + 0.2
    doc.words[-1].end = doc.words[-1].start + 3.0    # last word: no neighbour to overlap
    assert set(qa_words.flagged_indices(doc, CFG)) == {2, 10, 11, len(doc.words) - 1}


def test_repair_windows_merge_neighbouring_defects():
    doc = clean_doc()
    windows = qa_words.repair_windows(doc, [10, 11, 12], CFG)
    assert len(windows) == 1
    lo, hi = windows[0]
    assert lo < 10 and hi > 12


# -- the repair pass ----------------------------------------------------------


def test_qa_re_aligns_flagged_windows_once_and_passes(tmp_path):
    paths = EditPaths.for_videos_dir(tmp_path).ensure()
    doc = clean_doc(30)
    for w in doc.words[10:14]:
        w.start = w.end = None                       # 26/30 = 0.867 coverage
    calls = []

    def fake_realign(d, wav, i, j, cfg, *, pad_s=2.0):
        calls.append((i, j))
        timed = 0
        for k in range(i, j):
            if not d.words[k].timed:
                d.words[k].start = k * 0.6
                d.words[k].end = k * 0.6 + 0.5
                timed += 1
        return timed

    report = qa_words.qa(doc, tmp_path / "raw01.wav", CFG,
                         edit_paths=paths, name="raw01", realign=fake_realign)

    assert len(calls) == 1, "one automatic pass, not a retry loop"
    assert report["passes"] == 2
    assert report["ok"]
    assert report["before_repair"]["coverage"] == round(26 / 30, 4)
    assert report["realigned_windows"][0]["words_timed"] == 4


def test_qa_reports_a_failed_re_align_instead_of_crashing(tmp_path):
    paths = EditPaths.for_videos_dir(tmp_path).ensure()
    doc = clean_doc(20)
    for w in doc.words[:5]:
        w.start = w.end = None

    def boom(*a, **kw):
        raise RuntimeError("no torch here")

    report = qa_words.qa(doc, tmp_path / "raw01.wav", CFG,
                         edit_paths=paths, name="raw01", realign=boom)
    assert not report["ok"]
    assert "no torch here" in report["realigned_windows"][0]["error"]


def test_qa_writes_the_report_next_to_the_transcript(tmp_path):
    paths = EditPaths.for_videos_dir(tmp_path).ensure()
    report = qa_words.qa(clean_doc(), None, CFG, edit_paths=paths, name="raw01")
    out = paths.transcripts / "raw01.qa.json"
    assert out.exists()
    assert json.loads(out.read_text(encoding="utf-8"))["ok"] is True
    assert report["report_path"] == str(out)
    assert report["passes"] == 1, "no wav means nothing to re-align"


# -- cli ----------------------------------------------------------------------


def test_main_exits_non_zero_when_the_transcript_is_bad(tmp_path, capsys):
    paths = EditPaths.for_videos_dir(tmp_path).ensure()
    doc = clean_doc(20)
    doc.words[3].end = doc.words[4].start + 0.5
    doc.save(paths.words_json("raw01"))

    code = qa_words.main([str(paths.words_json("raw01")),
                          "--videos-dir", str(tmp_path), "--no-repair"])
    assert code == 1
    assert "FAIL" in capsys.readouterr().out


def test_main_exits_zero_on_a_clean_transcript(tmp_path):
    paths = EditPaths.for_videos_dir(tmp_path).ensure()
    clean_doc().save(paths.words_json("raw01"))
    code = qa_words.main([str(paths.words_json("raw01")),
                          "--videos-dir", str(tmp_path)])
    assert code == 0
    assert (paths.transcripts / "raw01.qa.json").exists()


def test_a_run_of_low_scores_is_flagged_but_a_lone_one_is_not():
    doc = clean_doc()
    doc.words[5].score = 0.01                      # a short function word: fine
    for i in (20, 21, 22):                         # a window the aligner lost
        doc.words[i].score = 0.0
    assert qa_words.flagged_indices(doc, CFG) == [20, 21, 22]


def test_a_passing_take_with_flagged_words_is_still_repaired(tmp_path):
    doc = clean_doc()
    doc.words[10].start = doc.words[10].end = None  # 97.5% coverage: passes
    calls = []
    qa_words.qa(doc, tmp_path / "x.wav", CFG, realign=lambda *a, **k: calls.append(a) or 1)
    assert calls, "a flagged word in a passing take must still get its repair pass"


def test_a_long_gap_over_loud_audio_is_flagged_but_a_real_pause_is_not():
    doc = clean_doc()
    for w in doc.words[11:]:                 # open a 3 s gap after word 10
        w.start += 3.0
        w.end += 3.0
    doc.meta["silences"] = []                # the audio says: talking throughout
    assert {10, 11} <= set(qa_words.flagged_indices(doc, CFG))
    doc.meta["silences"] = [[doc.words[10].end, doc.words[11].start]]
    assert qa_words.flagged_indices(doc, CFG) == []
