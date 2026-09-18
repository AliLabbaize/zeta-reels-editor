"""Self evaluation: arithmetic, signal and image checks over a rendered cut."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from helpers import captions, gemini_client, self_eval
from helpers.edl import EDL, Overlay, Range
from helpers.paths import EditPaths
from helpers.words import Word, WordsDoc


@pytest.fixture
def edl():
    # Output: 0-4s (range 0), 4-9s (range 1). One cut, at 4.0s.
    return EDL(
        sources={"raw01": "/tmp/raw01.mp4"},
        ranges=[Range("raw01", 1.0, 5.0), Range("raw01", 20.0, 25.0)],
        subtitles="edit/captions/final.ass",
    )


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setenv("ZETA_LLM_MOCK", "1")
    gemini_client.register_mock(None)
    yield
    gemini_client.register_mock(None)


# -------- arithmetic ---------------------------------------------------------


def test_duration_mismatch_is_an_error(edl):
    ok = self_eval.check_duration(edl, "x.mp4", probed=9.02)
    assert [f.severity for f in ok] == ["info"]

    bad = self_eval.check_duration(edl, "x.mp4", probed=7.5)
    assert bad[0].severity == "error"
    assert bad[0].data["expected_s"] == pytest.approx(9.0)
    assert "9.000" in bad[0].message


def test_duration_not_checked_without_ffprobe(edl, monkeypatch):
    monkeypatch.setattr(self_eval, "probe_duration", lambda video: None)
    findings = self_eval.check_duration(edl, "x.mp4")
    assert findings[0].checked is False
    assert findings[0].severity == "warning"


def test_overlay_past_the_end_is_flagged(edl):
    edl.overlays = [Overlay("edit/screenshots/slot_01/overlay.mp4", 7.0, 4.0,
                            meta={"verified": True, "url": "https://x/1"})]
    findings = self_eval.check_overlay_windows(edl)
    assert [f.check for f in findings] == ["overlay_past_end"]
    assert findings[0].severity == "error"
    assert findings[0].data["end_s"] == pytest.approx(11.0)


def test_overlay_spanning_a_cut_is_flagged(edl):
    edl.overlays = [Overlay("slot_01/overlay.mp4", 3.0, 2.5)]
    findings = self_eval.check_overlay_windows(edl)
    assert [f.check for f in findings] == ["overlay_crosses_cut"]
    assert findings[0].data["boundaries"] == [4.0]


def test_overlay_inside_one_segment_is_clean(edl):
    edl.overlays = [Overlay("slot_01/overlay.mp4", 4.5, 3.0)]
    assert self_eval.check_overlay_windows(edl) == []


def test_caption_fully_inside_a_fullframe_insert_is_an_error(edl):
    edl.overlays = [Overlay("slot_01/overlay.mp4", 4.0, 4.0,
                            meta={"layout": "fullframe_pip", "verified": True,
                                  "url": "https://x/1"})]
    cues = [self_eval.CueWindow(5.0, 6.0, "الشركة قالت")]
    findings = self_eval.check_captions_under_overlays(cues, edl)
    hidden = [f for f in findings if f.check == "caption_under_overlay"]
    assert len(hidden) == 1
    assert hidden[0].severity == "error"
    assert "hidden" in hidden[0].message


def test_caption_under_a_card_insert_is_only_informational(edl):
    # fit_card anchors the card above the caption band, so it does not hide it.
    edl.overlays = [Overlay("slot_01/overlay.mp4", 4.0, 4.0, meta={"layout": "fit_card"})]
    cues = [self_eval.CueWindow(5.0, 6.0, "الشركة قالت")]
    findings = self_eval.check_captions_under_overlays(cues, edl)
    assert [f.severity for f in findings] == ["info"]


def test_missing_subtitles_with_overlays_is_an_error(edl):
    edl.subtitles = None
    edl.overlays = [Overlay("slot_01/overlay.mp4", 4.0, 2.0, meta={"layout": "fit_card"})]
    findings = self_eval.check_captions_under_overlays([], edl)
    assert any(f.check == "captions_missing" and f.severity == "error" for f in findings)


def test_view_points_cover_every_cut_and_insert_edge(edl):
    edl.overlays = [Overlay("slot_01/overlay.mp4", 5.0, 2.0)]
    points = dict((label, (a, b)) for label, a, b in self_eval.view_points(edl))
    assert points["cut_00"] == (2.5, 5.5)          # the cut at 4.0s, +/- 1.5s
    assert points["insert_00_in"] == (3.5, 6.5)
    assert points["insert_00_out"] == (5.5, 8.5)


def test_view_points_are_clamped_to_the_timeline(edl):
    edl.overlays = [Overlay("slot_01/overlay.mp4", 0.5, 8.0)]
    for _label, a, b in self_eval.view_points(edl):
        assert a >= 0.0 and b <= edl.total_duration_s


# -------- captions read back from the burned file ----------------------------


def test_parse_ass_round_trips_what_was_burned(edl, tmp_path):
    doc = WordsDoc(source={"name": "raw01"}, words=[
        Word(word="الشركة", start=1.0, end=1.3), Word(word="قالت", start=1.4, end=1.7)])
    paths = EditPaths.for_videos_dir(tmp_path)
    out = captions.write_captions(doc, edl, paths, aspect="9:16")
    cues = self_eval.parse_ass(out["ass"])
    assert len(cues) == 1
    assert cues[0].text == "الشركة قالت"          # override tags stripped
    assert cues[0].start == pytest.approx(0.0, abs=0.01)
    assert cues[0].end == pytest.approx(0.7, abs=0.01)


# -------- audio --------------------------------------------------------------


def _tone(seconds=8.0, sr=16000, freq=220.0):
    t = np.arange(int(seconds * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype("float32"), sr


def test_find_pops_ignores_a_clean_boundary():
    samples, sr = _tone()
    assert self_eval.find_pops(samples, sr, [4.0]) == []


def test_find_pops_catches_a_click_inside_the_fade(edl):
    samples, sr = _tone()
    samples = samples.copy()
    samples[int(4.0 * sr)] = 0.95  # a one-sample step: exactly what a bad splice sounds like
    pops = self_eval.find_pops(samples, sr, [4.0])
    assert len(pops) == 1
    assert pops[0]["t_output"] == pytest.approx(4.0)
    # A click 100 ms away from the boundary is someone's plosive, not our cut.
    assert self_eval.find_pops(samples, sr, [4.5]) == []


def test_check_audio_pops_reports_the_boundary(edl):
    samples, sr = _tone(seconds=9.0)
    samples = samples.copy()
    samples[int(4.0 * sr)] = 0.95
    findings = self_eval.check_audio_pops("x.mp4", edl, samples=samples, sample_rate=sr)
    assert findings[0].severity == "error"
    assert findings[0].t_output == pytest.approx(4.0)
    assert "30 ms" in findings[0].message


def test_check_audio_pops_degrades_without_ffmpeg(edl, monkeypatch):
    monkeypatch.setattr(self_eval, "extract_pcm", lambda *a, **k: (None, 16000))
    findings = self_eval.check_audio_pops("x.mp4", edl)
    assert findings[0].checked is False and findings[0].severity == "warning"


# -------- image checks -------------------------------------------------------


def test_vision_checks_degrade_to_not_checked(edl, tmp_path):
    png = tmp_path / "cut_00.png"
    png.write_bytes(b"\x89PNG\r\n")
    # ZETA_LLM_MOCK=1 with no handler registered: available, but nothing answers.
    findings = self_eval.check_visual_jumps({"cut_00": png}, edl, gemini_client.LLM())
    assert [f.checked for f in findings] == [False]
    assert findings[0].severity == "info"


def test_vision_flags_an_insert_showing_the_wrong_frames(edl, tmp_path):
    png = tmp_path / "insert_00_in.png"
    png.write_bytes(b"\x89PNG\r\n")
    edl.overlays = [Overlay("slot_01/overlay.mp4", 4.5, 3.0,
                            meta={"claim": "SpaceX IPO filing", "url": "https://x/1",
                                  "verified": True, "layout": "fullframe_pip"})]
    gemini_client.register_mock(lambda req: {"shows_claim": False, "caption_visible": False,
                                             "evidence": "the frame shows a cookie banner"})
    findings = self_eval.check_insert_frames({"insert_00_in": png}, edl, gemini_client.LLM())
    checks = {f.check: f for f in findings}
    assert checks["insert_wrong_frames"].severity == "error"
    assert "cookie banner" in checks["insert_wrong_frames"].message
    assert checks["caption_under_overlay"].severity == "error"


def test_timeline_view_failure_is_a_warning_not_a_crash(edl, tmp_path, monkeypatch):
    monkeypatch.setattr(self_eval, "run_timeline_view", lambda *a, **k: None)
    views, findings = self_eval.render_views("missing.mp4", edl, tmp_path / "verify")
    assert views == {}
    assert findings and all(f.check == "timeline_view" and not f.checked for f in findings)


# -------- the loop -----------------------------------------------------------


def test_loop_stops_at_three_fixes_and_flags_the_rest():
    seen = {"evals": 0, "fixes": 0}

    def evaluate(i):
        seen["evals"] += 1
        return [self_eval.Finding("duration", "error", "still short")]

    def fix(errors, i):
        seen["fixes"] += 1
        return True

    result = self_eval.run_loop(evaluate, fix)
    assert seen["fixes"] == self_eval.MAX_PASSES == 3
    assert seen["evals"] == 4          # the loop always ends on an evaluation
    assert result.flagged is True
    assert result.findings[0].data["unresolved_after_passes"] == 3


def test_loop_stops_as_soon_as_it_is_clean():
    calls = {"n": 0}

    def evaluate(i):
        calls["n"] += 1
        return [] if calls["n"] > 1 else [self_eval.Finding("duration", "error", "short")]

    result = self_eval.run_loop(evaluate, lambda errors, i: True)
    assert calls["n"] == 2 and result.fixes == 1 and result.flagged is False


def test_loop_stops_when_the_fix_gives_up():
    result = self_eval.run_loop(lambda i: [self_eval.Finding("x", "error", "bad")],
                                lambda errors, i: False)
    assert result.passes == 1 and result.fixes == 0 and result.flagged is True


# -------- output -------------------------------------------------------------


def test_write_self_eval_lands_in_verify(tmp_path, edl):
    paths = EditPaths.for_videos_dir(tmp_path)
    result = self_eval.run_loop(lambda i: [
        self_eval.Finding("duration", "info", "duration matches"),
        self_eval.Finding("overlay_past_end", "error", "insert 0 runs past the end", t_output=7.0),
        self_eval.Finding("visual_jump", "info", "not checked", checked=False),
    ])
    p = self_eval.write_self_eval(result, paths)
    assert p == tmp_path / "edit" / "verify" / "self_eval.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["counts"] == {"error": 1, "warning": 0, "info": 2}
    assert data["flagged"] is True
    assert data["max_passes"] == 3
    assert data["unchecked"] == ["visual_jump"]
    assert data["findings"][0]["severity"] == "error"  # errors sort first


def test_evaluate_runs_without_ffmpeg_or_a_key(tmp_path, edl, monkeypatch):
    monkeypatch.setattr(self_eval, "probe_duration", lambda video: None)
    monkeypatch.setattr(self_eval, "extract_pcm", lambda *a, **k: (None, 16000))
    monkeypatch.setattr(self_eval, "run_timeline_view", lambda *a, **k: None)
    paths = EditPaths.for_videos_dir(tmp_path)
    edl.overlays = [Overlay("slot_01/overlay.mp4", 4.0, 2.0,
                            meta={"layout": "fullframe_pip", "url": "https://x", "verified": True})]
    findings = self_eval.evaluate(edl, tmp_path / "final.mp4", cues=[
        self_eval.CueWindow(4.2, 5.0, "الشركة")], edit_paths=paths)
    checks = {f.check for f in findings}
    assert "duration" in checks and "caption_under_overlay" in checks
    assert any(not f.checked for f in findings)


def test_evaluate_finds_captions_spelled_the_render_py_way(tmp_path, edl, monkeypatch):
    monkeypatch.setattr(self_eval, "probe_duration", lambda video: None)
    monkeypatch.setattr(self_eval, "extract_pcm", lambda *a, **k: (None, 16000))
    paths = EditPaths.for_videos_dir(tmp_path)
    doc = WordsDoc(source={"name": "raw01"},
                   words=[Word(word="الشركة", start=20.5, end=20.9)])
    out = captions.write_captions(doc, edl, paths, aspect="9:16")
    edl.subtitles = out["subtitles_field"]          # "captions/final.ass"
    edl.overlays = [Overlay("slot_01/overlay.mp4", 4.0, 3.0,
                            meta={"layout": "fullframe_pip", "url": "https://x",
                                  "verified": True})]
    findings = self_eval.evaluate(edl, tmp_path / "final.mp4", edit_paths=paths,
                                  run_views=False)
    # The cue was read back off the burned file, so the overlay conflict is seen.
    assert any(f.check == "caption_under_overlay" for f in findings)
