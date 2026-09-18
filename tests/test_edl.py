"""EDL v2: the output timeline, and the guards that stop a wrong render."""

from __future__ import annotations

import json

import pytest

from helpers.edl import EDL, EDLError, Overlay, Range


def test_output_time_mapping_across_segments(simple_edl):
    assert simple_edl.offsets() == [0.0, 4.0]
    assert simple_edl.total_duration_s == 8.0
    assert simple_edl.to_output_time("raw01", 2.5) == 0.5
    assert simple_edl.to_output_time("raw01", 10.5) == 4.5
    # A word inside the removed stretch has no place on the output timeline.
    assert simple_edl.to_output_time("raw01", 8.0) is None


def test_source_time_round_trip(simple_edl):
    assert simple_edl.to_source_time(4.5) == ("raw01", 10.5)
    assert simple_edl.to_source_time(99.0) is None


def test_unknown_source_is_rejected():
    e = EDL(sources={"raw01": "/a.mp4"}, ranges=[Range("raw02", 0, 1)])
    with pytest.raises(EDLError, match="not in sources"):
        e.validate()


def test_overlapping_ranges_are_rejected():
    e = EDL(sources={"a": "/a.mp4"}, ranges=[Range("a", 0, 5), Range("a", 4.9, 8)])
    with pytest.raises(EDLError, match="overlap"):
        e.validate()


def test_unverified_overlay_never_ships(simple_edl):
    simple_edl.overlays = [Overlay("s.mp4", 1.0, 2.0, {"url": "https://x", "verified": False})]
    with pytest.raises(EDLError, match="verified"):
        simple_edl.validate()


def test_overlay_without_a_source_url_never_ships(simple_edl):
    simple_edl.overlays = [Overlay("s.mp4", 1.0, 2.0, {"verified": True})]
    with pytest.raises(EDLError, match="url"):
        simple_edl.validate()


def test_overlay_past_the_end_is_rejected(simple_edl):
    simple_edl.overlays = [Overlay("s.mp4", 7.0, 3.0, {"url": "u", "verified": True})]
    with pytest.raises(EDLError, match="past the end"):
        simple_edl.validate()


def test_overlapping_overlays_are_rejected(simple_edl):
    meta = {"url": "u", "verified": True}
    simple_edl.overlays = [Overlay("a.mp4", 0.0, 3.0, meta), Overlay("b.mp4", 2.0, 2.0, meta)]
    with pytest.raises(EDLError, match="overlap"):
        simple_edl.validate()


def test_round_trip_keeps_the_v1_subset_render_py_reads(simple_edl, tmp_path):
    simple_edl.overlays = [Overlay("o.mp4", 1.0, 2.0, {"url": "u", "verified": True})]
    simple_edl.subtitles = "edit/captions/final.ass"
    path = simple_edl.save(tmp_path / "edl.json")
    raw = json.loads(path.read_text())

    assert raw["version"] == 2
    # Exactly what vendored render.py reaches for.
    assert set(raw["sources"]) == {"raw01"}
    assert {"source", "start", "end"} <= set(raw["ranges"][0])
    assert {"file", "start_in_output", "duration"} <= set(raw["overlays"][0])
    assert raw["subtitles"] == "edit/captions/final.ass"
    assert raw["total_duration_s"] == 8.0

    assert EDL.load(path).to_output_time("raw01", 10.5) == 4.5


def test_empty_edl_is_refused():
    with pytest.raises(EDLError, match="no ranges"):
        EDL(sources={"a": "/a.mp4"}).validate()
