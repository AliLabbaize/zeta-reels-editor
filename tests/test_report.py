"""The decision report: self-contained HTML, and the project.md session memory."""

from __future__ import annotations

import json
from html.parser import HTMLParser

import pytest

from helpers import report
from helpers.edl import EDL, Overlay, Range
from helpers.paths import EditPaths

VOID = {"meta", "img", "br", "hr", "input", "link", "source", "area", "col"}

PNG_BYTES = (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
             b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
             b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82")


class _Balance(HTMLParser):
    """Every container tag opens and closes, in order."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.errors: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag not in VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"</{tag}> closes {self.stack[-1:] or ['nothing']}")
            return
        self.stack.pop()


def assert_valid_html(text: str) -> None:
    p = _Balance()
    p.feed(text)
    p.close()
    assert not p.errors, p.errors
    assert not p.stack, f"unclosed: {p.stack}"


@pytest.fixture
def edl():
    return EDL(
        sources={"raw01": "/tmp/raw01.mp4"},
        ranges=[
            Range("raw01", 2.0, 12.0, beat="HOOK",
                  cut_before={"class": "intro_trim", "text": "طيب... واحد, جوج, تلاتة",
                              "reason": "dead air before the hook",
                              "source_span": [0.0, 2.0]}),
            Range("raw01", 20.0, 44.0, reason="kept, the numbers",
                  cut_before={"class": "retake", "text": "الشركة ديال... la société",
                              "reason": "earlier take of the same sentence"}),
        ],
        overlays=[Overlay("edit/screenshots/slot_01/overlay.mp4", 14.2, 4.5, meta={
            "claim": "SpaceX IPO filing values the company at $400B",
            "url": "https://www.sec.gov/filing/spacex-s1",
            "source_type": "owner", "verified": True,
            "evidence": "headline and the figure are both visible",
            "layout": "fit_card", "trigger_word": "SpaceX"})],
        subtitles="edit/captions/final.ass",
        style_profile="style_profile.json",
        meta={"raw_duration_s": 60.0},
    )


@pytest.fixture
def profile():
    return {"cut_ratio": 0.31, "inserts": {"per_minute": 2.4}}


def _write(tmp_path, edl, **kw):
    paths = EditPaths.for_videos_dir(tmp_path)
    out = report.write_report(paths, edl, **kw)
    return out, out.read_text(encoding="utf-8")


# -------- html ---------------------------------------------------------------


def test_report_is_valid_self_contained_html(tmp_path, edl, profile):
    out, text = _write(tmp_path, edl, style_profile=profile)
    assert out == tmp_path / "edit" / "decision_report.html"
    assert_valid_html(text)
    assert text.startswith("<!DOCTYPE html>")
    assert "<style>" in text and "</style>" in text
    # Nothing is fetched at render time: no scripts, no stylesheets, no remote images.
    assert "<script" not in text.lower()
    assert "<link" not in text.lower()
    assert 'src="http' not in text
    assert "cdn" not in text.lower()


def test_report_carries_every_cut_with_its_reason(tmp_path, edl, profile):
    _out, text = _write(tmp_path, edl, style_profile=profile)
    assert "dead air before the hook" in text
    assert "earlier take of the same sentence" in text
    assert "intro_trim" in text and "retake" in text
    assert "طيب... واحد, جوج, تلاتة" in text          # the removed Darija, verbatim
    assert "0:00.00" in text and "0:10.00" in text     # output time of each cut


def test_report_carries_source_urls_and_evidence(tmp_path, edl, profile):
    _out, text = _write(tmp_path, edl, style_profile=profile)
    assert "https://www.sec.gov/filing/spacex-s1" in text
    assert "SpaceX IPO filing values the company at $400B" in text
    assert "headline and the figure are both visible" in text
    assert "verified" in text


def test_unverified_insert_is_called_out(tmp_path, edl, profile):
    edl.overlays[0].meta["verified"] = False
    edl.overlays[0].meta["url"] = ""
    _out, text = _write(tmp_path, edl, style_profile=profile)
    assert "UNVERIFIED" in text
    assert "no source url" in text


def test_thumbnail_is_inlined_as_a_data_uri(tmp_path, edl, profile):
    slot = tmp_path / "edit" / "screenshots" / "slot_01"
    slot.mkdir(parents=True)
    (slot / "shot.png").write_bytes(PNG_BYTES)
    _out, text = _write(tmp_path, edl, style_profile=profile)
    assert 'src="data:image/png;base64,' in text
    assert "iVBOR" in text  # the PNG magic, base64-encoded


def test_style_delta_compares_actual_against_target(tmp_path, edl, profile):
    rows = {r["metric"]: r for r in report.style_delta(edl, profile)}
    # 34s of output from 60s of raw: 43% removed against a 31% target.
    assert rows["cut ratio"]["actual"] == pytest.approx(1 - 34 / 60)
    assert rows["cut ratio"]["target"] == 0.31
    assert rows["cut ratio"]["within"] is False      # outside the +/-0.10 budget
    assert rows["inserts per minute"]["actual"] == pytest.approx(1 / (34 / 60))

    _out, text = _write(tmp_path, edl, style_profile=profile)
    assert "cut ratio" in text and "inserts per minute" in text
    assert "0.31" in text


def test_style_delta_without_a_profile_does_not_crash(edl):
    rows = report.style_delta(edl, None)
    assert [r["within"] for r in rows] == [None, None]


def test_report_shows_self_eval_findings_and_fingerprint(tmp_path, edl, profile):
    self_eval = {
        "passes": 4, "fixes": 3, "max_passes": 3, "flagged": True,
        "counts": {"error": 1, "warning": 0, "info": 1},
        "findings": [
            {"check": "overlay_past_end", "severity": "error", "checked": True,
             "t_output": 14.2, "message": "insert 0 runs past the end of the cut"},
            {"check": "visual_jump", "severity": "info", "checked": False,
             "message": "not checked (no vision model available)"},
        ],
    }
    _out, text = _write(tmp_path, edl, style_profile=profile, self_eval=self_eval)
    assert "insert 0 runs past the end of the cut" in text
    assert "overlay_past_end" in text
    assert "not checked" in text
    assert "flagged" in text

    fp = report.config_fingerprint()
    assert len(fp["hash"]) == 12 and "captions" in fp["configs"]
    assert fp["hash"] in text and fp["configs"]["captions"] in text


def test_report_is_rtl_aware_and_escapes_text(tmp_path, edl, profile):
    edl.overlays[0].meta["claim"] = '<script>alert("x")</script> & co'
    _out, text = _write(tmp_path, edl, style_profile=profile)
    assert 'dir="auto"' in text
    assert "unicode-bidi: plaintext" in text
    assert "&lt;script&gt;" in text
    assert "<script>alert" not in text
    assert_valid_html(text)


def test_report_survives_an_edl_with_nothing_recorded(tmp_path):
    bare = EDL(sources={"raw01": "/tmp/raw01.mp4"}, ranges=[Range("raw01", 0.0, 5.0)])
    _out, text = _write(tmp_path, bare)
    assert_valid_html(text)
    assert "No cuts recorded" in text and "No inserts" in text


# -------- project.md ---------------------------------------------------------


def test_append_project_md_writes_the_four_sections(tmp_path):
    paths = EditPaths.for_videos_dir(tmp_path)
    p = report.append_project_md(paths, {
        "title": "raw01",
        "when": "2026-09-18 10:00 UTC",
        "strategy": "Keep the hook, drop the second take, one insert on the SEC filing.",
        "decisions": [{"decision": "kept the last take", "why": "style profile: keep_last_complete"}],
        "reasoning": ["cut ratio landed at 0.43 against a 0.31 target"],
        "outstanding": ["insert 2 dropped: no verifiable source page"],
        "stats": {"cuts": 2, "inserts": 1},
    })
    assert p == tmp_path / "edit" / "project.md"
    text = p.read_text(encoding="utf-8")
    for section in ("### Strategy", "### Decisions", "### Reasoning log", "### Outstanding"):
        assert section in text
    assert "keep_last_complete" in text
    assert "insert 2 dropped: no verifiable source page" in text
    assert "cuts=2" in text


def test_append_project_md_appends_and_never_rewrites(tmp_path):
    paths = EditPaths.for_videos_dir(tmp_path)
    report.append_project_md(paths, {"title": "first run", "strategy": "one"})
    report.append_project_md(paths, {"title": "second run", "strategy": "two"})
    text = paths.project_md.read_text(encoding="utf-8")
    assert text.count("### Strategy") == 2
    assert text.index("first run") < text.index("second run")
    assert text.count("# Project memory") == 1


def test_append_project_md_accepts_a_session_object(tmp_path):
    paths = EditPaths.for_videos_dir(tmp_path)
    report.append_project_md(paths, report.Session(strategy="s", outstanding=[]))
    text = paths.project_md.read_text(encoding="utf-8")
    assert "### Outstanding" in text and "(none)" in text


# -------- rows ---------------------------------------------------------------


def test_cut_rows_are_placed_on_the_output_timeline(edl):
    rows = report.cut_rows(edl)
    assert [r.t_output for r in rows] == [0.0, 10.0]   # the offsets of their ranges
    assert [r.klass for r in rows] == ["intro_trim", "retake"]
    assert rows[0].source_span == (0.0, 2.0)


def test_insert_rows_keep_the_verification_record(edl):
    rows = report.insert_rows(edl, None)
    assert rows[0].verified is True
    assert rows[0].url.startswith("https://")
    assert rows[0].source_type == "owner"
    assert rows[0].t_output == pytest.approx(14.2)


def test_thumbnail_found_under_the_edit_relative_spelling(tmp_path, edl, profile):
    # An EDL written for render.py spells overlay paths from the edit dir.
    edl.overlays[0].file = "screenshots/slot_02/overlay.mp4"
    slot = tmp_path / "edit" / "screenshots" / "slot_02"
    slot.mkdir(parents=True)
    (slot / "capture.png").write_bytes(PNG_BYTES)
    _out, text = _write(tmp_path, edl, style_profile=profile)
    assert 'src="data:image/png;base64,' in text


def test_cut_rows_carry_the_derive_cuts_extras(tmp_path, edl, profile):
    edl.ranges[1].cut_before["removed_s"] = 3.25
    edl.ranges[1].cut_before["needs_visual_check"] = True
    rows = report.cut_rows(edl)
    assert rows[1].removed_s == 3.25 and rows[1].needs_visual_check is True
    _out, text = _write(tmp_path, edl, style_profile=profile)
    assert "-3.25s" in text and "eyeball this boundary" in text


def test_style_delta_reads_the_cut_plan_metadata(profile):
    edl = EDL(sources={"raw01": "/tmp/raw01.mp4"}, ranges=[Range("raw01", 0.0, 34.0)],
              meta={"cut_plan": {"source_duration_s": 60.0, "cut_ratio": 0.43}})
    rows = {r["metric"]: r for r in report.style_delta(edl, profile)}
    assert rows["cut ratio"]["actual"] == pytest.approx(1 - 34 / 60)
