"""Capture settings come from the configs, and the readability arithmetic holds."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from helpers import config as cfgmod
from helpers import gemini_client, screenshot, verify_screenshot
from helpers.gemini_client import LLM
from helpers.screenshot import CaptureError

ASPECTS = ["9:16", "4:5", "1:1", "16:9"]

SOURCES = {
    "capture": {"timeout_s": 45, "retries_per_slot": 1,
                "wait_for_network_idle": True, "dismiss_cookie_banners": True},
}


class FakeProc:
    def __init__(self, returncode=0, stderr=""):
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = ""


def _runner(write: Path | None = None, returncode: int = 0, stderr: str = "",
            log: list | None = None):
    """A subprocess.run stand-in: records argv, optionally creates the output."""
    def run(cmd, **kw):
        if log is not None:
            log.append(cmd)
        if write is not None and returncode == 0:
            write.parent.mkdir(parents=True, exist_ok=True)
            write.write_bytes(b"\x89PNG\r\n\x1a\n")
        return FakeProc(returncode, stderr)
    return run


# -- width is derived from the aspect, never hardcoded ----------------------


@pytest.mark.parametrize("aspect", ASPECTS)
def test_capture_width_satisfies_the_layout_readability_arithmetic(aspect):
    s = screenshot.capture_settings(aspect=aspect, sources_cfg=SOURCES)
    _key, acfg = cfgmod.aspect_config(aspect)

    # configs/layout.yaml header: safe_width / capture.width must stay at ~1.0,
    # so a CSS pixel of the page survives as a device pixel in the safe column.
    assert s["readability"] == pytest.approx(
        screenshot.safe_width_px(acfg) / acfg["capture"]["width"])
    assert s["readability"] >= 1.0
    assert s["width"] == acfg["capture"]["width"]
    assert s["image_width_px"] == s["width"] * (2 if s["retina"] else 1)


def test_capture_width_differs_per_aspect():
    widths = {a: screenshot.capture_settings(aspect=a, sources_cfg=SOURCES)["width"]
              for a in ASPECTS}
    # The 1600 of the spec is the YouTube number; a Reel must not inherit it.
    assert widths["9:16"] == 800
    assert widths["16:9"] == 1600
    assert len(set(widths.values())) > 1


def test_capture_settings_follow_an_edited_layout_config():
    layout = {"default_aspect": "9:16",
              "aspects": {"9:16": {"resolution": [1080, 1920], "fps": 30,
                                   "safe_area": {"top": 250, "bottom": 420,
                                                 "left": 60, "right": 200},
                                   "capture": {"width": 820, "retina": False}}}}
    s = screenshot.capture_settings(aspect="9:16", sources_cfg=SOURCES, layout_cfg=layout)
    assert s["width"] == 820 and s["retina"] is False
    assert s["image_width_px"] == 820
    assert s["readability"] == pytest.approx(1.0)


def test_capture_policy_comes_from_sources_yaml():
    s = screenshot.capture_settings(
        aspect="9:16", sources_cfg={"capture": {"timeout_s": 12, "retries_per_slot": 3,
                                                "dismiss_cookie_banners": False}})
    assert s["timeout_ms"] == 12000
    assert s["retries"] == 3
    assert s["dismiss_cookie_banners"] is False


# -- shot-scraper -----------------------------------------------------------


def test_shot_scraper_cmd_shape():
    cmd = screenshot.shot_scraper_cmd(
        "https://openai.com/a", "out.png", selector="article header",
        width=800, retina=True, timeout_ms=45000,
        javascript=screenshot.cookie_banner_js())

    assert cmd[:3] == ["shot-scraper", "shot", "https://openai.com/a"]
    assert cmd[cmd.index("--width") + 1] == "800"
    assert "--retina" in cmd
    assert cmd[cmd.index("--selector") + 1] == "article header"
    assert cmd[cmd.index("--timeout") + 1] == "45000"
    assert "readyState" in cmd[cmd.index("--wait-for") + 1]
    assert "click" in cmd[cmd.index("--javascript") + 1]


def test_capture_url_uses_the_aspect_width_and_returns_the_png(tmp_path):
    out = tmp_path / "shot.png"
    log: list = []
    res = screenshot.capture_url("https://openai.com/a", out, selector="article header",
                                 aspect="9:16", sources_cfg=SOURCES,
                                 runner=_runner(write=out, log=log))

    assert res.path == out and res.source_type == "html"
    assert log[0][log[0].index("--width") + 1] == "800"


def test_capture_url_raises_when_shot_scraper_fails(tmp_path):
    with pytest.raises(CaptureError, match="shot-scraper failed"):
        screenshot.capture_url("https://openai.com/a", tmp_path / "s.png",
                               aspect="9:16", sources_cfg=SOURCES,
                               runner=_runner(returncode=2, stderr="timeout"))


def test_capture_url_raises_when_no_file_appears(tmp_path):
    with pytest.raises(CaptureError, match="wrote no file"):
        screenshot.capture_url("https://openai.com/a", tmp_path / "s.png",
                               aspect="9:16", sources_cfg=SOURCES, runner=_runner())


# -- pdf --------------------------------------------------------------------


def test_pdftoppm_cmd_targets_one_page_and_keeps_the_page_aspect():
    cmd = screenshot.pdftoppm_cmd("f.pdf", 7, "out", image_width_px=1600)
    assert cmd[cmd.index("-f") + 1] == "7" and cmd[cmd.index("-l") + 1] == "7"
    assert cmd[cmd.index("-scale-to-x") + 1] == "1600"
    assert cmd[cmd.index("-scale-to-y") + 1] == "-1"


def test_capture_pdf_page_renames_the_page_pdftoppm_produced(tmp_path):
    out = tmp_path / "shot.png"

    def run(cmd, **kw):
        # pdftoppm pads the page number onto the prefix; the caller cannot guess it.
        (tmp_path / "shot-07.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        return FakeProc()

    res = screenshot.capture_pdf_page("f.pdf", 7, out, aspect="9:16",
                                      sources_cfg=SOURCES, runner=run)
    assert res.source_type == "pdf" and out.exists()
    assert not (tmp_path / "shot-07.png").exists()


# -- chart ------------------------------------------------------------------


def test_render_chart_is_labelled_as_a_rendering(tmp_path):
    pytest.importorskip("matplotlib")
    out = tmp_path / "chart.png"
    res = screenshot.render_chart({"2024": 1.2, "2025": 3.4}, out,
                                  title="OpenAI revenue", source_label="openai.com",
                                  aspect="9:16", sources_cfg=SOURCES)
    assert out.exists()
    assert res.source_type == "chart"
    assert "openai.com" in (res.note or "")


def test_render_chart_refuses_empty_figures(tmp_path):
    pytest.importorskip("matplotlib")
    with pytest.raises(CaptureError):
        screenshot.render_chart({}, tmp_path / "c.png", title="t", source_label="s",
                                aspect="9:16", sources_cfg=SOURCES)


# -- slot dispatch ----------------------------------------------------------


def test_an_article_is_one_headline_viewport_and_an_x_post_walks_its_selectors(tmp_path):
    seen: list = []

    def fake_capture_url(url, out, *, selector, attempt, **kw):
        seen.append(selector)
        Path(out).write_bytes(b"\x89PNG\r\n\x1a\n")
        return screenshot.CaptureResult(path=Path(out), source_type="html",
                                        selector=selector, attempt=attempt)

    art = {"slot_id": "slot_01", "url": "https://openai.com/a",
           "selectors": ["article header", "article", "main"], "source_type": "owner"}
    screenshot.capture_for_slot(tmp_path, art, aspect="9:16", sources_cfg=SOURCES,
                                capture_url_fn=fake_capture_url)
    assert seen == [None]                   # the JS scrolls to the h1; no crop
    written = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert written["source_type"] == "owner"  # editorial type survives the capture
    with pytest.raises(screenshot.CaptureError):  # same shot again is no retry
        screenshot.capture_for_slot(tmp_path, written, aspect="9:16", attempt=1,
                                    sources_cfg=SOURCES, capture_url_fn=fake_capture_url)

    seen.clear()
    post = {"slot_id": "slot_02", "url": "https://x.com/OpenAI/status/1",
            "selectors": ["article[data-testid=tweet]"], "source_type": "owner"}
    screenshot.capture_for_slot(tmp_path / "x", post, aspect="9:16", sources_cfg=SOURCES,
                                capture_url_fn=fake_capture_url)
    assert seen == ["article[data-testid=tweet]"]


def test_capture_for_slot_marks_a_chart_as_a_chart(tmp_path):
    pytest.importorskip("matplotlib")
    meta = {"slot_id": "slot_02", "url": "https://openai.com/a", "source_type": "owner",
            "chart": {"figures": {"2024": 1.0, "2025": 2.0}, "title": "t",
                      "source_label": "openai.com"}}
    screenshot.capture_for_slot(tmp_path, meta, aspect="9:16", sources_cfg=SOURCES)

    written = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    # A drawing of the numbers must never be reported as a capture of the page.
    assert written["source_type"] == "chart"
    assert written["capture_source_type"] == "chart"


def test_capture_for_slot_without_a_url_is_an_error(tmp_path):
    with pytest.raises(CaptureError, match="no url"):
        screenshot.capture_for_slot(tmp_path, {"slot_id": "slot_03"}, aspect="9:16",
                                    sources_cfg=SOURCES)


def test_a_pdf_source_is_downloaded_then_rendered(tmp_path):
    meta = {"slot_id": "slot_04", "url": "https://cdn.openai.com/report.pdf"}
    got = {}

    def fake_download(url, dest, timeout):
        dest.write_bytes(b"%PDF-1.7")
        return dest

    def fake_pdf(local, page, out, **kw):
        got.update(local=local, page=page)
        Path(out).write_bytes(b"\x89PNG")
        return screenshot.CaptureResult(path=Path(out), source_type="pdf")

    screenshot.capture_for_slot(tmp_path, meta, aspect="9:16", sources_cfg=SOURCES,
                                capture_pdf_fn=fake_pdf, download_pdf_fn=fake_download)
    assert got == {"local": str(tmp_path / "source.pdf"), "page": 1}


def test_a_pdf_that_will_not_download_fails_the_capture(tmp_path):
    def refuse(url, dest, timeout):
        raise CaptureError("could not download the PDF")
    with pytest.raises(CaptureError, match="could not download"):
        screenshot.capture_for_slot(tmp_path, {"slot_id": "s", "url": "https://x/a.pdf"},
                                    aspect="9:16", sources_cfg=SOURCES, download_pdf_fn=refuse)


def test_module_imports_without_pulling_in_the_optional_extras():
    # Importing the module must cost nothing but stdlib + pyyaml: the `[shots]`
    # extras are imported inside the function that needs them.
    code = ("import sys, helpers.screenshot; "
            "print([m for m in ('matplotlib', 'PIL', 'google') if m in sys.modules])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(Path(__file__).resolve().parent.parent))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"


# -- verification: nothing unverified ever ships -----------------------------


def _offline_capture(*a, **kw):
    """A retry must never reach shot-scraper: it would load the real URL."""
    raise screenshot.CaptureError("offline test")


def _slot(tmp_path, **meta):
    d = tmp_path / "slot_01"
    d.mkdir()
    (d / "shot.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    m = {"slot_id": "slot_01", "claim": "OpenAI raised 40 billion",
         "url": "https://openai.com/index/funding/", "source_type": "owner",
         "selectors": ["article header", "article", "main"],
         "image": str(d / "shot.png"), "verified": False}
    m.update(meta)
    (d / "meta.json").write_text(json.dumps(m), encoding="utf-8")
    return d


def _vision(answers):
    """Mock vision backend answering `answers` in order, one per call."""
    seq = list(answers)
    calls: list = []

    def handler(req):
        calls.append(req)
        return seq.pop(0) if seq else seq_default
    seq_default = {"visible": False, "evidence": "nothing legible"}
    gemini_client.register_mock(handler)
    return calls


def test_verified_slot_records_its_evidence(tmp_path):
    d = _slot(tmp_path)
    _vision([{"visible": True, "evidence": "headline reads 'OpenAI raises $40B'"}])

    verdict = verify_screenshot.verify_slot(d, llm=LLM(), sources_cfg=SOURCES,
                                            capture_fn=_offline_capture)

    assert verdict.visible is True
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    assert meta["verified"] is True
    assert "OpenAI raises" in meta["evidence"]
    assert meta["dropped_reason"] is None


def test_a_failed_check_retries_with_a_different_selector(tmp_path):
    d = _slot(tmp_path)
    calls = _vision([{"visible": False, "evidence": "cookie wall only"},
                     {"visible": True, "evidence": "headline and figure visible"}])
    recaptured: list = []

    def fake_capture(slot_dir, meta, *, attempt, **kw):
        # The retry must shoot a DIFFERENT region, or it gets the same pixels.
        recaptured.append(meta["selectors"][attempt])
        meta["image"] = str(Path(slot_dir) / f"shot_retry{attempt}.png")
        Path(meta["image"]).write_bytes(b"\x89PNG\r\n\x1a\n")

    verdict = verify_screenshot.verify_slot(d, llm=LLM(), sources_cfg=SOURCES,
                                            capture_fn=fake_capture)

    assert recaptured == ["article"]
    assert verdict.visible is True and verdict.attempt == 1
    assert len(calls) == 2
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    assert len(meta["verification"]["attempts"]) == 2


def test_an_unverified_slot_is_dropped_with_the_reason(tmp_path):
    d = _slot(tmp_path)
    _vision([{"visible": False, "evidence": "cookie wall only"},
             {"visible": False, "evidence": "navigation bar, no headline"}])

    def fake_capture(slot_dir, meta, *, attempt, **kw):
        meta["image"] = str(Path(slot_dir) / "shot.png")

    verdict = verify_screenshot.verify_slot(d, llm=LLM(), sources_cfg=SOURCES,
                                            capture_fn=fake_capture)

    assert verdict.visible is False
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    # Zeta hard rule 12: dropped, flagged, and never silently included.
    assert meta["verified"] is False
    assert "no evidence" in meta["dropped_reason"]
    assert meta["evidence"] == ""
    assert len(meta["verification"]["attempts"]) == 2


def test_a_vision_outage_drops_the_slot_instead_of_shipping_it(tmp_path):
    d = _slot(tmp_path)

    def dead(_req):
        raise RuntimeError("vision API down")
    gemini_client.register_mock(dead)

    verdict = verify_screenshot.verify_slot(d, llm=LLM(), sources_cfg=SOURCES,
                                            capture_fn=_offline_capture)

    assert verdict.visible is False
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    assert meta["verified"] is False
    assert "unavailable" in meta["dropped_reason"]


def test_a_missing_capture_is_not_verifiable(tmp_path):
    d = _slot(tmp_path, image=str(tmp_path / "gone.png"))
    _vision([{"visible": True, "evidence": "should never be asked"}])

    verdict = verify_screenshot.verify_slot(d, llm=LLM(), sources_cfg=SOURCES,
                                            capture_fn=_offline_capture)
    assert verdict.visible is False and "no image" in (verdict.error or "")


def test_a_viewport_capture_is_bounded_to_the_card_shape():
    # Without --height shot-scraper grabs the full page: 1600x30926 on a real
    # press release, which fit_card would shrink past reading.
    cmd = screenshot.shot_scraper_cmd("https://x.test", "o.png", selector=None, width=800,
                                      retina=True, timeout_ms=1000, height=1220)
    assert cmd[cmd.index("--height") + 1] == "1220"
    with_sel = screenshot.shot_scraper_cmd("https://x.test", "o.png", selector="article",
                                           width=800, retina=True, timeout_ms=1000, height=1220)
    assert "--height" not in with_sel


def test_a_new_capture_saved_over_the_old_one_is_verified_afresh(tmp_path, monkeypatch):
    # The cache once keyed on the file NAME: a TechCrunch shot saved over a
    # Bloomberg bot wall at the same shot.png got the wall's "no" back.
    answers = iter(['{"visible": false, "evidence": "bot wall"}',
                    '{"visible": true, "evidence": "headline"}'])
    gemini_client.register_mock(None)
    llm = LLM(cache_dir=tmp_path / "cache")
    monkeypatch.setattr(type(llm), "mock_mode", property(lambda self: False), raising=False)
    monkeypatch.setattr(llm, "require", lambda: None)
    monkeypatch.setattr(llm, "_call_with_retry", lambda req, images, files: next(answers))
    shot = tmp_path / "shot.png"
    shot.write_bytes(b"\x89PNG wall")
    assert verify_screenshot.verify_image(shot, "claim", llm=llm).visible is False
    shot.write_bytes(b"\x89PNG headline")
    assert verify_screenshot.verify_image(shot, "claim", llm=llm).visible is True
