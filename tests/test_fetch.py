"""Download flags and pairs.csv rows, without a network."""

from __future__ import annotations

import csv
import subprocess

import pytest

from helpers import fetch


def test_cmd_merges_to_mp4_and_keeps_the_url_record(tmp_path):
    cmd = fetch.yt_dlp_cmd("https://www.tiktok.com/@zeta/video/123", tmp_path, name="ep14")
    assert cmd[0] == "yt-dlp"
    assert "--merge-output-format" in cmd and cmd[cmd.index("--merge-output-format") + 1] == "mp4"
    assert "--write-info-json" in cmd
    assert "ep14.%(ext)s" in cmd[cmd.index("-o") + 1]
    assert cmd[-1].endswith("/123")


def test_cookies_flag_only_when_asked(tmp_path):
    assert "--cookies-from-browser" not in fetch.yt_dlp_cmd("u", tmp_path)
    cmd = fetch.yt_dlp_cmd("u", tmp_path, cookies_from_browser="chrome")
    assert cmd[cmd.index("--cookies-from-browser") + 1] == "chrome"


def test_failure_explains_the_usual_cause(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch.shutil, "which", lambda _: "/usr/bin/yt-dlp")

    def failing(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 1, "", "ERROR: login required")

    with pytest.raises(fetch.FetchError, match="logged-in session"):
        fetch.fetch("https://www.instagram.com/reel/x/", tmp_path, runner=failing)


def test_success_records_the_source_url(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch.shutil, "which", lambda _: "/usr/bin/yt-dlp")

    def fake(cmd, **kw):
        (tmp_path / "ep14.mp4").write_bytes(b"\x00")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    video = fetch.fetch("https://www.tiktok.com/@zeta/video/123", tmp_path,
                        name="ep14", runner=fake)
    assert video.name == "ep14.mp4"
    import json
    assert json.loads(video.with_suffix(".source.json").read_text())["url"].endswith("/123")


def test_append_pair_writes_a_published_only_row(tmp_path):
    published = tmp_path / "published" / "ep14.mp4"
    published.parent.mkdir()
    published.write_bytes(b"\x00")
    pairs = fetch.append_pair(tmp_path / "pairs.csv", published, name="ep14")

    rows = list(csv.reader(pairs.open(encoding="utf-8")))
    assert rows[0] == ["raw", "published", "name"]
    assert rows[1] == ["", "published/ep14.mp4", "ep14"]


def test_append_pair_keeps_the_raw_take_when_there_is_one(tmp_path):
    for p in ("raw/ep15_raw.mp4", "published/ep15.mp4"):
        (tmp_path / p).parent.mkdir(exist_ok=True)
        (tmp_path / p).write_bytes(b"\x00")
    pairs = fetch.append_pair(tmp_path / "pairs.csv", tmp_path / "published/ep15.mp4",
                              raw=tmp_path / "raw/ep15_raw.mp4", name="ep15")

    row = list(csv.reader(pairs.open(encoding="utf-8")))[1]
    assert row == ["raw/ep15_raw.mp4", "published/ep15.mp4", "ep15"]
