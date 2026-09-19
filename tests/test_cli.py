"""The CLI is where the stages meet, so it is where mismatches show up.

These run `cli.main()` in process with the model mocked, which is the only way
to catch a wrong keyword argument between two modules that each pass their own
unit tests.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import cli
from helpers import gemini_client
from helpers.edl import EDL
from helpers.paths import REPO_ROOT
from helpers.words import WordsDoc

FIXTURES = REPO_ROOT / "tests" / "fixtures"
HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


@pytest.fixture
def session(tmp_path):
    """A videos dir holding the fixture take, with the media built if needed."""
    if not (FIXTURES / "raw01.mp4").exists():
        subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "make_fixtures.py")],
                       check=True, capture_output=True)
    shutil.copy(FIXTURES / "raw01.mp4", tmp_path / "raw01.mp4")
    return tmp_path


def _plan_response() -> dict:
    expected = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))
    return {
        "kept_text": expected["kept_text"],
        "inserts": [{"after_text": "OpenAI جابت مليار",
                     "claim": expected["insert"]["claim"],
                     "entity": "OpenAI",
                     "prefer_source": expected["insert"]["url"]}],
        "strategy": "Keep the hook, drop the three fillers, hold the source mention.",
    }


@pytest.fixture
def planner(monkeypatch):
    """Answer the planner; leave every other call to fall through to 'no fixture'."""
    def handler(req: dict):
        schema = req.get("schema") or {}
        props = (schema.get("properties") or {}) if isinstance(schema, dict) else {}
        if "kept_text" in props:
            return _plan_response()
        return None

    gemini_client.register_mock(handler)
    yield
    gemini_client.register_mock(None)


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
def test_ingest_records_the_source(session):
    assert cli.main(["ingest", str(session / "raw01.mp4"), "-v", str(session)]) == 0
    sources = json.loads((session / "edit" / "sources.json").read_text(encoding="utf-8"))
    entry = sources["sources"][0] if isinstance(sources, dict) else sources[0]
    assert entry["name"] == "raw01"
    assert len(entry["sha256"]) == 64


def _seed_transcript(session) -> None:
    """Put a current words.json in place so plan/edit skip the model and aligner."""
    sources = json.loads((session / "edit" / "sources.json").read_text(encoding="utf-8"))
    entry = (sources["sources"] if isinstance(sources, dict) else sources)[0]
    doc = WordsDoc.load(FIXTURES / "raw01.words.json")
    doc.source = dict(doc.source) | {"sha256": entry["sha256"], "name": "raw01"}
    doc.save(session / "edit" / "transcripts" / "raw01.words.json")
    # A cached transcript only counts once its QA passed (cli.stage_transcribe).
    (session / "edit" / "transcripts" / "raw01.qa.json").write_text(
        json.dumps({"ok": True, "source": {"sha256": entry["sha256"]}}), encoding="utf-8")


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
def test_plan_writes_a_valid_edl(session, planner):
    cli.main(["ingest", str(session / "raw01.mp4"), "-v", str(session)])
    _seed_transcript(session)

    assert cli.main(["plan", "-v", str(session), "--auto"]) == 0

    edl = EDL.load(session / "edit" / "edl.json")
    edl.validate()
    assert edl.ranges and edl.total_duration_s > 0
    assert (session / "edit" / "edit_plan.json").exists()

    plan = json.loads((session / "edit" / "edit_plan.json").read_text(encoding="utf-8"))
    insert = plan["inserts"][0]
    assert insert["trigger_word_index"] >= 0
    # Rule 12: the planner anchors to a word, never to a time.
    assert not any(k.endswith("_s") or k.endswith("time") for k in insert)


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
def test_edit_runs_every_stage_and_leaves_the_documented_artifacts(session, planner):
    cli.main(["ingest", str(session / "raw01.mp4"), "-v", str(session)])
    _seed_transcript(session)

    rc = cli.main(["edit", str(session / "raw01.mp4"), "-v", str(session),
                   "--auto", "--no-screenshots", "--preview"])
    assert rc == 0

    edit = session / "edit"
    for artifact in ("edl.json", "edit_plan.json", "takes_packed.md",
                     "captions/final.ass", "captions/final.srt",
                     "decision_report.html", "project.md", "preview.mp4"):
        assert (edit / artifact).exists(), f"{artifact} was not produced"

    # Nothing may be written into the repo itself (hard rule 11).
    assert not (REPO_ROOT / "edit").exists()

    report = (edit / "decision_report.html").read_text(encoding="utf-8")
    assert "<html" in report and "http" not in report.split("<body")[0].replace(
        "http-equiv", ""), "the report must be self-contained"

    project = (edit / "project.md").read_text(encoding="utf-8")
    assert "Strategy" in project and "fillers" in project.lower() or "cut ratio" in project
