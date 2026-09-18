"""The acceptance test: fixtures in, a real rendered file out.

Everything else in the suite runs on synthetic objects. This one shells out to
ffmpeg and checks the artifact, because the failures this pipeline is most
likely to ship are not logic errors - they are a caption burned under the
Instagram UI, a duration that does not match the EDL, or an ASS file whose font
was quietly replaced by the vendored force_style. None of those are visible
without rendering.

Skipped when ffmpeg is absent, so the rest of the suite stays dependency-free.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from helpers import captions, derive_cuts, edl as edl_mod, ingest as ingest_mod
from helpers.config import aspect_config
from helpers.paths import EditPaths, REPO_ROOT
from helpers.words import WordsDoc

FIXTURES = REPO_ROOT / "tests" / "fixtures"
HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))

pytestmark = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg/ffprobe not installed")


@pytest.fixture(scope="module")
def fixture_clip() -> Path:
    """The media fixtures are generated, not committed. Build them on demand."""
    clip = FIXTURES / "raw01.mp4"
    if not clip.exists():
        subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "make_fixtures.py")],
                       check=True, capture_output=True)
    return clip


@pytest.fixture(scope="module")
def rendered(fixture_clip, tmp_path_factory):
    """Cut, caption and render the fixture take once for the whole module."""
    videos = tmp_path_factory.mktemp("session")
    shutil.copy(fixture_clip, videos / "raw01.mp4")
    paths = EditPaths.for_videos_dir(videos).ensure()

    source = ingest_mod.ingest([videos / "raw01.mp4"], paths)[0]
    doc = WordsDoc.load(FIXTURES / "raw01.words.json")
    doc.save(paths.words_json("raw01"))
    expected = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))

    profile = {"cut_ratio": 0.16, "filler_policy": {"remove": ["euh", "donc"]}, "inserts": {}}
    plan = derive_cuts.derive(doc, expected["kept_text"], profile, "raw01")

    e = edl_mod.build({"raw01": str(source.path)}, plan.ranges, aspect="9:16")
    written = captions.write_captions(doc, e, paths, aspect="9:16")
    e.subtitles = written["subtitles_field"]
    e.save(paths.edl)

    subprocess.run([sys.executable, str(REPO_ROOT / "helpers" / "render.py"),
                    str(paths.edl), "-o", str(paths.final)],
                   check=True, capture_output=True)
    return {"paths": paths, "edl": e, "plan": plan, "doc": doc, "ass": Path(written["ass"])}


def _probe_duration(video: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nokey=1:noprint_wrappers=1", str(video)],
        capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def test_audio_extraction_produces_a_readable_wav(rendered):
    """Regression: the atomic .part temp name left ffmpeg unable to pick a muxer."""
    import wave

    wav = rendered["paths"].audio / "raw01.16k.wav"
    assert wav.exists()
    with wave.open(str(wav), "rb") as f:
        assert f.getframerate() == 16000
        assert f.getnchannels() == 1
        assert f.getnframes() > 0


def test_render_duration_matches_the_edl(rendered):
    expected = rendered["edl"].total_duration_s
    actual = _probe_duration(rendered["paths"].final)
    # Keyframe placement and the concat demuxer move the tail by a frame or two.
    assert abs(actual - expected) < 0.25, f"EDL says {expected:.2f}s, file is {actual:.2f}s"


def test_no_cut_lands_inside_a_word(rendered):
    doc, plan = rendered["doc"], rendered["plan"]
    edges = [t for r in plan.ranges for t in (r.start, r.end)]
    for edge in edges:
        for w in doc.words:
            if w.timed and w.start < edge < w.end:
                pytest.fail(f"cut at {edge:.3f}s falls inside {w.display!r} "
                            f"({w.start:.3f}-{w.end:.3f})")


def test_the_burned_font_survives_the_vendored_force_style(rendered):
    """render.py pins FontName=Helvetica, which has no Arabic glyphs at all.

    Only an inline override tag survives force_style, so if this disappears the
    Darija captions render as whatever fontconfig happens to substitute.
    """
    text = rendered["ass"].read_text(encoding="utf-8")
    dialogues = [ln for ln in text.splitlines() if ln.startswith("Dialogue:")]
    assert dialogues
    for line in dialogues:
        assert "\\fn" in line, "no inline font override; force_style will win"
        assert "\\pos(" in line, "no inline position; force_style's MarginV will win"


def test_captions_clear_the_instagram_ui_band(rendered):
    """A caption inside the bottom UI band is hidden by Instagram's own chrome."""
    _, aspect_cfg = aspect_config("9:16")
    height = aspect_cfg["resolution"][1]
    ui_band_top = height - aspect_cfg["safe_area"]["bottom"]

    for line in rendered["ass"].read_text(encoding="utf-8").splitlines():
        if not line.startswith("Dialogue:"):
            continue
        y = int(line.split("\\pos(")[1].split(")")[0].split(",")[1])
        assert y <= ui_band_top, (
            f"caption baseline at y={y} sits inside the bottom UI band "
            f"(starts at y={ui_band_top})")


def test_the_rendered_frame_actually_carries_ink_where_the_caption_is(rendered):
    """Proof the burn happened: the caption row differs from the row above it.

    The fixture video is flat colour bars, so any local variation in the caption
    row is text. A missing libass, a bad path, or a silently skipped subtitles
    filter all show up here and nowhere else.
    """
    import numpy as np
    from PIL import Image

    paths = rendered["paths"]
    frame = paths.verify / "e2e_frame.png"
    frame.parent.mkdir(parents=True, exist_ok=True)
    # 5.2s is inside a cue; see tests/fixtures/expected.json for the script.
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "5.2", "-i", str(paths.final),
                    "-frames:v", "1", str(frame)], check=True, capture_output=True)

    img = np.asarray(Image.open(frame).convert("L"), dtype=float)
    _, aspect_cfg = aspect_config("9:16")
    baseline = aspect_cfg["resolution"][1] - 480          # captions.yaml margin_v
    caption_band = img[baseline - 120:baseline, :]
    control_band = img[baseline - 400:baseline - 280, :]

    # Colour bars are vertically uniform, so row-to-row variance is ~0 without
    # text and clearly non-zero with it.
    assert caption_band.std(axis=0).mean() > control_band.std(axis=0).mean() + 1.0, (
        "the caption row looks identical to the background: nothing was burned")
