"""Stage 0 behaviour: identity by content, exact frame rates, no re-work."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from helpers import ingest
from helpers.paths import EditPaths

# A real `ffprobe -show_format -show_streams` payload, trimmed to the fields
# ingest reads: NTSC frame rate, a rotated phone capture, stereo 48 kHz audio.
PROBE_PAYLOAD = {
    "streams": [
        {
            "codec_type": "video",
            "codec_name": "h264",
            "width": 1920,
            "height": 1080,
            "r_frame_rate": "30000/1001",
            "avg_frame_rate": "30000/1001",
            "duration": "612.412000",
            "side_data_list": [{"side_data_type": "Display Matrix", "rotation": -90.0}],
        },
        {
            "codec_type": "audio",
            "codec_name": "aac",
            "channels": 2,
            "sample_rate": "48000",
            "duration": "612.437000",
        },
    ],
    "format": {"format_name": "mov,mp4,m4a", "duration": "612.437000"},
}


def test_parse_probe_keeps_the_frame_rate_exact():
    p = ingest.parse_probe(PROBE_PAYLOAD)
    assert (p.fps_num, p.fps_den) == (30000, 1001)
    assert p.fps.numerator == 30000 and p.fps.denominator == 1001
    # 29.97 is not the frame rate and must not become one anywhere.
    assert p.fps_float != pytest.approx(29.97, abs=1e-9)
    assert p.width == 1920 and p.height == 1080
    assert p.duration_s == pytest.approx(612.437)
    assert p.audio_channels == 2 and p.sample_rate == 48000
    assert p.has_audio


def test_parse_probe_normalises_rotation_and_display_size():
    p = ingest.parse_probe(PROBE_PAYLOAD)
    assert p.rotation == 270          # -90 wrapped into [0, 360)
    assert p.display_size == (1080, 1920)   # portrait, as the player shows it


def test_parse_probe_prefers_the_measured_rate_for_vfr():
    payload = json.loads(json.dumps(PROBE_PAYLOAD))
    payload["streams"][0]["avg_frame_rate"] = "2997/100"
    p = ingest.parse_probe(payload)
    assert (p.fps_num, p.fps_den) == (2997, 100)


def test_parse_probe_survives_a_source_without_audio():
    payload = {"streams": [PROBE_PAYLOAD["streams"][0]], "format": {"duration": "10.0"}}
    p = ingest.parse_probe(payload)
    assert p.audio_channels is None and not p.has_audio


def test_probe_roundtrips_through_json():
    p = ingest.parse_probe(PROBE_PAYLOAD)
    back = ingest.Probe.from_dict(json.loads(json.dumps(p.to_dict())))
    assert back == p


def test_sha256_file_matches_hashlib(tmp_path):
    f = tmp_path / "raw01.mp4"
    f.write_bytes(b"zeta" * 5000)
    assert ingest.sha256_file(f) == hashlib.sha256(b"zeta" * 5000).hexdigest()


def test_sha256_cache_shortcuts_on_size_and_mtime(tmp_path):
    f = tmp_path / "raw01.mp4"
    f.write_bytes(b"A" * 1024)
    cache = tmp_path / "sha.json"

    first = ingest.sha256_file(f, cache=cache)
    stat = f.stat()

    # Same size, same mtime, different bytes: the shortcut must fire, which is
    # only observable as the stale hash coming back.
    f.write_bytes(b"B" * 1024)
    os.utime(f, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert ingest.sha256_file(f, cache=cache) == first

    # Move the mtime and it must re-read the file.
    os.utime(f, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    second = ingest.sha256_file(f, cache=cache)
    assert second == hashlib.sha256(b"B" * 1024).hexdigest()
    assert second != first
    assert json.loads(cache.read_text())[str(f.resolve())]["sha256"] == second


def test_sha256_cache_survives_a_corrupt_cache_file(tmp_path):
    f = tmp_path / "raw01.mp4"
    f.write_bytes(b"xyz")
    cache = tmp_path / "sha.json"
    cache.write_text("{not json")
    assert ingest.sha256_file(f, cache=cache) == hashlib.sha256(b"xyz").hexdigest()


# -- ingest() -----------------------------------------------------------------


@pytest.fixture
def fake_ffmpeg(monkeypatch):
    """Replace the two subprocess calls; tests must not need ffmpeg."""
    calls = {"extract": []}

    def fake_probe(path):
        return ingest.parse_probe(PROBE_PAYLOAD)

    def fake_extract(src, out_wav, *, force=False):
        calls["extract"].append(Path(out_wav))
        Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
        Path(out_wav).write_bytes(b"RIFFfake")
        return Path(out_wav)

    monkeypatch.setattr(ingest, "probe_source", fake_probe)
    monkeypatch.setattr(ingest, "extract_wav", fake_extract)
    return calls


def test_ingest_writes_sources_json(tmp_path, fake_ffmpeg):
    raw = tmp_path / "raw01.mp4"
    raw.write_bytes(b"video-bytes")
    paths = EditPaths.for_videos_dir(tmp_path)

    sources = ingest.ingest([raw], paths)
    assert len(sources) == 1
    s = sources[0]
    assert s.name == "raw01"
    assert s.sha256 == hashlib.sha256(b"video-bytes").hexdigest()
    assert s.wav == paths.audio / "raw01.16k.wav"
    assert s.wav.exists()

    written = json.loads((paths.edit / "sources.json").read_text())
    assert written["sources"][0]["name"] == "raw01"
    assert ingest.load_sources(paths)[0].to_dict() == s.to_dict()
    # The words.json source block is the contract the transcriber consumes.
    assert s.words_source_block() == {
        "name": "raw01", "path": str(raw.resolve()),
        "sha256": s.sha256, "duration_s": pytest.approx(612.437)}


def test_ingest_does_not_re_extract_an_unchanged_source(tmp_path, fake_ffmpeg):
    raw = tmp_path / "raw01.mp4"
    raw.write_bytes(b"video-bytes")
    paths = EditPaths.for_videos_dir(tmp_path)

    ingest.ingest([raw], paths)
    assert len(fake_ffmpeg["extract"]) == 1

    ingest.ingest([raw], paths)          # hard rule 9
    assert len(fake_ffmpeg["extract"]) == 1

    raw.write_bytes(b"different-bytes")  # edited re-export: must not hit the cache
    ingest.ingest([raw], paths)
    assert len(fake_ffmpeg["extract"]) == 2


def test_ingest_deduplicates_names_from_different_directories(tmp_path, fake_ffmpeg):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    (a / "raw01.mp4").write_bytes(b"one")
    (b / "raw01.mp4").write_bytes(b"two")
    paths = EditPaths.for_videos_dir(tmp_path)

    names = [s.name for s in ingest.ingest([a / "raw01.mp4", b / "raw01.mp4"], paths)]
    assert names == ["raw01", "raw01_2"]


def test_ingest_refuses_a_source_with_no_audio(tmp_path, monkeypatch):
    raw = tmp_path / "silent.mp4"
    raw.write_bytes(b"x")
    monkeypatch.setattr(ingest, "probe_source", lambda p: ingest.Probe(duration_s=5.0))
    with pytest.raises(ingest.IngestError, match="no audio"):
        ingest.ingest([raw], EditPaths.for_videos_dir(tmp_path))
