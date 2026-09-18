"""Stage 0: ingest.

Every later stage assumes two things about a source, and this module is where
both are established:

  1. The SOURCE TIMELINE IS NEVER TOUCHED. The extracted WAV starts at the same
     instant as the video and runs to the same instant. Trimming here would
     shift every word timestamp by an unknown amount and there is no way to
     detect it downstream -- the transcript would simply be wrong.
  2. A source is identified by the sha256 of its bytes, not by its path. Hard
     rule 9 (never re-transcribe an unchanged source) is only enforceable if
     "unchanged" means content, so that a renamed or re-copied file still hits
     the cache and an edited re-export never does.

Run standalone:
    python helpers/ingest.py raw01.mp4 raw02.mp4 --videos-dir ./videos
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    from .paths import EditPaths
except ImportError:  # running as `python helpers/ingest.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from helpers.paths import EditPaths

# Overridable so a sandbox or a CI box can point at a pinned build.
FFPROBE = os.environ.get("ZETA_FFPROBE", "ffprobe")
FFMPEG = os.environ.get("ZETA_FFMPEG", "ffmpeg")

# WhisperX and every wav2vec2 alignment model run at 16 kHz mono. Extracting at
# anything else means resampling twice.
WAV_SAMPLE_RATE = 16000
WAV_CHANNELS = 1

SHA_CACHE_NAME = ".sha256_cache.json"
SOURCES_NAME = "sources.json"


class IngestError(RuntimeError):
    """A source that cannot be probed, decoded, or has no audio."""


# -- hashing ------------------------------------------------------------------


def sha256_file(path: str | Path, *, cache: str | Path | None = None,
                chunk_bytes: int = 1 << 20) -> str:
    """sha256 of a file, with a (size, mtime_ns) shortcut.

    Hashing a 4 GB take costs seconds and `zeta edit` re-runs constantly during
    a session, so a sidecar cache keyed by size+mtime_ns skips the read when the
    file demonstrably has not been rewritten. size+mtime alone is NOT trusted as
    an identity (a same-size same-mtime overwrite is possible, e.g. restored
    from a backup): it only authorises reusing a hash we computed ourselves.
    """
    p = Path(path).resolve()
    st = p.stat()
    entries: dict[str, Any] = {}
    cache_path = Path(cache) if cache else None

    if cache_path and cache_path.exists():
        try:
            entries = json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            entries = {}  # a corrupt cache is a slow run, never a wrong one
        hit = entries.get(str(p))
        if hit and hit.get("size") == st.st_size and hit.get("mtime_ns") == st.st_mtime_ns:
            return hit["sha256"]

    h = hashlib.sha256()
    with p.open("rb") as fh:
        while block := fh.read(chunk_bytes):
            h.update(block)
    digest = h.hexdigest()

    if cache_path:
        entries[str(p)] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": digest}
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(entries, indent=1), encoding="utf-8")
    return digest


# -- probing ------------------------------------------------------------------


@dataclass(frozen=True)
class Probe:
    """What ffprobe knows about a source. Times in seconds, sizes in pixels."""

    duration_s: float = 0.0
    width: int = 0
    height: int = 0
    # Frame rate as an exact rational. 30000/1001 is not 29.97, and the EDL's
    # cut edges are eventually snapped to frames -- a float here accumulates
    # drift over a ten-minute take.
    fps_num: int = 0
    fps_den: int = 1
    rotation: int = 0
    audio_channels: int | None = None
    sample_rate: int | None = None
    video_codec: str | None = None
    audio_codec: str | None = None
    container: str | None = None

    @property
    def fps(self) -> Fraction:
        return Fraction(self.fps_num, self.fps_den) if self.fps_den else Fraction(0)

    @property
    def fps_float(self) -> float:
        return float(self.fps)

    @property
    def has_audio(self) -> bool:
        return bool(self.audio_channels)

    @property
    def display_size(self) -> tuple[int, int]:
        """Width/height as the player shows them, i.e. after rotation metadata.

        Phone footage is very often stored landscape with a 90 degree display
        matrix. Cropping or fitting overlays against the stored size produces a
        sideways frame, so layout code must use this, never width/height.
        """
        return (self.height, self.width) if self.rotation % 180 else (self.width, self.height)

    def to_dict(self) -> dict:
        d = {
            "duration_s": round(self.duration_s, 3),
            "width": self.width,
            "height": self.height,
            "fps": f"{self.fps_num}/{self.fps_den}",
            "fps_float": round(self.fps_float, 6),
            "rotation": self.rotation,
            "audio_channels": self.audio_channels,
            "sample_rate": self.sample_rate,
            "video_codec": self.video_codec,
            "audio_codec": self.audio_codec,
            "container": self.container,
        }
        return {k: v for k, v in d.items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict) -> "Probe":
        num, _, den = str(d.get("fps", "0/1")).partition("/")
        return cls(
            duration_s=float(d.get("duration_s") or 0.0),
            width=int(d.get("width") or 0),
            height=int(d.get("height") or 0),
            fps_num=int(num or 0),
            fps_den=int(den or 1),
            rotation=int(d.get("rotation") or 0),
            audio_channels=d.get("audio_channels"),
            sample_rate=d.get("sample_rate"),
            video_codec=d.get("video_codec"),
            audio_codec=d.get("audio_codec"),
            container=d.get("container"),
        )


def _rational(value: Any) -> tuple[int, int]:
    """Parse ffprobe's "num/den" rate strings; 0/0 means "unknown"."""
    if not value:
        return (0, 1)
    try:
        f = Fraction(str(value))
    except (ZeroDivisionError, ValueError):
        return (0, 1)
    return (f.numerator, f.denominator)


def _stream_rotation(stream: dict) -> int:
    """Rotation in degrees, normalised to [0, 360)."""
    for sd in stream.get("side_data_list") or []:
        if "rotation" in sd:
            return int(round(float(sd["rotation"]))) % 360
    tag = (stream.get("tags") or {}).get("rotate")
    if tag is not None:
        try:
            return int(round(float(tag))) % 360
        except ValueError:
            return 0
    return 0


def parse_probe(payload: dict) -> Probe:
    """Pure parse of an `ffprobe -print_format json` payload."""
    streams = payload.get("streams") or []
    fmt = payload.get("format") or {}
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    duration = fmt.get("duration")
    if duration in (None, "N/A") and video is not None:
        duration = video.get("duration")
    if duration in (None, "N/A") and audio is not None:
        duration = audio.get("duration")

    num, den = (0, 1)
    if video is not None:
        # r_frame_rate is the stream's nominal rate; avg_frame_rate is derived
        # from the actual packet count and is the honest number for VFR phone
        # footage, so it wins when the two disagree and it is usable.
        num, den = _rational(video.get("r_frame_rate"))
        anum, aden = _rational(video.get("avg_frame_rate"))
        if anum and aden and (anum, aden) != (num, den):
            num, den = anum, aden

    return Probe(
        duration_s=float(duration) if duration not in (None, "N/A") else 0.0,
        width=int(video.get("width") or 0) if video else 0,
        height=int(video.get("height") or 0) if video else 0,
        fps_num=num,
        fps_den=den or 1,
        rotation=_stream_rotation(video) if video else 0,
        audio_channels=int(audio["channels"]) if audio and audio.get("channels") else None,
        sample_rate=int(audio["sample_rate"]) if audio and audio.get("sample_rate") else None,
        video_codec=video.get("codec_name") if video else None,
        audio_codec=audio.get("codec_name") if audio else None,
        container=fmt.get("format_name"),
    )


def probe_source(path: str | Path) -> Probe:
    """Run ffprobe on one file."""
    p = Path(path)
    if not p.exists():
        raise IngestError(f"source not found: {p}")
    cmd = [FFPROBE, "-v", "error", "-print_format", "json",
           "-show_format", "-show_streams", str(p)]
    try:
        out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    except FileNotFoundError as exc:
        raise IngestError(
            f"{FFPROBE} not found. ffmpeg/ffprobe are required for ingest "
            f"(see section 6 of the spec); set ZETA_FFPROBE to override.") from exc
    except subprocess.CalledProcessError as exc:
        raise IngestError(f"ffprobe failed on {p.name}: {exc.stderr.strip()[:400]}") from exc
    return parse_probe(json.loads(out))


# -- audio extraction ---------------------------------------------------------


def extract_wav(src: str | Path, out_wav: str | Path, *, force: bool = False) -> Path:
    """Decode the first audio stream to 16 kHz mono PCM, whole file.

    There is deliberately no `-ss`/`-t` here and there never will be: word
    timestamps are only meaningful on the source timeline (hard rule: the cut
    engine and the caption builder both read them as source time).
    """
    src, out_wav = Path(src), Path(out_wav)
    if out_wav.exists() and not force:
        return out_wav
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_wav.with_suffix(out_wav.suffix + ".part")
    cmd = [
        FFMPEG, "-nostdin", "-v", "error", "-y",
        "-i", str(src),
        "-vn", "-map", "0:a:0",
        "-ac", str(WAV_CHANNELS), "-ar", str(WAV_SAMPLE_RATE),
        "-c:a", "pcm_s16le",
        str(tmp),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise IngestError(
            f"{FFMPEG} not found. ffmpeg is required for ingest; "
            f"set ZETA_FFMPEG to override.") from exc
    except subprocess.CalledProcessError as exc:
        tmp.unlink(missing_ok=True)
        raise IngestError(f"audio extraction failed for {src.name}: "
                          f"{exc.stderr.strip()[:400]}") from exc
    # Rename only on success so a killed run never leaves a truncated WAV that
    # a later run would happily reuse as "already extracted".
    tmp.replace(out_wav)
    return out_wav


# -- sources ------------------------------------------------------------------


@dataclass
class Source:
    """One raw take, resolved."""

    name: str
    path: Path
    sha256: str
    probe: Probe = field(default_factory=Probe)
    wav: Path | None = None

    @property
    def duration_s(self) -> float:
        return self.probe.duration_s

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "path": str(self.path),
            "sha256": self.sha256,
            "probe": self.probe.to_dict(),
            "wav": str(self.wav) if self.wav else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Source":
        return cls(
            name=d["name"],
            path=Path(d["path"]),
            sha256=d["sha256"],
            probe=Probe.from_dict(d.get("probe") or {}),
            wav=Path(d["wav"]) if d.get("wav") else None,
        )

    def words_source_block(self) -> dict:
        """The `source` block embedded in every words.json."""
        return {"name": self.name, "path": str(self.path), "sha256": self.sha256,
                "duration_s": round(self.probe.duration_s, 3)}


def _unique_name(stem: str, taken: set[str]) -> str:
    if stem not in taken:
        return stem
    i = 2
    while f"{stem}_{i}" in taken:
        i += 1
    return f"{stem}_{i}"


def load_sources(edit_paths: EditPaths) -> list[Source]:
    """Read back `edit/sources.json` (empty list when absent)."""
    p = edit_paths.edit / SOURCES_NAME
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    return [Source.from_dict(s) for s in data.get("sources", [])]


def ingest(
    paths: Sequence[str | Path] | Iterable[str | Path],
    edit_paths: EditPaths,
    *,
    extract_audio: bool = True,
    force: bool = False,
) -> list[Source]:
    """Probe and de-mux every source, then write `edit/sources.json`.

    Re-running on unchanged files is close to free: the hash comes from the
    size+mtime cache and the WAV is reused unless its source hash moved.
    """
    edit_paths.ensure()
    sha_cache = edit_paths.edit / SHA_CACHE_NAME
    previous = {s.sha256: s for s in load_sources(edit_paths)}

    sources: list[Source] = []
    taken: set[str] = set()
    for raw in paths:
        p = Path(raw).expanduser().resolve()
        name = _unique_name(p.stem, taken)
        taken.add(name)

        digest = sha256_file(p, cache=sha_cache)
        probe = probe_source(p)
        if not probe.has_audio:
            raise IngestError(f"{p.name} has no audio stream: nothing to transcribe")

        wav: Path | None = None
        if extract_audio:
            wav = edit_paths.audio / f"{name}.16k.wav"
            prior = previous.get(digest)
            # Hard rule 9 in its cheapest form: same bytes in, same WAV out.
            reusable = (not force and prior is not None and prior.wav
                        and Path(prior.wav) == wav and wav.exists())
            if not reusable:
                extract_wav(p, wav, force=force)

        sources.append(Source(name=name, path=p, sha256=digest, probe=probe, wav=wav))

    write_sources(sources, edit_paths)
    return sources


def write_sources(sources: Sequence[Source], edit_paths: EditPaths) -> Path:
    out = edit_paths.edit / SOURCES_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "zeta.sources.v1",
        "videos_dir": str(edit_paths.videos_dir),
        "sources": [s.to_dict() for s in sources],
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


# -- cli ----------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Probe raw takes and extract 16 kHz mono audio (Stage 0).")
    ap.add_argument("sources", nargs="+", help="raw video files")
    ap.add_argument("--videos-dir", default=None,
                    help="output root; defaults to the first source's directory")
    ap.add_argument("--no-audio", action="store_true", help="probe only, skip WAV extraction")
    ap.add_argument("--force", action="store_true", help="re-extract audio even if cached")
    args = ap.parse_args(argv)

    edit_paths = EditPaths.for_videos_dir(args.videos_dir or Path(args.sources[0]).parent)
    try:
        sources = ingest(args.sources, edit_paths,
                         extract_audio=not args.no_audio, force=args.force)
    except IngestError as exc:
        print(f"ingest failed: {exc}", file=sys.stderr)
        return 1

    for s in sources:
        p = s.probe
        w, h = p.display_size
        print(f"{s.name:20s} {w}x{h} @ {p.fps_num}/{p.fps_den} "
              f"({p.fps_float:.3f} fps) rot={p.rotation} {p.duration_s:8.2f}s "
              f"{p.audio_channels}ch/{p.sample_rate}Hz sha={s.sha256[:12]}")
    print(f"\nwrote {edit_paths.edit / SOURCES_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
