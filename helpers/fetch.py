"""Pull a published video down by URL, for learn mode.

`yt-dlp` does the work. Two things this wrapper adds:

* It records the source URL next to the file, because six months from now
  "ep14.mp4" does not say which post it was.
* It can append the row to `pairs.csv` for you, in published-only form, so the
  file learn mode reads stays consistent.

WHAT A PUBLISHED-ONLY ROW CAN AND CANNOT TEACH
----------------------------------------------
It teaches insert density, insert duration, layout, PiP corner and which kinds
of mention trigger a visual. It cannot teach the cut ratio, the filler policy or
the retake policy, because a finished video does not record what was removed.
Those need the raw take. Learn mode says so in the profile it writes.

Network access to Instagram and TikTok is commonly blocked (datacenter IP
ranges, login walls, or an agent proxy's allowlist). When the download fails
that is usually the reason, not a broken URL; download on a machine that can
reach the platform and point `zeta learn` at the file instead.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

try:
    from .paths import EditPaths
except ImportError:  # running as `python helpers/fetch.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from helpers.paths import EditPaths


class FetchError(RuntimeError):
    pass


def yt_dlp_cmd(url: str, out_dir: Path, *, name: str | None = None,
               cookies_from_browser: str | None = None) -> list[str]:
    """argv for one download. Pure, so the flags are testable without a network."""
    template = f"{name}.%(ext)s" if name else "%(title).80B-%(id)s.%(ext)s"
    cmd = [
        "yt-dlp", "--no-playlist", "--restrict-filenames",
        # Merge to mp4: the pipeline probes and cuts one container, and a
        # webm/m4a pair would make the published side of a pair a special case.
        "--merge-output-format", "mp4",
        "--write-info-json",
        "-o", str(out_dir / template),
    ]
    if cookies_from_browser:
        # Instagram in particular serves most posts only to a logged-in session.
        cmd += ["--cookies-from-browser", cookies_from_browser]
    cmd.append(url)
    return cmd


def _newest(out_dir: Path, before: set[Path]) -> Path | None:
    made = [p for p in out_dir.glob("*.mp4") if p not in before]
    return max(made, key=lambda p: p.stat().st_mtime) if made else None


def fetch(url: str, out_dir: str | Path, *, name: str | None = None,
          cookies_from_browser: str | None = None,
          runner=subprocess.run) -> Path:
    """Download one video; returns the file. Raises FetchError with the reason."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if shutil.which("yt-dlp") is None:
        raise FetchError("yt-dlp is not installed: uv pip install yt-dlp")

    before = set(out_dir.glob("*.mp4"))
    proc = runner(yt_dlp_cmd(url, out_dir, name=name,
                             cookies_from_browser=cookies_from_browser),
                  capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        raise FetchError(
            f"yt-dlp failed for {url}:\n  " + "\n  ".join(tail) +
            "\n\nInstagram and TikTok often refuse datacenter IPs or require a "
            "logged-in session. Try --cookies-from-browser, or download on a "
            "machine that can reach the platform and pass the file directly.")

    video = _newest(out_dir, before)
    if video is None:
        raise FetchError(f"yt-dlp reported success but produced no mp4 in {out_dir}")

    (video.with_suffix(".source.json")).write_text(
        json.dumps({"url": url, "file": video.name}, indent=1), encoding="utf-8")
    return video


def append_pair(pairs_csv: str | Path, published: str | Path, *,
                raw: str | Path | None = None, name: str = "") -> Path:
    """Append a `raw,published,name` row, writing the header on first use."""
    pairs = Path(pairs_csv)
    pairs.parent.mkdir(parents=True, exist_ok=True)
    fresh = not pairs.exists() or not pairs.read_text(encoding="utf-8").strip()
    base = pairs.resolve().parent

    def rel(p) -> str:
        p = Path(p).resolve()
        try:
            return str(p.relative_to(base))
        except ValueError:
            return str(p)

    with pairs.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        if fresh:
            writer.writerow(["raw", "published", "name"])
        writer.writerow([rel(raw) if raw else "", rel(published), name])
    return pairs


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Download a published Zeta video for learn mode.")
    ap.add_argument("urls", nargs="+", help="Instagram / TikTok / YouTube URLs")
    ap.add_argument("-v", "--videos-dir", required=True,
                    help="where to put the downloads")
    ap.add_argument("--name", help="basename for a single download (e.g. ep14)")
    ap.add_argument("--pairs", help="pairs.csv to append a published-only row to")
    ap.add_argument("--raw", help="the matching raw take, if you have it "
                                  "(this is what teaches the cut ratio)")
    ap.add_argument("--cookies-from-browser",
                    help="chrome | firefox | safari - for posts that need a login")
    args = ap.parse_args(argv)

    out_dir = Path(args.videos_dir)
    for i, url in enumerate(args.urls):
        name = args.name if (args.name and len(args.urls) == 1) else None
        try:
            video = fetch(url, out_dir, name=name,
                          cookies_from_browser=args.cookies_from_browser)
        except FetchError as exc:
            print(f"zeta: {exc}", file=sys.stderr)
            return 1
        print(f"zeta: {video}")
        if args.pairs:
            append_pair(args.pairs, video, raw=args.raw if i == 0 else None,
                        name=name or video.stem)
            print(f"zeta: row appended to {args.pairs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
