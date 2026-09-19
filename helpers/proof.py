"""Still frames of what a render WILL look like, in seconds instead of minutes.

Every layout mistake this pipeline shipped (a card over Ali's mouth, captions
sitting on a card's last line, a screenshot too small to read) cost a full
render to discover. A proof frame is the real video frame at the moment of a
card, with the real overlay composited and the real .ass captions burned, so
the layout is judged before anything is encoded.

    zeta proof -v <videos_dir>            # one PNG per insert, plus a sheet
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from helpers import config as configs
from helpers.edl import EDL
from helpers.paths import EditPaths


def proof_frame(e: EDL, edit_dir: Path, overlay: dict, out_png: Path,
                *, at: float | None = None, runner=subprocess.run) -> Path:
    """One frame: source at that moment + this overlay + the burned captions."""
    t_out = float(overlay["start_in_output"]) + (1.0 if at is None else at)
    name, t_src = e.to_source_time(t_out)
    src = e.sources[name]
    ov = Path(overlay["file"])
    if not ov.is_absolute():
        ov = edit_dir / ov
    frame = ov.with_name("overlay_frame.png")
    acfg = configs.aspect_config(e.aspect)[1]
    w, h = acfg["resolution"]
    chain = [f"[0:v]scale={w}:{h},setsar=1,setpts=PTS-STARTPTS+{t_out:.3f}/TB[v]"]
    last = "[v]"
    if frame.exists():
        chain.append(f"{last}[1:v]overlay=0:0[ov]")
        last = "[ov]"
    if e.subtitles:
        subs = (edit_dir / e.subtitles) if not Path(e.subtitles).is_absolute() else Path(e.subtitles)
        if subs.exists():
            esc = str(subs.resolve()).replace(":", r"\:").replace("'", r"\'")
            chain.append(f"{last}subtitles='{esc}'[out]")
            last = "[out]"
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{t_src:.3f}", "-i", str(src)]
    if frame.exists():
        cmd += ["-i", str(frame)]
    cmd += ["-filter_complex", ";".join(chain), "-map", last, "-frames:v", "1", str(out_png)]
    runner(cmd, check=True)
    return out_png


def proof_sheet(paths: EditPaths, edl_path: Path | None = None,
                out_png: Path | None = None) -> Path:
    """A contact sheet of every insert, side by side."""
    e = EDL.load(edl_path or paths.edl)
    edit_dir = (edl_path or paths.edl).parent
    shots_dir = paths.edit / "proof"
    shots_dir.mkdir(parents=True, exist_ok=True)
    shots = []
    for i, ov in enumerate(json.loads((edl_path or paths.edl).read_text())["overlays"]):
        shots.append(proof_frame(e, edit_dir, ov, shots_dir / f"{i:02d}.png"))
    out_png = out_png or shots_dir / "sheet.png"
    if shots:
        cols = min(len(shots), 6)
        cmd = ["ffmpeg", "-v", "error", "-y"]
        for s in shots:
            cmd += ["-i", str(s)]
        n = len(shots)
        rows = (n + cols - 1) // cols
        streams = "".join(f"[{i}:v]scale=270:-1[s{i}];" for i in range(n))
        cmd += ["-filter_complex",
                streams + "".join(f"[s{i}]" for i in range(n)) +
                f"xstack=inputs={n}:layout=" + "|".join(
                    f"{(i % cols)}_{(i // cols)}" if False else
                    f"{(i % cols) * 270}_{(i // cols) * 480}" for i in range(n)) + "[out]"
                if n > 1 else streams + "[s0]null[out]",
                "-map", "[out]", "-frames:v", "1", str(out_png)]
        subprocess.run(cmd, check=False)
    return out_png


def main() -> None:
    ap = argparse.ArgumentParser(description="Proof frames for every insert")
    ap.add_argument("-v", "--videos-dir", required=True)
    ap.add_argument("--edl", default=None)
    args = ap.parse_args()
    paths = EditPaths.for_videos_dir(args.videos_dir)
    out = proof_sheet(paths, Path(args.edl) if args.edl else None)
    print(out)


if __name__ == "__main__":
    main()
