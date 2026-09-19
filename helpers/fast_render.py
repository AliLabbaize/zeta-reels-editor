"""Render without redoing work that has not changed.

`helpers/render.py` is vendored and never edited (docs/VENDOR.md), so this
module CALLS its functions in the same order it does:

    extract per segment -> lossless concat -> overlays -> captions LAST

The only thing added is a cache: cutting the segments out of the raw take is
about half the render, and it depends on nothing but the ranges, the sources
and the frame rate. Change a card or a caption and those are identical, so the
base is reused and a revision costs the composite pass alone (~2 min -> ~1 min
on a 3 min part). `zeta render --fresh` forces the extraction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from helpers import render as vendored


def base_key(edl: dict, fps: str | None, preview: bool, draft: bool) -> str:
    """Everything the base video depends on, and nothing else."""
    payload = {
        "ranges": [[r.get("source"), round(float(r["start"]), 3), round(float(r["end"]), 3)]
                   for r in edl.get("ranges") or []],
        "sources": edl.get("sources"),
        "grade": edl.get("grade"),
        "fps": str(fps), "preview": preview, "draft": draft,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def render(edl_path: str | Path, out_path: str | Path, *, fps: str | None = None,
           preview: bool = False, draft: bool = False, no_loudnorm: bool = False,
           reuse_base: bool = True, quiet: bool = False) -> Path:
    edl_path, out_path = Path(edl_path).resolve(), Path(out_path).resolve()
    edl = json.loads(edl_path.read_text(encoding="utf-8"))
    edit_dir = edl_path.parent
    name = "base_draft.mp4" if draft else "base_preview.mp4" if preview else "base.mp4"
    base_path, stamp = edit_dir / name, edit_dir / (name + ".key")
    key = base_key(edl, fps, preview, draft)

    if reuse_base and base_path.exists() and stamp.exists() \
            and stamp.read_text(encoding="utf-8").strip() == key:
        if not quiet:
            print(f"reusing {name} (same ranges, sources and fps)")
    else:
        segments = vendored.extract_all_segments(edl, edit_dir, preview=preview,
                                                 draft=draft, fps=fps)
        vendored.concat_segments(segments, base_path, edit_dir)
        stamp.write_text(key, encoding="utf-8")

    subs = None
    if edl.get("subtitles"):
        subs = vendored.resolve_path(edl["subtitles"], edit_dir)
        if not subs.exists():
            print(f"warning: subtitles path in EDL does not exist: {subs}")
            subs = None

    overlays = edl.get("overlays") or []
    if no_loudnorm:
        vendored.build_final_composite(base_path, overlays, subs, out_path, edit_dir)
    else:
        tmp = out_path.with_suffix(".prenorm.mp4")
        vendored.build_final_composite(base_path, overlays, subs, tmp, edit_dir)
        vendored.apply_loudnorm_two_pass(tmp, out_path, preview=preview or draft)
        tmp.unlink(missing_ok=True)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Render an EDL, reusing the cut segments")
    ap.add_argument("edl")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--fps", default=None)
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--draft", action="store_true")
    ap.add_argument("--no-loudnorm", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="re-cut the segments")
    a = ap.parse_args()
    print(render(a.edl, a.output, fps=a.fps, preview=a.preview, draft=a.draft,
                 no_loudnorm=a.no_loudnorm, reuse_base=not a.fresh))


if __name__ == "__main__":
    main()
