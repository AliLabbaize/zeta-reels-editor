"""The gate that makes `--auto` safe.

Unattended batch mode cuts video from `words.json` without anyone looking at it
first. Three defects in a transcript produce a bad cut silently:

  * LOW COVERAGE - words with no timing are words the cut engine cannot place,
    so a phrase can be dropped or kept wholesale by accident.
  * OVERLAPS - two words claiming the same instant means the alignment slipped;
    every boundary after the slip is suspect, and a cut edge snapped to one of
    them lands inside a syllable.
  * LONG WORDS - a two-second "word" is the aligner having absorbed a pause or
    a stretch of speech it could not read. Padding a cut against that edge puts
    the cut in the middle of the neighbouring word.

So: check, re-align the flagged windows ONCE (a second automatic pass would be
a retry loop hiding a real problem), check again, and fail the run if it is
still bad. `main()` exits non-zero, which is what stops a batch.

Run standalone:
    python helpers/qa_words.py edit/transcripts/raw01.words.json edit/audio/raw01.16k.wav
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Sequence

try:
    from . import align_whisperx, config as cfgmod
    from .paths import EditPaths
    from .words import WordsDoc
except ImportError:  # running as `python helpers/qa_words.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from helpers import align_whisperx, config as cfgmod
    from helpers.paths import EditPaths
    from helpers.words import WordsDoc

SCHEMA = "zeta.qa.v1"

# Reports are read by humans and committed to `project.md`; a transcript that
# fails on 4000 words must not produce a 4000-entry JSON file.
MAX_LISTED = 50
# A repair window that keeps growing is a misalignment, not a local defect.
MAX_WINDOW_WORDS = 60


def thresholds(cfg: dict | None = None) -> dict:
    qa = cfgmod.get(cfg or {}, "qa") or {}
    return {
        "min_coverage": float(qa.get("min_coverage", 0.95)),
        "max_word_duration_s": float(qa.get("max_word_duration_s", 2.0)),
        "allow_overlaps": bool(qa.get("allow_overlaps", False)),
        "realign_flagged_windows": bool(qa.get("realign_flagged_windows", True)),
        "window_pad_s": float(qa.get("window_pad_s", 2.0)),
    }


def check(doc: WordsDoc, cfg: dict | None = None) -> dict:
    """Pure inspection of a WordsDoc against the QA thresholds."""
    th = thresholds(cfg)
    coverage = doc.coverage()
    untimed = [i for i, w in enumerate(doc.words) if not w.timed]
    overlaps = doc.overlaps()
    long_words = doc.long_words(th["max_word_duration_s"])

    failures: list[str] = []
    if not doc.words:
        failures.append("transcript is empty")
    if coverage < th["min_coverage"]:
        failures.append(f"coverage {coverage:.3f} < min_coverage {th['min_coverage']:.3f}")
    if overlaps and not th["allow_overlaps"]:
        failures.append(f"{len(overlaps)} overlapping word pair(s)")
    if long_words:
        failures.append(f"{len(long_words)} word(s) longer than "
                        f"{th['max_word_duration_s']:.2f}s")

    return {
        "schema": SCHEMA,
        "source": doc.source,
        "backend": doc.backend,
        "aligner": doc.aligner,
        "thresholds": th,
        "words": len(doc.words),
        "timed": len(doc.timed_words()),
        "coverage": round(coverage, 4),
        "untimed_count": len(untimed),
        "untimed": untimed[:MAX_LISTED],
        "overlaps": [
            {"a": a, "b": b,
             "a_word": doc.words[a].display or doc.words[a].word,
             "b_word": doc.words[b].display or doc.words[b].word,
             "a_end": doc.words[a].end, "b_start": doc.words[b].start}
            for a, b in overlaps[:MAX_LISTED]
        ],
        "overlap_count": len(overlaps),
        "long_words": [
            {"index": i,
             "word": doc.words[i].display or doc.words[i].word,
             "start": doc.words[i].start, "end": doc.words[i].end,
             "duration": round(doc.words[i].duration, 3)}
            for i in long_words[:MAX_LISTED]
        ],
        "long_word_count": len(long_words),
        "uncertain_spans": len(doc.meta.get("uncertain") or []),
        "failures": failures,
        "ok": not failures,
    }


def flagged_indices(doc: WordsDoc, cfg: dict | None = None) -> list[int]:
    """Word indices that something is wrong with (untimed, overlapping, long)."""
    th = thresholds(cfg)
    flagged: set[int] = {i for i, w in enumerate(doc.words) if not w.timed}
    for a, b in doc.overlaps():
        flagged.update((a, b))
    flagged.update(doc.long_words(th["max_word_duration_s"]))
    return sorted(flagged)


def expand_window(doc: WordsDoc, i: int, j: int, pad_s: float) -> tuple[int, int]:
    """Grow `[i, j)` outwards to cover `pad_s` of speech on each side.

    Alignment is only as good as its context: handing the model the two seconds
    around a defect gives it real acoustic anchors instead of a bare fragment.
    """
    lo, hi = i, j
    timed = [w for w in doc.words[i:j] if w.timed]
    t0 = timed[0].start if timed else None
    t1 = timed[-1].end if timed else None

    while lo > 0 and (j - lo) < MAX_WINDOW_WORDS:
        prev = doc.words[lo - 1]
        if prev.timed and t0 is not None and prev.end < t0 - pad_s:
            break
        lo -= 1
        if prev.timed and t0 is None:
            t0 = prev.start
    while hi < len(doc.words) and (hi - i) < MAX_WINDOW_WORDS:
        nxt = doc.words[hi]
        if nxt.timed and t1 is not None and nxt.start > t1 + pad_s:
            break
        hi += 1
        if nxt.timed and t1 is None:
            t1 = nxt.end
    return (lo, hi)


def repair_windows(doc: WordsDoc, indices: Sequence[int],
                   cfg: dict | None = None) -> list[tuple[int, int]]:
    """Merge flagged indices into as few re-alignment windows as possible."""
    if not indices:
        return []
    pad_s = thresholds(cfg)["window_pad_s"]
    windows: list[tuple[int, int]] = []
    for idx in indices:
        lo, hi = expand_window(doc, idx, idx + 1, pad_s)
        if windows and lo <= windows[-1][1]:
            windows[-1] = (windows[-1][0], max(windows[-1][1], hi))
        else:
            windows.append((lo, hi))
    return windows


RealignFn = Callable[..., int]


def qa(
    doc: WordsDoc,
    wav: str | Path | None = None,
    cfg: dict | None = None,
    *,
    edit_paths: EditPaths | None = None,
    name: str | None = None,
    realign: RealignFn | None = None,
) -> dict:
    """Check, repair once, re-check, write `edit/transcripts/<name>.qa.json`."""
    cfg = cfg or {}
    report = check(doc, cfg)
    report["passes"] = 1
    report["realigned_windows"] = []

    th = thresholds(cfg)
    can_repair = bool(wav) and th["realign_flagged_windows"] and not report["ok"]
    if can_repair:
        fn = realign or align_whisperx.realign_window
        windows = repair_windows(doc, flagged_indices(doc, cfg), cfg)
        repaired: list[dict] = []
        for lo, hi in windows:
            try:
                timed = fn(doc, wav, lo, hi, cfg, pad_s=th["window_pad_s"])
                err = None
            except Exception as exc:  # a failed repair is a QA finding, not a crash
                timed, err = 0, f"{type(exc).__name__}: {exc}"
            entry = {"start_index": lo, "end_index": hi, "words_timed": timed}
            if err:
                entry["error"] = err
            repaired.append(entry)
        before = report
        report = check(doc, cfg)
        report["passes"] = 2
        report["realigned_windows"] = repaired
        report["before_repair"] = {
            "coverage": before["coverage"],
            "overlap_count": before["overlap_count"],
            "long_word_count": before["long_word_count"],
            "failures": before["failures"],
        }

    out_name = name or (doc.source or {}).get("name") or "transcript"
    if edit_paths is not None:
        report["report_path"] = str(write_report(report, edit_paths, out_name))
    return report


def write_report(report: dict, edit_paths: EditPaths, name: str) -> Path:
    edit_paths.transcripts.mkdir(parents=True, exist_ok=True)
    p = edit_paths.transcripts / f"{name}.qa.json"
    p.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return p


def format_report(report: dict) -> str:
    lines = [
        f"words              {report['timed']}/{report['words']} timed",
        f"coverage           {report['coverage'] * 100:.2f}% "
        f"(min {report['thresholds']['min_coverage'] * 100:.0f}%)",
        f"overlaps           {report['overlap_count']}",
        f"long words         {report['long_word_count']} "
        f"(> {report['thresholds']['max_word_duration_s']:.2f}s)",
        f"uncertain spans    {report['uncertain_spans']}",
        f"passes             {report.get('passes', 1)}",
    ]
    for w in report.get("realigned_windows") or []:
        lines.append(f"  re-aligned [{w['start_index']}:{w['end_index']}) "
                     f"-> {w['words_timed']} timed"
                     + (f"  {w['error']}" if w.get("error") else ""))
    for f in report["failures"]:
        lines.append(f"FAIL  {f}")
    if report["ok"]:
        lines.append("OK")
    return "\n".join(lines)


# -- cli ----------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Quality gate for a words.json. Non-zero exit = do not render.")
    ap.add_argument("words_json")
    ap.add_argument("wav", nargs="?", default=None,
                    help="WAV to re-align flagged windows from (optional)")
    ap.add_argument("--config", default="transcribe")
    ap.add_argument("--videos-dir", default=None,
                    help="where to write the .qa.json (default: alongside words_json)")
    ap.add_argument("--no-repair", action="store_true", help="check only, never re-align")
    ap.add_argument("--save", action="store_true",
                    help="write the repaired words.json back over the input")
    args = ap.parse_args(argv)

    cfg = cfgmod.load(args.config)
    if args.no_repair:
        cfg.setdefault("qa", {})["realign_flagged_windows"] = False

    doc = WordsDoc.load(args.words_json)
    words_path = Path(args.words_json)
    # transcripts/<name>.words.json -> the edit dir is two levels up.
    edit_paths = EditPaths.for_videos_dir(
        args.videos_dir or words_path.resolve().parent.parent.parent)
    name = words_path.name.replace(".words.json", "")

    report = qa(doc, args.wav, cfg, edit_paths=edit_paths, name=name)
    if args.save and report.get("passes", 1) > 1:
        doc.save(words_path)
    print(format_report(report))
    print(f"\nwrote {report.get('report_path')}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
