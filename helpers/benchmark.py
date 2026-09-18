"""The Stage 1 benchmark: pick the default transcriber from numbers, not vibes.

Four variants, the ones the spec names:

    a  gemini-3.1-flash-lite + WhisperX      (the September 2026 v1 pipeline)
    b  gemini-3.5-flash-lite + WhisperX      (v2 candidate default)
    c  gemini-3.5-transcribe, words only     (model-emitted word timestamps)
    d  gemini-3.5-transcribe text + WhisperX (model text, acoustic timing)

Two numbers per variant, because they fail differently and a pipeline needs
both to be good:

  * WER on human-corrected references, computed on NORMALISED tokens. Darija
    has no settled orthography; scoring raw strings would mark "الشركه" against
    "الشركة" as an error and rank transcribers by spelling convention instead
    of by hearing. `textnorm.normalize_token` is the same normaliser the cut
    engine diffs with, so the WER measured here is the error rate that actually
    reaches the edit.
  * Alignment coverage, i.e. the fraction of words that ended up with a usable
    timestamp. A transcript with perfect WER that the aligner cannot place is
    useless to the cut engine -- variant (c) is in the table precisely because
    Google documents that word timestamps cost accuracy.

Run standalone:
    python helpers/benchmark.py --manifest refs.csv --videos-dir ./videos
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

try:
    from . import config as cfgmod
    from . import textnorm
    from .paths import EditPaths
    from .words import WordsDoc
except ImportError:  # running as `python helpers/benchmark.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from helpers import config as cfgmod
    from helpers import textnorm
    from helpers.paths import EditPaths
    from helpers.words import WordsDoc


# -- WER ----------------------------------------------------------------------


def levenshtein_counts(ref: Sequence[str], hyp: Sequence[str]) -> tuple[int, int, int]:
    """`(substitutions, deletions, insertions)` between two token sequences.

    Full DP with a backtrace: WER alone hides which way a backend is wrong, and
    deletions vs insertions is the difference between a model that drops fast
    Darija and one that hallucinates filler.
    """
    n, m = len(ref), len(hyp)
    # cost[i][j] and the operation that produced it: 0 match, 1 sub, 2 del, 3 ins
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    op = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0], op[i][0] = i, 2
    for j in range(1, m + 1):
        cost[0][j], op[0][j] = j, 3
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                cost[i][j], op[i][j] = cost[i - 1][j - 1], 0
                continue
            sub, dele, ins = cost[i - 1][j - 1] + 1, cost[i - 1][j] + 1, cost[i][j - 1] + 1
            best = min(sub, dele, ins)
            cost[i][j] = best
            op[i][j] = 1 if best == sub else (2 if best == dele else 3)

    s = d = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        o = op[i][j]
        if o == 0:
            i, j = i - 1, j - 1
        elif o == 1:
            s += 1
            i, j = i - 1, j - 1
        elif o == 2:
            d += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return (s, d, ins)


def wer(reference: str, hypothesis: str) -> dict:
    """Word error rate on normalised tokens, with the per-class counts.

    jiwer when it is installed (it is a dev dependency and its DP is C-fast on
    long takes); the built-in above otherwise, so `zeta benchmark` runs on a
    bare interpreter. Both are fed the SAME normalised tokens, so the numbers
    are comparable either way.
    """
    ref = textnorm.normalize_tokens(reference)
    hyp = textnorm.normalize_tokens(hypothesis)
    n = len(ref)

    try:
        import jiwer
    except ImportError:
        s, d, i = levenshtein_counts(ref, hyp)
        backend = "builtin"
    else:
        # Pre-normalised tokens, so no jiwer transform: it must not re-tokenise
        # or case-fold Arabic on its own.
        out = jiwer.process_words([" ".join(ref)], [" ".join(hyp)])
        s, d, i = out.substitutions, out.deletions, out.insertions
        backend = "jiwer"

    return {"wer": (s + d + i) / n if n else 0.0,
            "substitutions": s, "deletions": d, "insertions": i,
            "ref_words": n, "hyp_words": len(hyp), "scorer": backend}


# -- variants -----------------------------------------------------------------


@dataclass
class Variant:
    id: str
    label: str
    backend: str
    overrides: dict = field(default_factory=dict)
    align: bool = True

    def config(self, base: dict) -> dict:
        cfg = cfgmod.deep_merge(base, self.overrides)
        cfg["default_backend"] = self.backend
        return cfg


DEFAULT_VARIANTS: list[Variant] = [
    Variant("a", "v1: gemini-3.1-flash-lite + WhisperX", "gemini_flash_lite",
            {"backends": {"gemini_flash_lite": {"model": "gemini-3.1-flash-lite"}},
             "use_backend_timings": False}),
    Variant("b", "gemini-3.5-flash-lite + WhisperX", "gemini_flash_lite",
            {"backends": {"gemini_flash_lite": {"model": "gemini-3.5-flash-lite"}},
             "use_backend_timings": False}),
    Variant("c", "gemini-3.5-transcribe, words only", "gemini_transcribe",
            {"use_backend_timings": True}, align=False),
    Variant("d", "gemini-3.5-transcribe text + WhisperX", "gemini_transcribe",
            {"use_backend_timings": False}),
]


@dataclass
class Sample:
    """One clip with a human-corrected transcript."""

    name: str
    wav: Path
    reference: str


def load_manifest(path: str | Path) -> list[Sample]:
    """Read `name,wav,reference` rows (CSV) or the same shape as JSON.

    `reference` is either the text itself or a path to a .txt file next to the
    manifest -- human-corrected transcripts live in files, not in cells.
    """
    p = Path(path)
    if p.suffix.lower() == ".json":
        rows = json.loads(p.read_text(encoding="utf-8"))
        rows = rows.get("samples", rows) if isinstance(rows, dict) else rows
    else:
        with p.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))

    out: list[Sample] = []
    for row in rows:
        ref = str(row["reference"]).strip()
        cand = (p.parent / ref) if not Path(ref).is_absolute() else Path(ref)
        if len(ref) < 512 and cand.exists():
            ref = cand.read_text(encoding="utf-8")
        wav = Path(row["wav"])
        out.append(Sample(name=str(row.get("name") or wav.stem),
                          wav=wav if wav.is_absolute() else (p.parent / wav),
                          reference=ref))
    return out


# -- running ------------------------------------------------------------------

TranscribeFn = Callable[..., WordsDoc]
AlignFn = Callable[..., WordsDoc]


def _default_transcribe(wav: Path, cfg: dict, backend: str) -> WordsDoc:
    from . import transcribe_gemini  # lazy: keeps `wer()` importable anywhere
    return transcribe_gemini.transcribe(wav, cfg, None, backend=backend)


def _default_align(doc: WordsDoc, wav: Path, cfg: dict) -> WordsDoc:
    from . import align_whisperx
    return align_whisperx.align(doc, wav, cfg)


def score_doc(sample: Sample, doc: WordsDoc, max_word_duration_s: float = 2.0) -> dict:
    row = wer(sample.reference, doc.text())
    row.update({
        "sample": sample.name,
        "coverage": round(doc.coverage(), 4),
        "overlaps": len(doc.overlaps()),
        "long_words": len(doc.long_words(max_word_duration_s)),
        "aligner": doc.aligner,
    })
    return row


def run_variant(
    variant: Variant,
    samples: Sequence[Sample],
    base_cfg: dict,
    *,
    transcribe_fn: TranscribeFn | None = None,
    align_fn: AlignFn | None = None,
) -> dict:
    cfg = variant.config(base_cfg)
    tfn = transcribe_fn or (lambda wav, c, backend: _default_transcribe(wav, c, backend))
    afn = align_fn or _default_align
    max_dur = float(cfgmod.get(cfg, "qa.max_word_duration_s", 2.0))

    rows: list[dict] = []
    for sample in samples:
        doc = tfn(sample.wav, cfg, variant.backend)
        if variant.align:
            doc = afn(doc, sample.wav, cfg) or doc
        rows.append(score_doc(sample, doc, max_dur))

    n = sum(r["ref_words"] for r in rows)
    errors = sum(r["substitutions"] + r["deletions"] + r["insertions"] for r in rows)
    words = sum(r["hyp_words"] for r in rows) or 1
    return {
        "id": variant.id,
        "label": variant.label,
        "backend": variant.backend,
        "model": cfgmod.get(cfg, f"backends.{variant.backend}.model"),
        "timing": "backend" if not variant.align else "whisperx",
        # Corpus WER: total errors over total reference words, NOT the mean of
        # per-clip WERs, which would let a short clip outvote a long one.
        "wer": errors / n if n else 0.0,
        "substitutions": sum(r["substitutions"] for r in rows),
        "deletions": sum(r["deletions"] for r in rows),
        "insertions": sum(r["insertions"] for r in rows),
        "coverage": sum(r["coverage"] * r["hyp_words"] for r in rows) / words,
        "overlaps": sum(r["overlaps"] for r in rows),
        "long_words": sum(r["long_words"] for r in rows),
        "samples": rows,
    }


def pick_winner(results: Sequence[dict], min_coverage: float = 0.95) -> dict | None:
    """Lowest WER among variants that clear the coverage bar; else lowest WER.

    Coverage is a gate rather than a weight: below it, `qa_words` refuses the
    run, so a variant that cannot be aligned cannot be the default no matter how
    well it reads.
    """
    if not results:
        return None
    usable = [r for r in results if r["coverage"] >= min_coverage and not r["overlaps"]]
    return min(usable or list(results), key=lambda r: r["wer"])


def render_table(results: Sequence[dict], winner: dict | None = None) -> str:
    lines = [
        "# Stage 1 transcription benchmark",
        "",
        "WER on normalised tokens (`helpers/textnorm.py`); coverage = words with "
        "a usable timestamp.",
        "",
        "| # | Variant | Model | Timing | WER | Sub | Del | Ins | Coverage | Overlaps | >max words |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        lines.append(
            f"| {r['id']} | {r['label']} | `{r['model']}` | {r['timing']} | "
            f"{r['wer'] * 100:.1f}% | {r['substitutions']} | {r['deletions']} | "
            f"{r['insertions']} | {r['coverage'] * 100:.1f}% | {r['overlaps']} | "
            f"{r['long_words']} |")
    if winner:
        lines += [
            "",
            f"**Winner: ({winner['id']}) {winner['label']}** - WER "
            f"{winner['wer'] * 100:.1f}%, coverage {winner['coverage'] * 100:.1f}%.",
            "",
            "Set in `configs/transcribe.yaml`:",
            "",
            "```yaml",
            f"default_backend: {winner['backend']}",
            f"use_backend_timings: {str(winner['timing'] == 'backend').lower()}",
            "```",
        ]
    return "\n".join(lines) + "\n"


def run_benchmark(
    samples: Sequence[Sample],
    variants: Sequence[Variant],
    base_cfg: dict,
    edit_paths: EditPaths | None = None,
    *,
    transcribe_fn: TranscribeFn | None = None,
    align_fn: AlignFn | None = None,
    write: bool = True,
) -> dict:
    results = [run_variant(v, samples, base_cfg,
                           transcribe_fn=transcribe_fn, align_fn=align_fn)
               for v in variants]
    winner = pick_winner(results, float(cfgmod.get(base_cfg, "qa.min_coverage", 0.95)))
    payload = {
        "schema": "zeta.benchmark.v1",
        "samples": [s.name for s in samples],
        "results": results,
        "winner": winner["id"] if winner else None,
        "markdown": render_table(results, winner),
    }
    if write and edit_paths is not None:
        edit_paths.edit.mkdir(parents=True, exist_ok=True)
        (edit_paths.edit / "benchmark.md").write_text(payload["markdown"], encoding="utf-8")
        (edit_paths.edit / "benchmark.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return payload


# -- cli ----------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Benchmark transcription variants against human-corrected "
                    "references (Stage 1).")
    ap.add_argument("--manifest", required=True,
                    help="CSV/JSON with name,wav,reference columns")
    ap.add_argument("--videos-dir", default=None, help="where benchmark.md is written")
    ap.add_argument("--config", default="transcribe")
    ap.add_argument("--variants", default=None,
                    help="comma-separated ids to run (default: a,b,c,d)")
    args = ap.parse_args(argv)

    samples = load_manifest(args.manifest)
    if not samples:
        print("manifest is empty", file=sys.stderr)
        return 1
    wanted = {v.strip() for v in args.variants.split(",")} if args.variants else None
    variants = [v for v in DEFAULT_VARIANTS if not wanted or v.id in wanted]

    edit_paths = EditPaths.for_videos_dir(
        args.videos_dir or Path(args.manifest).parent)
    payload = run_benchmark(samples, variants, cfgmod.load(args.config), edit_paths)
    print(payload["markdown"])
    print(f"wrote {edit_paths.edit / 'benchmark.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
