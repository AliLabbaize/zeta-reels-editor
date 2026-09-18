"""zeta - the command line for the Zeta auto editor.

    zeta ingest      raw/*.mp4 -v <videos_dir>
    zeta transcribe  -v <videos_dir> [--backend gemini_flash_lite] [--force]
    zeta learn       --pairs pairs.csv --out style_profile.json
    zeta plan        -v <videos_dir> --profile style_profile.json [--auto]
    zeta research    -v <videos_dir> [--links links.txt]
    zeta render      -v <videos_dir> [--preview]
    zeta edit        raw01.mp4 --profile style_profile.json [--links links.txt] [--auto]
    zeta review      -v <videos_dir> [--export resolve]
    zeta benchmark   --manifest bench.json -v <videos_dir>

`edit` is every stage in order; the others are the same stages individually so a
run can be resumed after a fix without redoing the expensive parts. Nothing is
re-transcribed unless the source file itself changed (hard rule 9).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import config as configs
from helpers import (
    align_whisperx, build_overlay, captions, derive_cuts, edl as edl_mod,
    ingest as ingest_mod, pack_transcripts, plan_edit, qa_words, report as report_mod,
    research as research_mod, screenshot, self_eval, transcribe_gemini, verify_screenshot,
)
from helpers.gemini_client import LLM, LLMUnavailable
from helpers.paths import EditPaths, REPO_ROOT
from helpers.words import WordsDoc

RENDER_PY = REPO_ROOT / "helpers" / "render.py"


# -- small shared plumbing ----------------------------------------------------


def _paths(videos_dir: str | Path) -> EditPaths:
    return EditPaths.for_videos_dir(videos_dir).ensure()


def _llm(cfg: dict | None = None, paths: EditPaths | None = None) -> LLM:
    """One LLM for the whole run, caching into the session's edit dir."""
    backend = (cfg or {}).get("backends", {}).get(
        (cfg or {}).get("default_backend", "gemini_flash_lite"), {})
    return LLM(model=backend.get("model", "gemini-3.5-flash-lite"),
               fallback_model=backend.get("fallback_model"),
               cache_dir=(paths.edit / "llm_cache") if paths else None)


def _load_profile(path: str | Path | None) -> dict:
    """The style profile, or the cold-start defaults from configs/."""
    if path and Path(path).exists():
        return json.loads(Path(path).read_text(encoding="utf-8"))
    fillers = configs.load("fillers_darija")
    layout = configs.load("layout")
    return {
        "cut_ratio": 0.25,
        "filler_policy": {
            "remove": [w for lang in fillers.get("remove", {}).values() for w in lang],
            "keep": [w for lang in fillers.get("keep", {}).values() for w in lang],
        },
        "retake_policy": "keep_last_complete",
        "inserts": {
            "per_minute": 2.0,
            "median_duration_s": configs.get(layout, "inserts.timing.default_duration_s", 4.5),
            "lead_in_s": configs.get(layout, "inserts.timing.lead_in_s", 0.3),
            "layout": configs.get(layout, "inserts.default_layout", "fit_card"),
            "triggers": ["first_mention_company", "number", "quote", "headline"],
        },
        "_source": "defaults (no style_profile.json yet -- run `zeta learn`)",
    }


def _sources(paths: EditPaths) -> list:
    sources = ingest_mod.load_sources(paths)
    if not sources:
        raise SystemExit("no sources: run `zeta ingest <files> -v <videos_dir>` first")
    return sources


def _words_for(paths: EditPaths, name: str) -> WordsDoc:
    p = paths.words_json(name)
    if not p.exists():
        raise SystemExit(f"no transcript for {name}: run `zeta transcribe` first")
    return WordsDoc.load(p)


def _say(msg: str) -> None:
    print(f"zeta: {msg}", flush=True)


# -- stages -------------------------------------------------------------------


def stage_ingest(paths: EditPaths, files: Sequence[str], force: bool = False) -> list:
    sources = ingest_mod.ingest(files, paths, force=force)
    for s in sources:
        p = s.probe
        _say(f"{s.name}: {p.width}x{p.height} @{p.fps} {p.duration_s:.1f}s "
             f"sha={s.sha256[:12]}")
    return sources


def stage_transcribe(paths: EditPaths, *, backend: str | None = None,
                     force: bool = False, align: bool = True) -> list[WordsDoc]:
    cfg = configs.load("transcribe")
    llm = _llm(cfg, paths)
    docs = []
    for source in _sources(paths):
        _say(f"transcribing {source.name} ({backend or cfg['default_backend']})")
        doc = transcribe_gemini.transcribe_source(
            source, cfg, paths, llm, backend=backend, force=force,
            second_opinion=configs.get(cfg, "second_opinion.enabled", False))

        wav = Path(source.wav_path) if getattr(source, "wav_path", None) else \
            paths.audio / f"{source.name}.wav"
        if align and not cfg.get("use_backend_timings", False):
            _say(f"aligning {source.name} (timing authority stays acoustic)")
            doc = align_whisperx.align(doc, wav, cfg)

        report = qa_words.qa(doc, wav, cfg, edit_paths=paths, name=source.name)
        _say(qa_words.format_report(report).strip().splitlines()[0])
        if not report.get("ok"):
            raise SystemExit(
                f"QA failed for {source.name}: {report.get('failures')}\n"
                f"see {paths.transcripts / (source.name + '.qa.json')}")
        doc.save(paths.words_json(source.name))
        docs.append(doc)

    packed = pack_transcripts.pack(paths)
    _say(f"packed -> {packed['path'] if isinstance(packed, dict) else paths.packed}")
    return docs


def stage_plan(paths: EditPaths, *, profile_path: str | None, source_name: str | None,
               auto: bool) -> tuple[plan_edit.EditPlan, WordsDoc, str]:
    sources = _sources(paths)
    name = source_name or sources[0].name
    doc = _words_for(paths, name)
    profile = _load_profile(profile_path)
    cfg = configs.load("transcribe")
    packed = plan_edit.packed_view(doc)

    few_shot = ""
    if profile_path:
        fs = Path(profile_path).parent / str(profile.get("examples", "few_shot_examples.md"))
        if fs.exists():
            few_shot = fs.read_text(encoding="utf-8")

    edit_plan = plan_edit.plan(doc, packed, profile, source_name=name,
                               llm=_llm(cfg, paths), few_shot=few_shot)
    if not plan_edit.confirm(edit_plan, auto=auto):
        raise SystemExit("aborted at the strategy confirmation (hard rule 10)")
    paths.plan.write_text(json.dumps(edit_plan.to_dict(), ensure_ascii=False, indent=1),
                          encoding="utf-8")
    _say(f"plan -> {paths.plan}  (cut ratio {edit_plan.cut_ratio:.2f})")
    return edit_plan, doc, name


def stage_cut(paths: EditPaths, edit_plan: plan_edit.EditPlan, doc: WordsDoc,
              name: str, *, profile: dict, aspect: str | None) -> tuple[edl_mod.EDL, object]:
    cut_plan = derive_cuts.derive(doc, edit_plan.kept_text, profile, name)
    source = next(s for s in _sources(paths) if s.name == name)
    e = edl_mod.build({name: str(source.path)}, cut_plan.ranges,
                      style_profile="style_profile.json", aspect=aspect,
                      meta={"cut_ratio": round(cut_plan.cut_ratio, 4),
                            "target_cut_ratio": cut_plan.target_cut_ratio,
                            "strategy": edit_plan.strategy})
    _say(f"{len(cut_plan.ranges)} ranges, {e.total_duration_s:.1f}s "
         f"(cut ratio {cut_plan.cut_ratio:.2f})")
    for check in cut_plan.needs_visual_check:
        _say(f"  boundary needs a look: {check}")
    return e, cut_plan


def stage_research(paths: EditPaths, e: edl_mod.EDL, edit_plan: plan_edit.EditPlan,
                   doc: WordsDoc, name: str, *, links: str | None, aspect: str | None,
                   profile: dict) -> list[edl_mod.Overlay]:
    """Resolve sources, capture, verify, and build one overlay per surviving slot."""
    layout_cfg = configs.load("layout")
    cfg = configs.load("transcribe")
    llm = _llm(cfg, paths)
    link_urls = []
    if links and Path(links).exists():
        link_urls = [ln.strip() for ln in Path(links).read_text().splitlines() if ln.strip()]

    outcome = research_mod.research(edit_plan.to_dict(), paths.videos_dir,
                                    links=link_urls, aspect=aspect, llm=llm)
    _say(f"{len(outcome['shippable'])} slot(s) resolved, "
         f"{len(outcome['dropped'])} dropped at resolution")

    by_anchor = {i.after_text: i for i in edit_plan.inserts}
    lead_in = float(profile.get("inserts", {}).get(
        "lead_in_s", configs.get(layout_cfg, "inserts.timing.lead_in_s", 0.3)))
    duration = float(profile.get("inserts", {}).get(
        "median_duration_s", configs.get(layout_cfg, "inserts.timing.default_duration_s", 4.5)))
    layout_name = profile.get("inserts", {}).get(
        "layout", configs.get(layout_cfg, "inserts.default_layout", "fit_card"))

    overlays: list[edl_mod.Overlay] = []
    for slot_meta in outcome["slots"]:
        slot_id = slot_meta["slot_id"]
        slot_dir = paths.slot(slot_id)
        if slot_meta.get("dropped_reason"):
            continue

        insert = by_anchor.get((slot_meta.get("claim_detail") or {}).get("after_text"))
        if insert is None or insert.trigger_word_index < 0:
            _say(f"  {slot_id}: no anchor word, dropped")
            continue
        trigger_word = doc.words[insert.trigger_word_index]
        t_out = e.to_output_time(name, trigger_word.start) if trigger_word.timed else None
        if t_out is None:
            _say(f"  {slot_id}: its trigger word was cut, dropped")
            continue

        try:
            screenshot.capture_for_slot(slot_dir, slot_meta, aspect=aspect)
            verdict = verify_screenshot.verify_slot(slot_dir, llm=llm, aspect=aspect)
        except Exception as exc:                      # a capture failure drops one
            _say(f"  {slot_id}: capture/verify failed ({exc}), dropped")
            continue
        if not getattr(verdict, "visible", False):
            _say(f"  {slot_id}: not verified ({getattr(verdict, 'evidence', '')}), dropped")
            continue

        still = slot_dir / "facecam.png"
        build_overlay.extract_facecam_still(
            e.sources[name], e.to_source_time(t_out)[1], still)
        build_overlay.build_overlay(slot_dir / "shot.png", slot_dir / "overlay.mp4",
                                    facecam_still=still, aspect=aspect,
                                    layout=layout_name, duration_s=duration)
        meta = json.loads((slot_dir / "meta.json").read_text(encoding="utf-8"))
        overlays.append(build_overlay.overlay_for_slot(
            slot_id, meta, trigger_time_output=t_out,
            total_duration_s=e.total_duration_s, duration_s=duration,
            layout=layout_name, aspect=aspect))
        _say(f"  {slot_id}: verified, at {t_out - lead_in:.2f}s")
    return overlays


def stage_captions(paths: EditPaths, e: edl_mod.EDL, doc: WordsDoc, *,
                   aspect: str | None) -> dict:
    out = captions.write_captions(doc, e, paths, aspect=aspect)
    e.subtitles = out["subtitles_field"]
    bidi = out.get("bidi") or []
    _say(f"captions -> {out['ass']}  ({len(bidi)} mixed RTL/LTR line(s) to eyeball)")
    return out


def stage_render(paths: EditPaths, *, preview: bool = False) -> Path:
    out = paths.preview if preview else paths.final
    cmd = [sys.executable, str(RENDER_PY), str(paths.edl), "-o", str(out)]
    if preview:
        cmd.append("--preview")
    _say(f"rendering -> {out.name}")
    subprocess.run(cmd, check=True)
    return out


def stage_self_eval(paths: EditPaths, e: edl_mod.EDL, doc: WordsDoc, video: Path,
                    *, aspect: str | None) -> dict:
    cues = captions.build_cues(doc, e, aspect=aspect)
    findings = self_eval.evaluate(e, video, cues=cues, edit_paths=paths,
                                  llm=_llm(configs.load("transcribe"), paths),
                                  transcript=doc)
    errors = [f for f in getattr(findings, "findings", findings)
              if getattr(f, "severity", "info") == "error"]
    _say(f"self-eval: {len(errors)} error(s)")
    return findings


# -- commands -----------------------------------------------------------------


def cmd_ingest(args) -> int:
    stage_ingest(_paths(args.videos_dir or Path(args.files[0]).parent), args.files,
                 force=args.force)
    return 0


def cmd_transcribe(args) -> int:
    stage_transcribe(_paths(args.videos_dir), backend=args.backend,
                     force=args.force, align=not args.no_align)
    return 0


def cmd_learn(args) -> int:
    from helpers import learn_style       # lazy: pulls scenedetect and cv2
    profile = learn_style.learn(args.pairs, out=args.out,
                                llm=_llm(configs.load("transcribe")))
    _say(f"style profile -> {args.out}")
    print(json.dumps(profile if isinstance(profile, dict) else {}, ensure_ascii=False,
                     indent=1)[:1200])
    return 0


def cmd_plan(args) -> int:
    paths = _paths(args.videos_dir)
    edit_plan, doc, name = stage_plan(paths, profile_path=args.profile,
                                      source_name=args.source, auto=args.auto)
    e, _ = stage_cut(paths, edit_plan, doc, name,
                     profile=_load_profile(args.profile), aspect=args.aspect)
    e.save(paths.edl)
    _say(f"edl -> {paths.edl}")
    return 0


def cmd_research(args) -> int:
    paths = _paths(args.videos_dir)
    e = edl_mod.EDL.load(paths.edl)
    edit_plan = plan_edit.EditPlan.from_dict(json.loads(paths.plan.read_text(encoding="utf-8"))) \
        if hasattr(plan_edit.EditPlan, "from_dict") else None
    if edit_plan is None:
        raise SystemExit("plan_edit.EditPlan has no from_dict; run `zeta edit` instead")
    name = e.ranges[0].source
    doc = _words_for(paths, name)
    e.overlays = stage_research(paths, e, edit_plan, doc, name, links=args.links,
                                aspect=args.aspect or e.aspect,
                                profile=_load_profile(args.profile))
    e.save(paths.edl)
    return 0


def cmd_render(args) -> int:
    paths = _paths(args.videos_dir)
    stage_render(paths, preview=args.preview)
    return 0


def cmd_edit(args) -> int:
    """Every stage, in order. This is the one people actually run."""
    videos_dir = args.videos_dir or Path(args.files[0]).resolve().parent
    paths = _paths(videos_dir)
    aspect = args.aspect or configs.load("layout").get("default_aspect")
    profile = _load_profile(args.profile)
    if "_source" in profile:
        _say("no style profile: using cold-start defaults. Run `zeta learn` for "
             "cuts that look like Ali's.")

    stage_ingest(paths, args.files, force=False)
    stage_transcribe(paths, backend=args.backend, force=False)
    edit_plan, doc, name = stage_plan(paths, profile_path=args.profile,
                                      source_name=None, auto=args.auto)
    e, cut_plan = stage_cut(paths, edit_plan, doc, name, profile=profile, aspect=aspect)

    if not args.no_screenshots:
        try:
            e.overlays = stage_research(paths, e, edit_plan, doc, name,
                                        links=args.links, aspect=aspect, profile=profile)
        except LLMUnavailable as exc:
            _say(f"screenshots skipped: {exc}")

    stage_captions(paths, e, doc, aspect=aspect)
    e.save(paths.edl)

    video = stage_render(paths, preview=args.preview)
    findings = stage_self_eval(paths, e, doc, video, aspect=aspect)

    report_mod.write_report(paths, e, style_profile=profile, self_eval=findings,
                            raw_duration_s=doc.duration())
    report_mod.append_project_md(paths, report_mod.Session(
        strategy=edit_plan.strategy,
        decisions=[f"{len(e.ranges)} ranges kept, cut ratio {cut_plan.cut_ratio:.2f}",
                   f"{len(e.overlays)} verified insert(s)"],
        reasoning=[c.reason for c in cut_plan.cuts[:10] if getattr(c, "reason", None)],
        outstanding=[str(c) for c in cut_plan.needs_visual_check],
    ))
    _say(f"done: {video}")
    _say(f"report: {paths.report}")
    return 0


def cmd_review(args) -> int:
    paths = _paths(args.videos_dir)
    if args.export:
        # auto-editor can turn the EDL into an NLE timeline for manual finishing.
        if subprocess.run(["which", "auto-editor"], capture_output=True).returncode != 0:
            raise SystemExit("auto-editor is not installed: uv pip install auto-editor")
        subprocess.run(["auto-editor", str(paths.final), "--export", args.export], check=True)
        return 0
    print(paths.report)
    return 0


def cmd_benchmark(args) -> int:
    from helpers import benchmark
    paths = _paths(args.videos_dir)
    samples = benchmark.load_manifest(args.manifest)
    results = benchmark.run_benchmark(samples, benchmark.default_variants()
                                      if hasattr(benchmark, "default_variants") else [],
                                      configs.load("transcribe"), paths)
    print(benchmark.render_table(results.get("results", results)))
    return 0


# -- argument parsing ---------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="zeta", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    def add_common(p, *, videos_required: bool = True) -> None:
        p.add_argument("-v", "--videos-dir", required=videos_required,
                       help="session directory; everything lands in <videos_dir>/edit/")

    p = sub.add_parser("ingest", help="probe sources and extract 16 kHz mono audio")
    p.add_argument("files", nargs="+")
    add_common(p, videos_required=False)
    p.add_argument("--force", action="store_true", help="re-extract audio even if cached")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("transcribe", help="transcribe, align, QA, pack")
    add_common(p)
    p.add_argument("--backend", choices=sorted(transcribe_gemini.BACKENDS))
    p.add_argument("--force", action="store_true", help="ignore the transcript cache")
    p.add_argument("--no-align", action="store_true",
                   help="skip forced alignment (debugging only: timings stay approximate)")
    p.set_defaults(func=cmd_transcribe)

    p = sub.add_parser("learn", help="learn the editing style from raw/published pairs")
    p.add_argument("--pairs", required=True, help="CSV of raw,published paths")
    p.add_argument("--out", default="style_profile.json")
    p.set_defaults(func=cmd_learn)

    p = sub.add_parser("plan", help="plan the edit and derive the cuts")
    add_common(p)
    p.add_argument("--profile", help="style_profile.json")
    p.add_argument("--source", help="source name (default: the first one)")
    p.add_argument("--aspect", help="9:16 | 4:5 | 1:1 | 16:9")
    p.add_argument("--auto", action="store_true", help="skip the strategy confirmation")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("research", help="resolve sources, capture, verify, build overlays")
    add_common(p)
    p.add_argument("--links", help="links.txt of URLs you already have")
    p.add_argument("--profile")
    p.add_argument("--aspect")
    p.set_defaults(func=cmd_research)

    p = sub.add_parser("render", help="render the EDL")
    add_common(p)
    p.add_argument("--preview", action="store_true")
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("edit", help="every stage, in order")
    p.add_argument("files", nargs="+")
    add_common(p, videos_required=False)
    p.add_argument("--profile", help="style_profile.json")
    p.add_argument("--links")
    p.add_argument("--aspect")
    p.add_argument("--backend", choices=sorted(transcribe_gemini.BACKENDS))
    p.add_argument("--auto", action="store_true",
                   help="unattended: the decision report is the review")
    p.add_argument("--preview", action="store_true", help="render the preview, not the final")
    p.add_argument("--no-screenshots", action="store_true")
    p.set_defaults(func=cmd_edit)

    p = sub.add_parser("review", help="print the decision report path, or export an NLE timeline")
    add_common(p)
    p.add_argument("--export", choices=["resolve", "premiere", "final-cut-pro", "shotcut"])
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("benchmark", help="score transcription variants (spec stage 1)")
    add_common(p)
    p.add_argument("--manifest", required=True, help="JSON of wav/reference pairs")
    p.set_defaults(func=cmd_benchmark)

    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except LLMUnavailable as exc:
        print(f"zeta: {exc}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as exc:
        print(f"zeta: a subprocess failed ({exc.cmd[0]}): {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
