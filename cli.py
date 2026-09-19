"""zeta - the command line for the Zeta auto editor.

    zeta ingest      raw/*.mp4 -v <videos_dir>
    zeta transcribe  -v <videos_dir> [--backend gemini_flash_lite] [--force]
    zeta fetch       <ig|tiktok|youtube url> -v <dir> --name ep14 --pairs pairs.csv
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
import urllib.parse
import os
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import config as configs
from helpers import (
    align_whisperx, build_overlay, captions, derive_cuts, edl as edl_mod,
    fetch as fetch_mod, ingest as ingest_mod, pack_transcripts, plan_edit, qa_words,
    report as report_mod, research as research_mod, screenshot, self_eval,
    transcribe_gemini, verify_screenshot,
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
        # No target until `zeta learn` measures one. A guessed 0.25 failed a tight
        # 28 s take cut at 0.11 and pushed the retry into cutting real content.
        "cut_ratio": None,
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
        # Hard rule 9 covers the transcript; alignment deserves the same
        # treatment, because forced alignment of a ten minute take on CPU costs
        # more than the transcription did. If the words.json on disk was built
        # from this exact file, it is still true.
        cached = paths.words_json(source.name)
        qa_file = paths.transcripts / (source.name + ".qa.json")
        if cached.exists() and qa_file.exists() and not force:
            doc = WordsDoc.load(cached)
            # transcribe_source writes words.json before alignment as its own
            # resume point, so only a take whose QA passed counts as finished.
            qa_ok = json.loads(qa_file.read_text(encoding="utf-8"))
            if (doc.source.get("sha256") == source.sha256 and qa_ok.get("ok")
                    and (qa_ok.get("source") or {}).get("sha256") == source.sha256):
                _say(f"{source.name}: transcript is current, reusing it")
                docs.append(doc)
                continue

        _say(f"transcribing {source.name} ({backend or cfg['default_backend']})")
        doc = transcribe_gemini.transcribe_source(
            source, cfg, paths, llm, backend=backend, force=force,
            second_opinion=configs.get(cfg, "second_opinion.enabled", False))

        wav = Path(source.wav)
        if align and not cfg.get("use_backend_timings", False):
            _say(f"aligning {source.name} (timing authority stays acoustic)")
            doc = align_whisperx.align(doc, wav, cfg)
            # Retakes the transcriber collapsed come back as their own words,
            # so the editor can keep the last take (see fill_loud_gaps).
            spans = (transcribe_gemini.fill_loud_gaps(doc, wav, llm)
                     if configs.get(cfg, "qa.fill_loud_gaps", False) else [])
            for lo, hi in spans:
                align_whisperx.realign_window(doc, wav, lo, hi, cfg)
            if spans:
                _say(f"{source.name}: {len(spans)} untranscribed stretch(es) filled "
                     f"({sum(h - l for l, h in spans)} words)")

        report = qa_words.qa(doc, wav, cfg, edit_paths=paths, name=source.name)
        _say(qa_words.format_report(report).strip().splitlines()[0])
        if not report.get("ok"):
            raise SystemExit(
                f"QA failed for {source.name}: {report.get('failures')}\n"
                f"see {paths.transcripts / (source.name + '.qa.json')}")
        doc.save(paths.words_json(source.name))
        docs.append(doc)

    pack_transcripts.pack(paths)
    _say(f"packed -> {paths.packed}")
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


# Things a human must look at before publishing, collected across stages.
review_flags: list[str] = []


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

    story = getattr(edit_plan, "story", "") or ""
    overlays: list[edl_mod.Overlay] = []
    missing: list[tuple[str, Any, float]] = []   # (slot_id, insert, t_out) with no card yet
    from concurrent.futures import ThreadPoolExecutor

    def emit(slot_id: str, insert, t_out: float) -> None:
        slot_dir = paths.slot(slot_id)
        still = slot_dir / "facecam.png"
        build_overlay.extract_facecam_still(
            e.sources[name], e.to_source_time(t_out)[1], still)
        # The image that PASSED verification: after a retry that is
        # shot_retry1.png, and shot.png may be the bot wall that failed.
        meta = json.loads((slot_dir / "meta.json").read_text(encoding="utf-8"))
        build_overlay.build_overlay(meta["image"], slot_dir / "overlay.mp4",
                                    facecam_still=still, aspect=aspect,
                                    layout=layout_name, duration_s=duration)
        ov = build_overlay.overlay_for_slot(
            slot_id, meta, trigger_time_output=t_out,
            total_duration_s=e.total_duration_s, duration_s=duration,
            layout=layout_name, aspect=aspect)
        # What Ali says at that moment, for the review page.
        i = insert.trigger_word_index
        ov.meta["said"] = " ".join(w.display or w.word for w in doc.words[max(0, i - 4):i + 10])
        ov.meta["image"] = meta.get("image")
        overlays.append(ov)
        _say(f"  {slot_id}: verified, at {t_out - lead_in:.2f}s  {meta.get('url')}")

    def walk(slot_meta: dict) -> tuple[str, Any, float, bool] | None:
        """Capture and verify one slot's candidates. Safe to run in a thread:
        it touches only this slot's directory."""
        slot_id = slot_meta["slot_id"]
        slot_dir = paths.slot(slot_id)
        insert = by_anchor.get((slot_meta.get("claim_detail") or {}).get("after_text"))
        if insert is None or insert.trigger_word_index < 0:
            _say(f"  {slot_id}: no anchor word, dropped")
            return None
        trigger_word = doc.words[insert.trigger_word_index]
        t_out = e.to_output_time(name, trigger_word.start) if trigger_word.timed else None
        if t_out is None:
            _say(f"  {slot_id}: its trigger word was cut, dropped")
            return None
        if slot_meta.get("dropped_reason"):
            return slot_id, insert, t_out, False

        # A bot wall (Cloudflare on openai.com) or a page that does not show the
        # claim sinks one URL, not the slot: research already ranked the other
        # allowed candidates, so walk them before giving up.
        m0 = json.loads((slot_dir / "meta.json").read_text(encoding="utf-8"))
        m0["story"] = story
        (slot_dir / "meta.json").write_text(json.dumps(m0, ensure_ascii=False, indent=1),
                                            encoding="utf-8")
        alternates = [c for c in slot_meta.get("candidates") or []
                      if c.get("url") and not c.get("rejected_reason")][:4]
        verdict = None
        for n, cand in enumerate(alternates or [slot_meta]):
            meta = json.loads((slot_dir / "meta.json").read_text(encoding="utf-8"))
            if n:
                meta.update(url=cand["url"], source_type=cand.get("source_type"))
                for k in ("captures", "image", "verification", "verified"):
                    meta.pop(k, None)
                (slot_dir / "meta.json").write_text(
                    json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
            try:
                screenshot.capture_for_slot(slot_dir, meta, aspect=aspect)
                verdict = verify_screenshot.verify_slot(slot_dir, llm=llm, aspect=aspect)
            except Exception as exc:
                _say(f"  {slot_id}: {meta.get('url')}: capture/verify failed ({exc})")
                verdict = None
                continue
            if getattr(verdict, "visible", False):
                break
            _say(f"  {slot_id}: {meta.get('url')}: not verified "
                 f"({getattr(verdict, 'evidence', '')})")
        return slot_id, insert, t_out, bool(getattr(verdict, "visible", False))

    # Capture and verify every slot at once: each is a browser page and a vision
    # call, and one after another was the slowest part of research.
    with ThreadPoolExecutor(max_workers=4) as pool:
        walked = list(pool.map(walk, outcome["slots"]))
    for got in walked:
        if got is None:
            continue
        slot_id, insert, t_out, ok = got
        if ok:
            emit(slot_id, insert, t_out)
        else:
            missing.append((slot_id, insert, t_out))

    if missing and story:
        _fill_from_pool(paths, missing, story, doc=doc, llm=llm, aspect=aspect, emit=emit)
    return _spaced(overlays, layout_cfg)


def _fill_from_pool(paths: EditPaths, missing: list, story: str, *, doc: WordsDoc,
                    llm, aspect, emit) -> None:
    """Moments whose own search failed get a card from the story's source pool.

    Each moment gets one search, and on a real take 6 of 9 came back empty (bot
    walls, homepages for things that have no page). The story itself has plenty
    of capturable sources: the original report, the victim's disclosure, major
    coverage. They are searched once, captured and verified once, then matched
    to the empty moments. Every card is used at most once.
    """
    sources_cfg = configs.load("sources")
    claim = research_mod.Claim(slot_id="pool", claim=story, entity="")
    prompt = (f"News story: {story}\nFind up to 12 pages that document it, ORIGINAL "
              "sources first: the incident or technical report (PDFs welcome), the "
              "official statements or disclosures of the organisations involved, then "
              "major news coverage. Return only URLs you actually saw.")
    try:
        found = research_mod._claude_search(prompt, sources_cfg.get("search") or {})
    except Exception as exc:
        _say(f"  pool: search failed ({exc})")
        return
    cands = [research_mod.make_candidate(str(r.get("url", "")), claim, sources_cfg,
                                         title=r.get("title"))
             for r in (found or {}).get("candidates", []) if isinstance(r, dict)]
    cands = sorted([c for c in cands if c.usable], key=lambda c: c.rank)

    pool: list[dict] = []
    for n, c in enumerate(cands):
        d = paths.edit / "screenshots" / f"pool_{n:02d}"
        d.mkdir(parents=True, exist_ok=True)
        meta = {"slot_id": d.name, "url": c.url, "source_type": c.source_type,
                "claim": story, "story": story, "title": c.title or ""}
        (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
        try:
            screenshot.capture_for_slot(d, meta, aspect=aspect)
            if verify_screenshot.verify_slot(d, llm=llm, aspect=aspect).visible:
                pool.append(json.loads((d / "meta.json").read_text(encoding="utf-8")))
        except Exception as exc:
            _say(f"  pool: {c.url}: {exc}")
    _say(f"  pool: {len(pool)} verified card(s) for the story")
    if not pool:
        return

    # One text call matches moments to cards by what Ali says there.
    lines = "\n".join(f"{i}. at {t:.0f}s, Ali says: "
                       + " ".join(w.display or w.word for w in doc.words[
                           max(0, ins.trigger_word_index - 4):ins.trigger_word_index + 10])
                       + f" (wanted: {ins.claim})"
                       for i, (_sid, ins, t) in enumerate(missing))
    cards = "\n".join(f"{j}. {p.get('title') or ''} ({urllib.parse.urlparse(p['url']).netloc})"
                       for j, p in enumerate(pool))
    try:
        out = llm.generate(
            f"Moments in a news video that need an on-screen source:\n{lines}\n\n"
            f"Verified source cards:\n{cards}\n\nAssign each moment the card that best "
            "fits what is said there, using every card at most once; -1 if none fits. "
            "Original reports and official disclosures go where Ali describes what "
            "happened.", schema={"type": "object", "required": ["assign"], "properties": {
                "assign": {"type": "array", "items": {"type": "integer"}}}})
        assign = [int(x) for x in (out or {}).get("assign", [])]
    except Exception:
        assign = list(range(len(missing)))
    used: set[int] = set()
    for (slot_id, insert, t_out), j in zip(missing, assign):
        if j < 0 or j >= len(pool) or j in used:
            continue
        used.add(j)
        card = pool[j]
        meta = dict(card, slot_id=slot_id, verified=True, pool=True)
        (paths.slot(slot_id) / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        emit(slot_id, insert, t_out)


def _spaced(overlays: list[edl_mod.Overlay], layout_cfg: dict) -> list[edl_mod.Overlay]:
    """Drop inserts that would overlap an earlier-chosen one.

    The planner put two inserts 7 words apart and the EDL refused the overlap.
    A real source beats a Bing News fallback; otherwise the earlier one wins.
    """
    gap = float(configs.get(layout_cfg, "inserts.density.min_gap_between_inserts_s", 1.5))
    is_bing = lambda o: (o.meta or {}).get("url", "").startswith("https://www.bing.com/")
    kept: list[edl_mod.Overlay] = []
    for o in sorted(overlays, key=lambda o: (is_bing(o), o.start_in_output)):
        if all(o.start_in_output >= k.start_in_output + k.duration + gap
               or k.start_in_output >= o.start_in_output + o.duration + gap for k in kept):
            kept.append(o)
        else:
            _say(f"  {(o.meta or {}).get('slot_id')}: overlaps another insert, dropped")
            review_flags[:] = [f for f in review_flags
                               if not f.startswith(f"{(o.meta or {}).get('slot_id')}:")]
    return sorted(kept, key=lambda o: o.start_in_output)


def stage_captions(paths: EditPaths, e: edl_mod.EDL, doc: WordsDoc, *,
                   aspect: str | None) -> dict:
    out = captions.write_captions(doc, e, paths, aspect=aspect,
                                  llm=_llm(configs.load("transcribe"), paths))
    if out.get("english_error"):
        review_flags.append(f"English subtitles missing: {out['english_error']}")
    e.subtitles = out["subtitles_field"]
    bidi = out.get("bidi") or []
    _say(f"captions -> {out['ass']}  ({len(bidi)} mixed RTL/LTR line(s) to eyeball)")
    return out


def stage_render(paths: EditPaths, *, preview: bool = False) -> Path:
    out = paths.preview if preview else paths.final
    # The delivery rate from layout.yaml, never the source's: an iPhone's VFR
    # average (3217800/53633) became a 1/911709 timebase that QuickTime shows
    # as a black video. Overlays are built at this rate too.
    aspect = json.loads(paths.edl.read_text(encoding="utf-8")).get("aspect")
    fps = configs.get(configs.aspect_config(aspect)[1], "fps", 30)
    _say(f"rendering -> {out.name}")
    # fast_render calls render.py's own steps and reuses the cut segments when
    # only cards or captions changed (helpers/fast_render.py).
    from helpers import fast_render
    fast_render.render(paths.edl, out, fps=str(fps), preview=preview)
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


def cmd_fetch(args) -> int:
    return fetch_mod.main(
        list(args.urls) + ["-v", str(args.videos_dir)]
        + (["--name", args.name] if args.name else [])
        + (["--pairs", args.pairs] if args.pairs else [])
        + (["--raw", args.raw] if args.raw else [])
        + (["--cookies-from-browser", args.cookies_from_browser]
           if args.cookies_from_browser else []))


def cmd_learn(args) -> int:
    from helpers import learn_style       # lazy: pulls scenedetect and cv2
    # Every video goes through the same Stage 1 as an edit (ingest, transcribe,
    # align, QA) into <pairs dir>/edit/, where learn_style.words_for looks.
    # Cached per source hash, so re-learning costs nothing (hard rule 9).
    pairs_dir = Path(args.pairs).resolve().parent
    media = [str(m) for p in learn_style.load_pairs(args.pairs)
             for m in (p.raw, p.published) if m]
    if media:
        main(["ingest", *media, "-v", str(pairs_dir)])
        main(["transcribe", "-v", str(pairs_dir)])
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
        if args.review_shots and e.overlays:
            from helpers import review_shots
            keep = review_shots.review([
                {"slot_id": o.meta["slot_id"], "time": o.start_in_output,
                 "said": o.meta.get("said", ""), "claim": o.meta.get("claim", ""),
                 "url": o.meta.get("url", ""), "image": o.meta.get("image", "")}
                for o in e.overlays])
            (paths.edit / "screenshots" / "review.json").write_text(
                json.dumps(keep, ensure_ascii=False, indent=1), encoding="utf-8")
            notes = {k: v["note"] for k, v in keep.items() if v["note"]}
            swaps = {k: v["url"] for k, v in keep.items() if v["url"] and v["keep"]}
            if notes or swaps:
                # Feedback means "redo these", so nothing renders yet. The review
                # is saved; a swap link goes to links.txt for the next run.
                if swaps:
                    links = paths.videos_dir / "links.txt"
                    with links.open("a", encoding="utf-8") as f:
                        f.writelines(f"{u}  # {k}\n" for k, u in swaps.items())
                for k, v in {**notes, **{k: f"use {u}" for k, u in swaps.items()}}.items():
                    _say(f"review: {k}: {v}")
                raise SystemExit("zeta: screenshot feedback saved to screenshots/review.json; "
                                 "not rendering until it is addressed")
            e.overlays = [o for o in e.overlays if keep[o.meta["slot_id"]]["keep"]]
            _say(f"review: kept {len(e.overlays)} of {len(keep)} screenshot(s)")

    stage_captions(paths, e, doc, aspect=aspect)
    e.save(paths.edl)

    video = stage_render(paths, preview=args.preview)
    findings = ({} if args.no_self_eval
                else stage_self_eval(paths, e, doc, video, aspect=aspect))

    report_mod.write_report(paths, e, style_profile=profile, self_eval=findings,
                            raw_duration_s=doc.duration())
    report_mod.append_project_md(paths, report_mod.Session(
        strategy=edit_plan.strategy,
        decisions=[f"{len(e.ranges)} ranges kept, cut ratio {cut_plan.cut_ratio:.2f}",
                   f"{len(e.overlays)} verified insert(s)"],
        reasoning=[c.reason for c in cut_plan.cuts[:10] if getattr(c, "reason", None)],
        outstanding=review_flags + [str(c) for c in cut_plan.needs_visual_check],
    ))
    for flag in review_flags:
        _say(f"REVIEW: {flag}")
    _say(f"done: {video}")
    _say(f"report: {paths.report}")
    return 0


def cmd_proof(args) -> int:
    """Still frames of every insert: layout review in seconds, not renders."""
    from helpers import proof
    paths = _paths(args.videos_dir)
    out = proof.proof_sheet(paths, Path(args.edl) if args.edl else None)
    _say(f"proof -> {out}")
    if args.open:
        subprocess.run(["open", str(out)], check=False)
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
    variants = [v for v in benchmark.DEFAULT_VARIANTS
                if not args.variants or v.id in set(args.variants.split(","))]
    results = benchmark.run_benchmark(samples, variants, configs.load("transcribe"), paths)
    rows = results.get("results", results) if isinstance(results, dict) else results
    print(benchmark.render_table(rows, benchmark.pick_winner(rows)))
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

    p = sub.add_parser("fetch", help="download a published video by URL, for learn mode")
    p.add_argument("urls", nargs="+")
    add_common(p)
    p.add_argument("--name", help="basename for a single download, e.g. ep14")
    p.add_argument("--pairs", help="pairs.csv to append the row to")
    p.add_argument("--raw", help="the matching raw take, if you have it")
    p.add_argument("--cookies-from-browser", help="chrome | firefox | safari")
    p.set_defaults(func=cmd_fetch)

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
    p.add_argument("--no-self-eval", action="store_true",
                   help="skip the post-render vision check (1-2 min); for iterations")
    p.add_argument("--review-shots", action="store_true",
                   help="open a page to keep or drop each screenshot before rendering")
    p.set_defaults(func=cmd_edit)

    p = sub.add_parser("proof", help="still frames of every insert (fast layout check)")
    p.add_argument("-v", "--videos-dir", required=True)
    p.add_argument("--edl", default=None)
    p.add_argument("--open", action="store_true")
    p.set_defaults(func=cmd_proof)

    p = sub.add_parser("review", help="print the decision report path, or export an NLE timeline")
    add_common(p)
    p.add_argument("--export", choices=["resolve", "premiere", "final-cut-pro", "shotcut"])
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("benchmark", help="score transcription variants (spec stage 1)")
    add_common(p)
    p.add_argument("--manifest", required=True, help="JSON of wav/reference pairs")
    p.add_argument("--variants", help="comma-separated variant ids (default: a,b,c,d)")
    p.set_defaults(func=cmd_benchmark)

    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # The extras' tools (shot-scraper, yt-dlp) live beside this interpreter, which
    # is not on PATH when `.venv/bin/python cli.py` runs without an activated venv.
    os.environ["PATH"] = os.pathsep.join([str(Path(sys.executable).parent),
                                          os.environ.get("PATH", "")])
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
