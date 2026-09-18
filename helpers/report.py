"""`decision_report.html` and the `project.md` session memory.

In `--auto` mode nobody confirms the strategy, so the report IS the review: it
has to answer, without opening the video, why every cut happened and where
every insert came from. That is why each row carries the removed text and the
source URL with its verification evidence, and why the thumbnail is inlined as
a data URI -- a report that depends on files next to it stops being evidence
the moment it is copied into a message.

No CDN, no external stylesheet, no webfont: one file that renders the same on a
laptop with no network. Darija is written right-to-left and the source URLs are
left-to-right, often in the same cell, so every text cell is `dir="auto"` with
`unicode-bidi: plaintext` -- the browser then resolves each string on its own
first strong character instead of dragging a URL to the wrong side of a line.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:  # package import
    from . import config as configs
    from .edl import EDL
    from .paths import EditPaths
except ImportError:  # `python helpers/report.py ...`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from helpers import config as configs
    from helpers.edl import EDL
    from helpers.paths import EditPaths

CUT_RATIO_TOLERANCE = 0.10  # the planner prompt's own budget (spec stage 4)
THUMB_MAX_BYTES = 4_000_000
THUMB_NAMES = ("thumb.png", "shot.png", "capture.png", "screenshot.png", "shot.jpg")
CONFIG_NAMES = ("captions", "layout", "transcribe", "sources", "fillers_darija")

_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
         ".webp": "image/webp", ".gif": "image/gif"}


# -------- rows ----------------------------------------------------------------


@dataclass
class CutRow:
    t_output: float
    klass: str
    text: str
    reason: str
    source: str = ""
    source_span: tuple[float, float] | None = None
    removed_s: float | None = None
    needs_visual_check: bool = False


@dataclass
class InsertRow:
    t_output: float
    duration: float
    claim: str
    url: str
    source_type: str
    verified: bool
    evidence: str
    layout: str = ""
    thumbnail: str | None = None  # data URI


def cut_rows(edl: EDL) -> list[CutRow]:
    """Every cut, at the output time where the viewer meets it.

    A cut is recorded on the range that FOLLOWS it (`cut_before`), so its
    output time is that range's offset: the moment the removed material would
    have played.
    """
    rows: list[CutRow] = []
    for r, off in zip(edl.ranges, edl.offsets()):
        cb = r.cut_before or {}
        if not cb:
            continue
        span = cb.get("source_span") or cb.get("span")
        rows.append(CutRow(
            t_output=off,
            klass=cb.get("class", "other"),
            text=cb.get("text", ""),
            reason=cb.get("reason") or r.reason or "",
            source=r.source,
            source_span=(float(span[0]), float(span[1])) if span else None,
            removed_s=cb.get("removed_s"),
            needs_visual_check=bool(cb.get("needs_visual_check")),
        ))
    return rows


def _thumbnail_uri(overlay_file: str, meta: Mapping[str, Any], base: Path | None,
                   max_bytes: int = THUMB_MAX_BYTES) -> str | None:
    candidates: list[Path] = []
    explicit = meta.get("thumbnail") or meta.get("image")
    # An overlay path in an EDL is relative to the edit dir for render.py and
    # to the videos dir in the spec's example; look under both.
    roots = [p for p in (base, base / "edit" if base else None, Path.cwd()) if p]
    for root in roots:
        if explicit:
            candidates.append(root / explicit if not Path(explicit).is_absolute() else Path(explicit))
        slot_dir = (root / overlay_file).parent if overlay_file else None
        if slot_dir:
            candidates += [slot_dir / n for n in THUMB_NAMES]
            if slot_dir.is_dir():
                candidates += sorted(p for p in slot_dir.iterdir()
                                     if p.suffix.lower() in _MIME)
    for p in candidates:
        try:
            if not p.is_file() or p.stat().st_size > max_bytes:
                continue
            data = base64.b64encode(p.read_bytes()).decode("ascii")
        except OSError:
            continue
        return f"data:{_MIME.get(p.suffix.lower(), 'image/png')};base64,{data}"
    return None


def insert_rows(edl: EDL, base: Path | None = None) -> list[InsertRow]:
    rows: list[InsertRow] = []
    for o in edl.overlays:
        m = o.meta or {}
        rows.append(InsertRow(
            t_output=o.start_in_output,
            duration=o.duration,
            claim=m.get("claim", ""),
            url=m.get("url", ""),
            source_type=m.get("source_type", ""),
            verified=bool(m.get("verified")),
            evidence=m.get("evidence", ""),
            layout=m.get("layout", ""),
            thumbnail=_thumbnail_uri(o.file, m, base),
        ))
    return rows


# -------- style profile delta -------------------------------------------------


def style_delta(edl: EDL, style_profile: Mapping[str, Any] | None, *,
                raw_duration_s: float | None = None) -> list[dict]:
    """Actual vs target, so drift from the learned style is visible as a number."""
    profile = style_profile or {}
    total = edl.total_duration_s
    rows: list[dict] = []

    # derive_cuts records the achieved ratio and the source duration; an EDL
    # built by hand may carry neither, so both routes are tried before giving up.
    meta = dict(edl.meta or {})
    meta.update(dict(meta.get("cut_plan") or {}))
    raw = raw_duration_s or meta.get("raw_duration_s") or meta.get("source_duration_s")
    actual_ratio = (1.0 - total / raw) if raw else meta.get("cut_ratio")
    target_ratio = profile.get("cut_ratio")
    rows.append(_delta_row("cut ratio", target_ratio, actual_ratio, CUT_RATIO_TOLERANCE))

    per_min = (len(edl.overlays) / (total / 60.0)) if total > 0 else None
    target_pm = (profile.get("inserts") or {}).get("per_minute")
    rows.append(_delta_row("inserts per minute", target_pm, per_min, 1.0))
    return rows


def _delta_row(metric: str, target: Any, actual: Any, tolerance: float) -> dict:
    row: dict[str, Any] = {"metric": metric, "target": target, "actual": actual,
                           "tolerance": tolerance, "delta": None, "within": None}
    if isinstance(target, (int, float)) and isinstance(actual, (int, float)):
        row["delta"] = actual - target
        row["within"] = abs(row["delta"]) <= tolerance
    return row


# -------- config fingerprint --------------------------------------------------


def config_fingerprint(names: Sequence[str] = CONFIG_NAMES,
                       loaded: Mapping[str, Mapping] | None = None) -> dict:
    """Which configs produced this cut.

    Two runs of the same footage that differ only in `captions.yaml` produce
    different files; without this the report cannot say which one it describes.
    """
    per: dict[str, str] = {}
    for name in names:
        try:
            cfg = dict(loaded[name]) if loaded and name in loaded else configs.load(name)
        except (FileNotFoundError, KeyError):
            continue
        blob = json.dumps(cfg, sort_keys=True, ensure_ascii=False, default=str)
        per[name] = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]
    combined = hashlib.sha256("".join(f"{k}:{v}" for k, v in sorted(per.items()))
                              .encode("utf-8")).hexdigest()[:12]
    return {"hash": combined, "configs": per}


# -------- html ----------------------------------------------------------------


def _e(v: Any) -> str:
    return html.escape("" if v is None else str(v), quote=True)


def _t(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    s = max(0.0, float(seconds))
    return f"{int(s // 60):d}:{s % 60:05.2f}"


def _num(v: Any, nd: int = 2) -> str:
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "-"


_CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { margin: 0; padding: 28px; background: #12121a; color: #e8e8ef;
       font: 15px/1.55 -apple-system, "Segoe UI", system-ui, sans-serif; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 32px 0 10px; text-transform: uppercase;
     letter-spacing: .08em; color: #9aa0b5; }
.sub { color: #8b90a3; font-size: 13px; margin-bottom: 8px; }
table { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 14px; }
th { text-align: left; font-weight: 600; color: #9aa0b5; font-size: 12px;
     text-transform: uppercase; letter-spacing: .05em;
     border-bottom: 1px solid #2c2c3a; padding: 6px 10px; }
td { border-bottom: 1px solid #22222e; padding: 8px 10px; vertical-align: top; }
tr:hover td { background: #191924; }
.t { font-variant-numeric: tabular-nums; white-space: nowrap; color: #b9bed2; }
/* Darija is RTL, URLs are LTR, and they share cells: let the browser resolve
   each string on its own first strong character. */
.txt { unicode-bidi: plaintext; text-align: start; max-width: 42ch; }
.tag { display: inline-block; padding: 1px 8px; border-radius: 999px;
       font-size: 12px; background: #262636; color: #c9cee2; white-space: nowrap; }
.ok { background: #15321f; color: #7ee2a8; }
.bad { background: #3a1720; color: #ff9aa9; }
.warn { background: #3a2f14; color: #f2c66a; }
.muted { color: #7e8397; }
a { color: #8ab4ff; word-break: break-all; }
img.thumb { width: 190px; border-radius: 6px; border: 1px solid #2c2c3a; display: block; }
code { background: #1c1c28; padding: 1px 5px; border-radius: 4px; font-size: 13px; }
.empty { color: #7e8397; font-style: italic; padding: 10px 0; }
footer { margin-top: 36px; border-top: 1px solid #22222e; padding-top: 12px;
         color: #7e8397; font-size: 12px; }
"""


def _sev_class(sev: str) -> str:
    return {"error": "bad", "warning": "warn"}.get(sev, "tag")


def _findings(self_eval: Any) -> tuple[list[dict], dict]:
    """Normalise a `SelfEvalResult`, its dict, or a bare list of findings."""
    if self_eval is None:
        return [], {}
    if hasattr(self_eval, "to_dict"):
        self_eval = self_eval.to_dict()
    if isinstance(self_eval, Mapping):
        return list(self_eval.get("findings") or []), dict(self_eval)
    out = []
    for f in self_eval:
        out.append(f.to_dict() if hasattr(f, "to_dict") else dict(f))
    return out, {}


def render_html(*, edl: EDL, cuts: Sequence[CutRow], inserts: Sequence[InsertRow],
                deltas: Sequence[Mapping], self_eval: Any = None,
                fingerprint: Mapping | None = None, title: str = "Zeta decision report",
                generated_at: str | None = None) -> str:
    findings, meta = _findings(self_eval)
    when = generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    total = edl.total_duration_s

    parts: list[str] = [
        "<!DOCTYPE html>",
        '<html lang="en" dir="ltr"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{_e(title)}</title><style>{_CSS}</style></head><body>",
        f"<h1>{_e(title)}</h1>",
        f'<div class="sub">{_e(when)} &middot; {len(edl.ranges)} segment(s) &middot; '
        f"{_t(total)} output &middot; {len(cuts)} cut(s) &middot; "
        f"{len(inserts)} insert(s)</div>",
    ]

    # -- cuts
    parts.append("<h2>Cuts</h2>")
    if cuts:
        parts.append("<table><thead><tr><th>Output time</th><th>Class</th>"
                     "<th>Removed</th><th>Reason</th></tr></thead><tbody>")
        for c in cuts:
            src = (f'<div class="muted">{_e(c.source)} '
                   f"{c.source_span[0]:.2f}-{c.source_span[1]:.2f}s</div>"
                   if c.source_span else "")
            removed = (f'<div class="muted">-{c.removed_s:.2f}s</div>'
                       if isinstance(c.removed_s, (int, float)) else "")
            eyeball = ('<span class="tag warn">eyeball this boundary</span>'
                       if c.needs_visual_check else "")
            parts.append(
                f'<tr><td class="t">{_t(c.t_output)}{removed}</td>'
                f'<td><span class="tag">{_e(c.klass)}</span></td>'
                f'<td class="txt" dir="auto">{_e(c.text)}{src}</td>'
                f'<td class="txt" dir="auto">{_e(c.reason)}{eyeball}</td></tr>')
        parts.append("</tbody></table>")
    else:
        parts.append('<div class="empty">No cuts recorded on this EDL.</div>')

    # -- inserts
    parts.append("<h2>Inserts</h2>")
    if inserts:
        parts.append("<table><thead><tr><th>Output time</th><th>Claim</th><th>Source</th>"
                     "<th>Verification</th><th>Capture</th></tr></thead><tbody>")
        for i in inserts:
            badge = ('<span class="tag ok">verified</span>' if i.verified
                     else '<span class="tag bad">UNVERIFIED</span>')
            link = (f'<a href="{_e(i.url)}" rel="noreferrer noopener">{_e(i.url)}</a>'
                    if i.url else '<span class="tag bad">no source url</span>')
            kind = f'<div class="muted">{_e(i.source_type)}</div>' if i.source_type else ""
            thumb = (f'<img class="thumb" alt="capture" src="{i.thumbnail}">'
                     if i.thumbnail else '<span class="muted">no capture</span>')
            parts.append(
                f'<tr><td class="t">{_t(i.t_output)}<div class="muted">'
                f"{i.duration:.1f}s {_e(i.layout)}</div></td>"
                f'<td class="txt" dir="auto">{_e(i.claim)}</td>'
                f'<td class="txt" dir="auto">{link}{kind}</td>'
                f'<td class="txt" dir="auto">{badge}<div>{_e(i.evidence)}</div></td>'
                f"<td>{thumb}</td></tr>")
        parts.append("</tbody></table>")
    else:
        parts.append('<div class="empty">No inserts in this cut.</div>')

    # -- style profile delta
    parts.append("<h2>Style profile delta</h2>")
    parts.append("<table><thead><tr><th>Metric</th><th>Target</th><th>Actual</th>"
                 "<th>Delta</th><th></th></tr></thead><tbody>")
    for d in deltas:
        within = d.get("within")
        badge = ("" if within is None else
                 f'<span class="tag {"ok" if within else "warn"}">'
                 f'{"within" if within else "off"} &plusmn;{_num(d.get("tolerance"))}</span>')
        parts.append(
            f'<tr><td>{_e(d.get("metric"))}</td><td class="t">{_num(d.get("target"))}</td>'
            f'<td class="t">{_num(d.get("actual"))}</td>'
            f'<td class="t">{_num(d.get("delta"))}</td><td>{badge}</td></tr>')
    parts.append("</tbody></table>")

    # -- self evaluation
    parts.append("<h2>Self evaluation</h2>")
    if meta:
        flagged = ('<span class="tag bad">flagged</span>' if meta.get("flagged")
                   else '<span class="tag ok">clean</span>')
        parts.append(f'<div class="sub">{flagged} {meta.get("passes", "?")} pass(es), '
                     f'{meta.get("fixes", 0)} fix(es), cap {meta.get("max_passes", "?")}</div>')
    if findings:
        parts.append("<table><thead><tr><th>Time</th><th>Check</th><th>Severity</th>"
                     "<th>Finding</th></tr></thead><tbody>")
        for f in findings:
            sev = f.get("severity", "info")
            note = "" if f.get("checked", True) else '<div class="muted">not checked</div>'
            parts.append(
                f'<tr><td class="t">{_t(f.get("t_output"))}</td>'
                f'<td><code>{_e(f.get("check"))}</code></td>'
                f'<td><span class="tag {_sev_class(sev)}">{_e(sev)}</span></td>'
                f'<td class="txt" dir="auto">{_e(f.get("message"))}{note}</td></tr>')
        parts.append("</tbody></table>")
    else:
        parts.append('<div class="empty">No self-evaluation findings recorded.</div>')

    # -- fingerprint
    fp = fingerprint or {}
    parts.append("<h2>Run fingerprint</h2>")
    rows = " ".join(f"<code>{_e(k)}:{_e(v)}</code>"
                    for k, v in sorted((fp.get("configs") or {}).items()))
    parts.append(f'<div class="sub">configs <code>{_e(fp.get("hash", "-"))}</code> {rows}</div>')
    if edl.style_profile:
        parts.append(f'<div class="sub">style profile <code>{_e(edl.style_profile)}</code></div>')
    if edl.subtitles:
        parts.append(f'<div class="sub">captions <code>{_e(edl.subtitles)}</code> '
                     f"(burned last, output timeline)</div>")

    parts.append("<footer>Generated by the Zeta auto editor. Every insert above must carry a "
                 "source URL and verified evidence; an unverified insert does not ship."
                 "</footer></body></html>")
    return "\n".join(parts)


def write_report(edit_paths: EditPaths, edl: EDL, *, style_profile: Mapping | None = None,
                 self_eval: Any = None, raw_duration_s: float | None = None,
                 fingerprint: Mapping | None = None, title: str = "Zeta decision report",
                 generated_at: str | None = None) -> Path:
    edit_paths.edit.mkdir(parents=True, exist_ok=True)
    html_text = render_html(
        edl=edl,
        cuts=cut_rows(edl),
        inserts=insert_rows(edl, edit_paths.videos_dir),
        deltas=style_delta(edl, style_profile, raw_duration_s=raw_duration_s),
        self_eval=self_eval,
        fingerprint=fingerprint if fingerprint is not None else config_fingerprint(),
        title=title,
        generated_at=generated_at,
    )
    edit_paths.report.write_text(html_text, encoding="utf-8")
    return edit_paths.report


# -------- project.md ----------------------------------------------------------


@dataclass
class Session:
    """One `zeta edit` run, as the next run needs to read it."""

    strategy: str = ""
    decisions: list[Any] = field(default_factory=list)
    reasoning: list[Any] = field(default_factory=list)
    outstanding: list[Any] = field(default_factory=list)
    title: str = ""
    when: str = ""
    stats: dict = field(default_factory=dict)

    @classmethod
    def coerce(cls, session: "Session | Mapping[str, Any]") -> "Session":
        if isinstance(session, Session):
            return session
        s = dict(session or {})
        return cls(
            strategy=s.get("strategy", ""),
            decisions=list(s.get("decisions") or []),
            reasoning=list(s.get("reasoning") or s.get("reasoning_log") or []),
            outstanding=list(s.get("outstanding") or []),
            title=s.get("title", ""),
            when=s.get("when", ""),
            stats=dict(s.get("stats") or {}),
        )


def _bullets(items: Iterable[Any]) -> list[str]:
    out: list[str] = []
    for it in items:
        if isinstance(it, Mapping):
            head = it.get("decision") or it.get("what") or it.get("step") or it.get("item") or ""
            why = it.get("why") or it.get("reason") or it.get("note") or ""
            out.append(f"- {head}" + (f" - {why}" if why else ""))
        else:
            out.append(f"- {it}")
    return out or ["- (none)"]


def append_project_md(edit_paths: EditPaths, session: "Session | Mapping[str, Any]") -> Path:
    """Append this run's memory to `edit/project.md`.

    Appended, never rewritten: the value of this file is that the next session
    can read what the last one decided and why, including what it failed to
    finish. Overwriting it would erase exactly that.
    """
    s = Session.coerce(session)
    when = s.when or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    heading = s.title or "Edit session"

    lines = [f"## {heading} - {when}", ""]
    if s.stats:
        lines += ["`" + "  ".join(f"{k}={v}" for k, v in s.stats.items()) + "`", ""]
    lines += ["### Strategy", "", (s.strategy.strip() or "(not recorded)"), ""]
    lines += ["### Decisions", ""] + _bullets(s.decisions) + [""]
    lines += ["### Reasoning log", ""] + _bullets(s.reasoning) + [""]
    lines += ["### Outstanding", ""] + _bullets(s.outstanding) + [""]

    path = edit_paths.project_md
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if not existing:
        existing = ("# Project memory\n\nSession memory for the Zeta auto editor: what was "
                    "decided on each run, why, and what is still open.\n\n")
    elif not existing.endswith("\n\n"):
        existing = existing.rstrip("\n") + "\n\n"
    path.write_text(existing + "\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Write decision_report.html from an EDL")
    ap.add_argument("edl", type=Path)
    ap.add_argument("--videos-dir", type=Path, default=None)
    ap.add_argument("--style-profile", type=Path, default=None)
    ap.add_argument("--self-eval", type=Path, default=None,
                    help="edit/verify/self_eval.json (default: alongside the EDL)")
    ap.add_argument("--raw-duration", type=float, default=None,
                    help="source duration in seconds, for the actual cut ratio")
    ap.add_argument("--title", default="Zeta decision report")
    args = ap.parse_args(argv)

    edl = EDL.load(args.edl)
    paths = EditPaths.for_videos_dir(args.videos_dir or args.edl.resolve().parent.parent)
    profile = json.loads(args.style_profile.read_text(encoding="utf-8")) if args.style_profile else None
    se_path = args.self_eval or (paths.verify / "self_eval.json")
    self_eval = json.loads(se_path.read_text(encoding="utf-8")) if se_path.exists() else None

    out = write_report(paths, edl, style_profile=profile, self_eval=self_eval,
                       raw_duration_s=args.raw_duration, title=args.title)
    print(f"{out}  ({out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
