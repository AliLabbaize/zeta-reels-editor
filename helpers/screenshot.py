"""Capture the evidence: web pages, PDF pages, or a rendered chart.

Stage 5 step 3. Three capture backends, one output contract: a PNG in the slot
directory plus the `source_type` that says what it actually is.

  html  -> shot-scraper (Playwright) with a CSS selector on the article header
           or post container
  pdf   -> pdftoppm on the single page that carries the number
  chart -> matplotlib, from figures already cited in the claim

CAPTURE WIDTH IS DERIVED, NEVER HARDCODED. It comes from the aspect's `capture`
block in configs/layout.yaml, because the only thing that matters is whether the
text survives at phone size: see `readability_ratio`. Capturing an article at a
desktop 1600 CSS px and fitting it into the 820 px safe column of a Reel halves
every glyph.

The chart backend is marked `source_type: "chart"` so the report can never
present a drawing of the numbers as a photograph of the source.

    python helpers/screenshot.py --url https://... -o shot.png --aspect 9:16
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):  # `python helpers/screenshot.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helpers import config as cfgmod

# Clicked in the page before the shot. Cookie walls are the single most common
# reason a capture shows a grey overlay instead of the headline. Best-effort by
# design: an unknown banner must not fail the capture, the vision check will.
COOKIE_BANNER_JS = """
(() => {
  const pats = /^(accept|agree|allow|got it|i agree|ok|tout accepter|accepter|j'accepte|consent|continue)/i;
  const ids = ['#onetrust-accept-btn-handler', '.fc-cta-consent', '#didomi-notice-agree-button',
               '[data-testid="cookie-policy-manage-dialog-accept-button"]', '.qc-cmp2-summary-buttons button'];
  for (const sel of ids) { const el = document.querySelector(sel); if (el) { el.click(); } }
  for (const el of document.querySelectorAll('button, a[role=button], [role=button]')) {
    const t = (el.innerText || '').trim();
    if (t && t.length < 40 && pats.test(t)) { el.click(); break; }
  }
  return true;
})()
"""


# shot-scraper waits for `load`; this waits for the page to settle, which is
# what "network idle" buys us -- late-loading headline images and web fonts.
NETWORK_IDLE_JS = "document.readyState === 'complete'"


class CaptureError(RuntimeError):
    """The capture did not produce a usable PNG. The slot gets dropped."""


@dataclass
class CaptureResult:
    path: Path
    source_type: str          # html | pdf | chart
    selector: str | None = None
    attempt: int = 0
    command: list[str] | None = None
    note: str | None = None

    def to_dict(self) -> dict:
        return {"image": str(self.path), "source_type": self.source_type,
                "selector": self.selector, "attempt": self.attempt,
                "command": self.command, "note": self.note}


# --------------------------------------------------------------------------
# Geometry-driven capture settings


def resolve_aspect(aspect: str | None = None, layout_cfg: dict | None = None) -> tuple[str, dict]:
    """`(aspect key, its block)` from configs/layout.yaml or an in-memory copy."""
    if layout_cfg is None:
        return cfgmod.aspect_config(aspect)
    key = aspect or layout_cfg.get("default_aspect", "9:16")
    aspects = layout_cfg.get("aspects", {})
    if key not in aspects:
        raise KeyError(f"unknown aspect {key!r}; known: {sorted(aspects)}")
    return key, aspects[key]


def safe_width_px(aspect_cfg: dict) -> int:
    w = int(aspect_cfg["resolution"][0])
    safe = aspect_cfg.get("safe_area", {}) or {}
    return w - int(safe.get("left", 0)) - int(safe.get("right", 0))


def readability_ratio(aspect_cfg: dict) -> float:
    """Delivered pixels per captured CSS pixel: `safe_width / capture.width`.

    The arithmetic the header of configs/layout.yaml demands. At >= ~1.0 one CSS
    pixel of the page survives as at least one device pixel inside the safe
    column, so body text stays readable on a phone. `retina` does not enter the
    ratio -- it doubles the pixels we DOWNSCALE from, which is what keeps the
    fit sharp instead of soft.
    """
    cap = aspect_cfg.get("capture", {}) or {}
    width = float(cap.get("width", 0) or 0)
    if width <= 0:
        raise ValueError("aspect has no capture.width")
    return safe_width_px(aspect_cfg) / width


def capture_settings(*, aspect: str | None = None, sources_cfg: dict | None = None,
                     layout_cfg: dict | None = None) -> dict:
    """Everything a capture needs, resolved from the two configs. No hardcoding."""
    key, acfg = resolve_aspect(aspect, layout_cfg)
    scfg = sources_cfg if sources_cfg is not None else cfgmod.load("sources")
    cap = acfg.get("capture", {}) or {}
    capture_policy = scfg.get("capture", {}) or {}
    width = int(cap.get("width"))
    retina = bool(cap.get("retina", False))
    timeout_s = float(capture_policy.get("timeout_s", 45))
    return {
        "aspect": key,
        "width": width,
        "retina": retina,
        # What the PNG actually contains, and what the PDF/chart backends must
        # match so all three source types fit the card identically.
        "image_width_px": width * (2 if retina else 1),
        "safe_width_px": safe_width_px(acfg),
        "readability": readability_ratio(acfg),
        "timeout_s": timeout_s,
        "timeout_ms": int(timeout_s * 1000),
        "retries": int(capture_policy.get("retries_per_slot", 1)),
        "wait_for_network_idle": bool(capture_policy.get("wait_for_network_idle", True)),
        "dismiss_cookie_banners": bool(capture_policy.get("dismiss_cookie_banners", True)),
    }


def cookie_banner_js() -> str:
    return COOKIE_BANNER_JS.strip()


# --------------------------------------------------------------------------
# html: shot-scraper


def shot_scraper_cmd(url: str, output: str | Path, *, selector: str | None,
                     width: int, retina: bool, timeout_ms: int,
                     javascript: str | None = None,
                     wait_for_network_idle: bool = True) -> list[str]:
    """The exact argv. Pure, so the tests can assert on it without Playwright."""
    cmd: list[str] = ["shot-scraper", "shot", url, "-o", str(output),
                      "--width", str(int(width))]
    if retina:
        cmd.append("--retina")
    if selector:
        cmd += ["--selector", selector]
    if wait_for_network_idle:
        cmd += ["--wait-for", NETWORK_IDLE_JS]
    if javascript:
        cmd += ["--javascript", javascript]
    cmd += ["--timeout", str(int(timeout_ms))]
    return cmd


def capture_url(url: str, out_png: str | Path, *, selector: str | None = None,
                aspect: str | None = None, sources_cfg: dict | None = None,
                layout_cfg: dict | None = None, attempt: int = 0,
                runner=subprocess.run) -> CaptureResult:
    """One shot-scraper capture. Raises `CaptureError` with the tool's stderr."""
    s = capture_settings(aspect=aspect, sources_cfg=sources_cfg, layout_cfg=layout_cfg)
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)

    if shutil.which("shot-scraper") is None and runner is subprocess.run:
        raise CaptureError(
            "shot-scraper is not installed: uv pip install -e '.[shots]' && "
            "shot-scraper install")

    cmd = shot_scraper_cmd(
        url, out, selector=selector, width=s["width"], retina=s["retina"],
        timeout_ms=s["timeout_ms"],
        javascript=cookie_banner_js() if s["dismiss_cookie_banners"] else None,
        wait_for_network_idle=s["wait_for_network_idle"])

    proc = runner(cmd, capture_output=True, text=True,
                  timeout=s["timeout_s"] + 30)
    if getattr(proc, "returncode", 1) != 0:
        raise CaptureError(f"shot-scraper failed ({proc.returncode}) on {url}: "
                           f"{(getattr(proc, 'stderr', '') or '')[-400:]}")
    if not out.exists():
        raise CaptureError(f"shot-scraper reported success but wrote no file: {out}")
    return CaptureResult(path=out, source_type="html", selector=selector,
                         attempt=attempt, command=cmd)


# --------------------------------------------------------------------------
# pdf: pdftoppm on the one page that carries the number


def pdftoppm_cmd(pdf: str | Path, page: int, prefix: str | Path, *,
                 image_width_px: int) -> list[str]:
    # -scale-to-x with -scale-to-y -1 keeps the page's aspect ratio: a PDF page
    # must never be squeezed, for the same reason a screenshot is never cropped.
    return ["pdftoppm", "-png", "-f", str(int(page)), "-l", str(int(page)),
            "-scale-to-x", str(int(image_width_px)), "-scale-to-y", "-1",
            str(pdf), str(prefix)]


def capture_pdf_page(pdf: str | Path, page: int, out_png: str | Path, *,
                     aspect: str | None = None, sources_cfg: dict | None = None,
                     layout_cfg: dict | None = None, attempt: int = 0,
                     runner=subprocess.run) -> CaptureResult:
    s = capture_settings(aspect=aspect, sources_cfg=sources_cfg, layout_cfg=layout_cfg)
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)

    if shutil.which("pdftoppm") is None and runner is subprocess.run:
        raise CaptureError("pdftoppm is not installed: it ships with poppler-utils "
                           "(apt install poppler-utils / brew install poppler)")

    prefix = out.with_suffix("")
    cmd = pdftoppm_cmd(pdf, page, prefix, image_width_px=s["image_width_px"])
    proc = runner(cmd, capture_output=True, text=True, timeout=s["timeout_s"] + 30)
    if getattr(proc, "returncode", 1) != 0:
        raise CaptureError(f"pdftoppm failed ({proc.returncode}) on {pdf} p{page}: "
                           f"{(getattr(proc, 'stderr', '') or '')[-400:]}")

    # pdftoppm appends the page number to the prefix and pads it to the page
    # count's width, so the produced name is not predictable from `page` alone.
    produced = sorted(prefix.parent.glob(f"{prefix.name}-*.png"))
    if not produced and not out.exists():
        raise CaptureError(f"pdftoppm wrote no page image for {pdf} p{page}")
    if produced:
        if out.exists():
            out.unlink()
        produced[0].replace(out)
        for leftover in produced[1:]:
            leftover.unlink()
    return CaptureResult(path=out, source_type="pdf", attempt=attempt, command=cmd,
                         note=f"page {page}")


# --------------------------------------------------------------------------
# chart: last resort, and always labelled as a rendering


def render_chart(figures: Sequence[dict] | dict, out_png: str | Path, *,
                 title: str, source_label: str, aspect: str | None = None,
                 sources_cfg: dict | None = None, layout_cfg: dict | None = None,
                 kind: str = "bar") -> CaptureResult:
    """Draw the cited figures when no page shows them legibly.

    `figures` is `[{"label": "2024", "value": 1.2}, ...]` or `{label: value}`.
    The source label is burned into the image: a chart is our rendering of
    someone else's numbers, and the viewer must be told whose numbers they are.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")           # no display in batch mode
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise CaptureError(
            "matplotlib is not installed: uv pip install -e '.[shots]'") from exc

    rows = ([{"label": str(k), "value": float(v)} for k, v in figures.items()]
            if isinstance(figures, dict)
            else [{"label": str(r.get("label")), "value": float(r.get("value"))}
                  for r in figures])
    if not rows:
        raise CaptureError("render_chart: no figures to draw")

    s = capture_settings(aspect=aspect, sources_cfg=sources_cfg, layout_cfg=layout_cfg)
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)

    dpi = 100
    width_in = s["image_width_px"] / dpi
    fig, ax = plt.subplots(figsize=(width_in, width_in * 0.62), dpi=dpi)
    labels = [r["label"] for r in rows]
    values = [r["value"] for r in rows]
    if kind == "line":
        ax.plot(labels, values, marker="o", linewidth=3, color="#1B6EF3")
    else:
        ax.bar(labels, values, color="#1B6EF3")
        for x, v in zip(labels, values):
            ax.annotate(f"{v:g}", (x, v), ha="center", va="bottom",
                        fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=17, fontweight="bold", loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=12)
    fig.text(0.01, 0.01, f"Source: {source_label}", fontsize=11, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, facecolor="white")
    plt.close(fig)

    return CaptureResult(path=out, source_type="chart",
                         note=f"rendered from cited figures, source: {source_label}")


# --------------------------------------------------------------------------
# Slot-level entry point


def capture_for_slot(slot_dir: str | Path, meta: dict, *, aspect: str | None = None,
                     attempt: int = 0, selector: str | None = None,
                     sources_cfg: dict | None = None, layout_cfg: dict | None = None,
                     capture_url_fn=capture_url, capture_pdf_fn=capture_pdf_page,
                     render_chart_fn=render_chart) -> CaptureResult:
    """Capture the evidence for one resolved slot and update its `meta.json`.

    `attempt` selects the selector from the slot's ordered `selectors` list, so
    verify_screenshot.py retries with a different one by calling back with
    `attempt=1` -- the retry policy lives there, the capture mechanics here.
    """
    d = Path(slot_dir)
    d.mkdir(parents=True, exist_ok=True)
    out = d / ("shot.png" if attempt == 0 else f"shot_retry{attempt}.png")

    url = meta.get("url") or ""
    declared = meta.get("source_type")
    kw = dict(aspect=aspect, sources_cfg=sources_cfg, layout_cfg=layout_cfg)

    if declared == "chart" or meta.get("chart"):
        chart = meta.get("chart") or {}
        result = render_chart_fn(
            chart.get("figures") or {}, out,
            title=chart.get("title") or meta.get("claim", ""),
            source_label=chart.get("source_label") or url or "cited figures", **kw)
    elif meta.get("pdf") or url.lower().endswith(".pdf"):
        pdf = meta.get("pdf") or {}
        local = pdf.get("path") or meta.get("local_pdf")
        if not local:
            raise CaptureError(
                f"{meta.get('slot_id')}: PDF source needs meta.pdf.path (the "
                f"downloaded file); {url} was not fetched")
        result = capture_pdf_fn(local, int(pdf.get("page", 1)), out,
                                attempt=attempt, **kw)
    else:
        if not url:
            raise CaptureError(f"{meta.get('slot_id')}: no url to capture")
        selectors = meta.get("selectors") or []
        sel = selector or (selectors[attempt] if attempt < len(selectors)
                           else (selectors[-1] if selectors else None))
        result = capture_url_fn(url, out, selector=sel, attempt=attempt, **kw)

    meta.setdefault("captures", []).append(result.to_dict())
    meta["image"] = str(result.path)
    meta["source_type"] = result.source_type if result.source_type in ("pdf", "chart") \
        else meta.get("source_type", result.source_type)
    meta["capture_source_type"] = result.source_type
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                 encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Capture one piece of source evidence")
    ap.add_argument("--url", help="page to screenshot with shot-scraper")
    ap.add_argument("--pdf", help="local PDF to render a page from")
    ap.add_argument("--page", type=int, default=1, help="PDF page carrying the number")
    ap.add_argument("--chart", help='JSON figures, e.g. \'{"2024": 1.2, "2025": 3.4}\'')
    ap.add_argument("--title", default="", help="chart title")
    ap.add_argument("--source-label", default="", help="chart source credit")
    ap.add_argument("--selector", help="CSS selector for the article header / post")
    ap.add_argument("-o", "--output", required=True, help="output PNG")
    ap.add_argument("--aspect", help="output aspect (default: configs/layout.yaml)")
    ap.add_argument("--settings", action="store_true",
                    help="print the resolved capture settings and exit")
    args = ap.parse_args()

    if args.settings:
        print(json.dumps(capture_settings(aspect=args.aspect), indent=1))
        return
    if args.url:
        res = capture_url(args.url, args.output, selector=args.selector, aspect=args.aspect)
    elif args.pdf:
        res = capture_pdf_page(args.pdf, args.page, args.output, aspect=args.aspect)
    elif args.chart:
        res = render_chart(json.loads(args.chart), args.output, title=args.title,
                           source_label=args.source_label or "cited figures",
                           aspect=args.aspect)
    else:
        ap.error("one of --url, --pdf or --chart is required")
    print(json.dumps(res.to_dict(), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
