"""Does the pixel actually show the claim?

Stage 5 step 4 and the enforcement point for Zeta hard rule 12: nothing
unverified ships. A capture that the vision model cannot confirm is retried once
with a different selector -- the usual failure is a selector that grabbed the
nav bar instead of the headline -- and then the slot is DROPPED and the reason
recorded. There is no third state: a slot is verified or it is not in the video.

    python helpers/verify_screenshot.py /videos/edit/screenshots/slot_01
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

if __package__ in (None, ""):  # `python helpers/verify_screenshot.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helpers.gemini_client import LLM, DEFAULT_VISION_MODEL

VERIFY_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "visible": {"type": "boolean"},
        "evidence": {"type": "string"},
    },
    "required": ["visible", "evidence"],
}

_SYSTEM = (
    "You check screenshots for a news channel before they go on air. Judge ONLY "
    "what is legible in the image. If the claim is not visibly present -- wrong "
    "section of the page, cookie wall, blank frame, text too small to read, or "
    "the figure simply absent -- answer visible: false. Never infer from the "
    "domain, the layout or your own knowledge. `evidence` quotes the words or "
    "figures in the image that carry the claim, or says what is there instead."
)


@dataclass
class Verdict:
    """The vision check's answer for one slot."""

    visible: bool
    evidence: str
    attempt: int = 0
    selector: str | None = None
    image: str | None = None
    model: str = DEFAULT_VISION_MODEL
    error: str | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


def verify_prompt(claim: str, story: str = "") -> str:
    # Zeta inserts are headline cards: the page must visibly be the article or
    # post about the claim, with its headline readable. A captcha, bot wall,
    # cookie wall, error page, ad or unrelated story is a no.
    # Same STORY, not same wording: "in talks to be acquired for $13B" was
    # rejected for "valued at $13 billion". A headline card shows what the
    # video is talking about; it is not proof of one figure.
    return (f"This screenshot illustrates a moment of a news video. It should show: {claim}\n"
            "Is it clearly that page, article, product, organisation or post (its "
            "headline, name or logo readable)? Different wording is fine. "
            + (f"A headline about the video's main story also counts: {story}. " if story else "")
            + "A captcha, bot check, cookie wall, error page, advert, or something "
            "about a different subject is NO.\n"
            "Answer JSON {visible: bool, evidence: str}.")


def verify_image(image: str | Path, claim: str, *, llm: LLM | None = None, story: str = "",
                 attempt: int = 0, selector: str | None = None) -> Verdict:
    """One vision call. A failed call is `visible: false`, never an exception.

    An unreachable model must behave exactly like a model that said no: the slot
    drops. Turning a verification outage into a shipped screenshot would be the
    one failure mode this module exists to prevent.
    """
    llm = llm or LLM(model=DEFAULT_VISION_MODEL)
    img = Path(image)
    if not img.exists():
        return Verdict(visible=False, evidence="", attempt=attempt, selector=selector,
                       image=str(img), error=f"no image at {img}")
    try:
        out = llm.vision_json(verify_prompt(claim, story), img, VERIFY_SCHEMA)
    except Exception as exc:
        return Verdict(visible=False, evidence="", attempt=attempt, selector=selector,
                       image=str(img), error=f"vision check unavailable: {exc}")

    if not isinstance(out, dict) or "visible" not in out:
        return Verdict(visible=False, evidence="", attempt=attempt, selector=selector,
                       image=str(img), error=f"unusable verdict: {out!r}")
    return Verdict(visible=bool(out.get("visible")),
                   evidence=str(out.get("evidence") or ""),
                   attempt=attempt, selector=selector, image=str(img))


def verify_slot(slot_dir: str | Path, *, llm: LLM | None = None,
                aspect: str | None = None, sources_cfg: dict | None = None,
                layout_cfg: dict | None = None, capture_fn=None,
                max_retries: int | None = None) -> Verdict:
    """Verify a captured slot, retrying once with a different selector.

    Writes the verdict into the slot's `meta.json` either way: `verified` true
    with the evidence, or false with `dropped_reason`. The EDL validator refuses
    any overlay whose meta is not `verified: true`, so a dropped slot cannot be
    rendered even by mistake.
    """
    d = Path(slot_dir)
    meta_path = d / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    claim = meta.get("claim") or ""
    selectors = meta.get("selectors") or []

    if capture_fn is None:
        from helpers.screenshot import capture_for_slot as capture_fn  # lazy
    if max_retries is None:
        cfg = sources_cfg
        if cfg is None:
            from helpers import config as cfgmod
            cfg = cfgmod.load("sources")
        max_retries = int((cfg.get("capture") or {}).get("retries_per_slot", 1))

    attempts: list[Verdict] = []
    verdict = verify_image(meta.get("image", ""), claim, llm=llm, attempt=0, story=meta.get("story", ""),
                           selector=selectors[0] if selectors else None)
    attempts.append(verdict)

    attempt = 1
    while not verdict.visible and attempt <= max_retries:
        # A different selector is the only retry worth making: re-shooting the
        # same region gets the same pixels and the same answer.
        if attempt >= len(selectors):
            break
        try:
            capture_fn(d, meta, aspect=aspect, attempt=attempt,
                       sources_cfg=sources_cfg, layout_cfg=layout_cfg)
        except Exception as exc:
            attempts.append(Verdict(visible=False, evidence="", attempt=attempt,
                                    selector=selectors[attempt],
                                    error=f"retry capture failed: {exc}"))
            break
        verdict = verify_image(meta.get("image", ""), claim, llm=llm, attempt=attempt, story=meta.get("story", ""),
                               selector=selectors[attempt])
        attempts.append(verdict)
        attempt += 1

    meta["verification"] = {"attempts": [v.to_dict() for v in attempts],
                            "final": verdict.to_dict()}
    meta["verified"] = bool(verdict.visible)
    meta["evidence"] = verdict.evidence if verdict.visible else ""
    if verdict.visible:
        meta["dropped_reason"] = None
    else:
        meta["dropped_reason"] = (
            verdict.error or
            f"vision check found no evidence of the claim after "
            f"{len(attempts)} attempt(s): {verdict.evidence or 'no evidence'}")
    d.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return verdict


def verify_all(screenshots_dir: str | Path, **kw) -> dict[str, Verdict]:
    out: dict[str, Verdict] = {}
    for d in sorted(Path(screenshots_dir).iterdir()):
        if (d / "meta.json").exists():
            out[d.name] = verify_slot(d, **kw)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Vision-verify captured screenshots; drop what cannot be proven")
    ap.add_argument("slot", nargs="+", help="slot directory, or edit/screenshots for all")
    ap.add_argument("--aspect", help="output aspect, used if a retry recaptures")
    args = ap.parse_args()

    dropped = 0
    for target in args.slot:
        p = Path(target)
        results = ({p.name: verify_slot(p)} if (p / "meta.json").exists()
                   else verify_all(p, aspect=args.aspect))
        for name, v in results.items():
            print(f"  {name:10s} {'VERIFIED' if v.visible else 'DROPPED ':9s} "
                  f"{v.evidence or v.error or ''}")
            dropped += 0 if v.visible else 1
    if dropped:
        print(f"{dropped} slot(s) dropped: they will not appear in the EDL")


if __name__ == "__main__":
    main()
