"""Claims -> sources -> `edit/shots.yml`.

Stage 5 steps 1 and 2. The planner marks WHERE a visual belongs and WHAT must be
visible; this module decides WHICH page is allowed to prove it.

Zeta hard rule 12 lives here: the owner's original source beats secondary
coverage, and a URL is never invented. Every candidate that was looked at --
including the ones thrown away and why -- is written into the slot's `meta.json`
so the decision report can show the reasoning, not just the winner.

The whole search backend is one function, `resolve_source(claim, cfg, llm)`.
Swapping Gemini search grounding for a search API means replacing that function
and nothing else: callers only ever see `Candidate`.

    python helpers/research.py plan.json -v /videos --links links.txt
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import urlparse

import yaml

if __package__ in (None, ""):  # `python helpers/research.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helpers import config as cfgmod
from helpers.gemini_client import LLM
from helpers.paths import EditPaths

# Domains that are "the official filing" rather than "a company page". Kept in
# code rather than configs/sources.yaml because it is a property of the world
# (regulators publish on government hosts), not a per-channel editorial choice.
OFFICIAL_FILING_HOSTS = (
    "sec.gov", "efts.sec.gov", "federalregister.gov", "europa.eu", "ec.europa.eu",
    "gov.uk", "gouv.fr", "bkam.ma", "finances.gov.ma", "cmr.gov.ma", "hcp.ma",
)

# Ordered fallbacks for `--selector`. The first that exists on the page wins;
# index 1 is what verify_screenshot.py retries with after a failed vision check.
SELECTOR_PRESETS: dict[str, list[str]] = {
    "x": ['article[data-testid="tweet"]', "article", "main"],
    "article": ["article header", "article", "main h1", "main"],
    "filing": ["#main-content", "article", "main", "body"],
}

_NUMBER_RE = re.compile(
    r"(?<![\w.])"
    r"(?:[$€£]\s?)?\d[\d\s,.]*\s*"
    r"(?:%|percent|bn|billion|million|milliard|mliar|md|k|trillion)?"
    r"(?![\w])", re.IGNORECASE)
_DATE_RE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}[/.]\d{1,2}[/.]\d{2,4}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s*\d{0,4}|"
    r"\b(?:19|20)\d{2})\b", re.IGNORECASE)
_QUOTE_RE = re.compile(r"[\"“«]([^\"”»]{6,240})[\"”»]")
_URL_RE = re.compile(r"https?://[^\s,;<>\"')\]]+")


# --------------------------------------------------------------------------
# Data


@dataclass
class Claim:
    """One thing that must be visible on screen, and where it is spoken."""

    slot_id: str
    claim: str
    entity: str | None = None
    kind: str = "entity"            # entity | number | quote | date
    after_text: str | None = None   # planner anchor: words the visual precedes
    trigger_word: str | None = None
    number: str | None = None
    quote: str | None = None
    date: str | None = None
    prefer_source: str | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v not in (None, "")}

    @classmethod
    def from_dict(cls, d: dict) -> "Claim":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def search_query(self) -> str:
        bits = [self.entity or "", self.claim]
        for extra in (self.number, self.date):
            if extra and extra not in self.claim:
                bits.append(extra)
        return " ".join(b.strip() for b in bits if b and b.strip())


@dataclass
class Candidate:
    """A page that could carry the claim, plus the verdict on it.

    `considered` is only populated on the object `resolve_source` returns: it is
    the full audit trail of the resolution, winner included, in the order the
    resolver looked at them.
    """

    url: str | None = None
    title: str | None = None
    domain: str | None = None
    # owner | official_filing | press | unlisted | unresolved (and, set later by
    # screenshot.py for non-web evidence: pdf | chart)
    source_type: str = "unresolved"
    origin: str = "search"          # links_txt | search | prefer_source
    rank: int = 99                  # lower wins; see _PRIORITY_RANK
    why: str | None = None
    rejected_reason: str | None = None
    chosen: bool = False
    considered: list["Candidate"] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return bool(self.url) and self.rejected_reason is None

    def to_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items()
             if k != "considered" and v not in (None, "", False)}
        if self.considered:
            d["considered"] = [c.to_dict() for c in self.considered]
        return d


# links.txt is the human's own list, so it outranks anything a model found.
_PRIORITY_RANK = {"links_txt": 0, "owner_domain": 1, "official_filing": 2,
                  "reputable_press": 3}


# --------------------------------------------------------------------------
# Claim extraction (Stage 5 step 1)


def _window_after(text: str, anchor: str | None, chars: int = 240) -> str:
    """The stretch of `kept_text` a visual is talking over.

    The planner's anchor is the text the insert appears BEFORE, so the numbers
    and dates that belong to the claim are the ones spoken right after it.
    """
    if not anchor:
        return text[:chars]
    idx = text.find(anchor)
    if idx < 0:  # anchor words may differ in spacing; retry on the first word
        first = anchor.split()[0] if anchor.split() else ""
        idx = text.find(first) if first else -1
    if idx < 0:
        return ""
    return text[idx:idx + chars]


def classify_claim(claim_text: str, context: str = "") -> str:
    blob = f"{claim_text} {context}"
    if _QUOTE_RE.search(claim_text):
        return "quote"
    if _NUMBER_RE.search(claim_text) and any(ch.isdigit() for ch in claim_text):
        # A bare year is a date, not a figure.
        digits = _NUMBER_RE.findall(claim_text)
        if digits and not re.fullmatch(r"\s*(19|20)\d{2}\s*", claim_text):
            return "number"
    if _DATE_RE.search(blob) and any(ch.isdigit() for ch in blob):
        return "date"
    return "entity"


def extract_claims(kept_text: str, inserts: Sequence[dict],
                   *, slot_prefix: str = "slot") -> list[Claim]:
    """Planner insert markers + the kept transcript -> typed claims.

    The marker carries the claim in prose; the transcript around the anchor is
    mined for the concrete figure, quote or date so the resolver can search for
    the number itself and the verifier can look for it in the pixels.
    """
    claims: list[Claim] = []
    for i, ins in enumerate(inserts, start=1):
        anchor = (ins.get("after_text") or "").strip() or None
        ctx = _window_after(kept_text, anchor)
        claim_text = (ins.get("claim") or "").strip() or (ins.get("entity") or "").strip()
        if not claim_text:
            continue

        number = _first(_NUMBER_RE, claim_text) or _first(_NUMBER_RE, ctx)
        quote = _first(_QUOTE_RE, claim_text, group=1) or _first(_QUOTE_RE, ctx, group=1)
        date = _first(_DATE_RE, claim_text) or _first(_DATE_RE, ctx)

        kind = ins.get("kind") or classify_claim(claim_text, ctx)
        trigger = ins.get("trigger_word") or (anchor.split()[0] if anchor else None)
        claims.append(Claim(
            slot_id=f"{slot_prefix}_{i:02d}",
            claim=claim_text,
            entity=(ins.get("entity") or "").strip() or None,
            kind=kind,
            after_text=anchor,
            trigger_word=trigger,
            number=(number or "").strip() or None,
            quote=(quote or "").strip() or None,
            date=(date or "").strip() or None,
            prefer_source=(ins.get("prefer_source") or "").strip() or None,
        ))
    return claims


def _first(rx: re.Pattern[str], text: str, group: int = 0) -> str | None:
    m = rx.search(text or "")
    return m.group(group) if m else None


# --------------------------------------------------------------------------
# Domain policy


def domain_of(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def is_blocked(url: str, cfg: dict) -> bool:
    """Blocklist entries are bare domains (match subdomains too) or globs."""
    host = domain_of(url)
    if not host:
        return True
    for pattern in cfg.get("blocklist", []) or []:
        p = str(pattern).lower().lstrip(".")
        if "*" in p or "?" in p:
            if fnmatch.fnmatch(host, p):
                return True
        elif host == p or host.endswith("." + p):
            return True
    return False


def owner_entry(entity: str | None, cfg: dict) -> tuple[str | None, dict]:
    """Case-insensitive, space-insensitive lookup in `sources.yaml: owners`."""
    if not entity:
        return None, {}
    want = re.sub(r"[^a-z0-9]", "", entity.lower())
    for name, block in (cfg.get("owners") or {}).items():
        if re.sub(r"[^a-z0-9]", "", str(name).lower()) == want:
            return name, block or {}
    return None, {}


def _host_matches(host: str, domain: str) -> bool:
    d = str(domain).lower().lstrip(".")
    return host == d or host.endswith("." + d)


def classify_source(url: str, claim: Claim, cfg: dict) -> str:
    """owner / official_filing / press / unlisted for one URL."""
    host = domain_of(url)
    _name, owner = owner_entry(claim.entity, cfg)
    for d in owner.get("domains", []) or []:
        if _host_matches(host, d):
            return "owner"
    # Any listed company's own site is first-hand, whatever the planner typed as
    # the entity: it wrote "AI agents" and openai.com's incident post, and
    # Hugging Face's own timeline, were thrown out as unlisted.
    for entry in (cfg.get("owners") or {}).values():
        if any(_host_matches(host, d) for d in (entry or {}).get("domains", []) or []):
            return "owner"
    if any(_host_matches(host, h) for h in OFFICIAL_FILING_HOSTS) or host.endswith(".gov"):
        return "official_filing"
    for d in cfg.get("reputable_press", []) or []:
        if _host_matches(host, d):
            return "press"
    return "unlisted"


_TYPE_TO_PRIORITY = {"owner": "owner_domain", "official_filing": "official_filing",
                     "press": "reputable_press"}


def _rank(source_type: str, origin: str, cfg: dict) -> int:
    priority = [str(p) for p in (cfg.get("priority") or list(_PRIORITY_RANK))]
    key = "links_txt" if origin == "links_txt" else _TYPE_TO_PRIORITY.get(source_type, "")
    return priority.index(key) if key in priority else 99


def make_candidate(url: str, claim: Claim, cfg: dict, *, origin: str = "search",
                   title: str | None = None, why: str | None = None) -> Candidate:
    """Classify one URL and decide, without any network, whether it may ship."""
    url = (url or "").strip()
    cand = Candidate(url=url or None, title=title, why=why, origin=origin)
    if not url or not url.lower().startswith(("http://", "https://")):
        cand.rejected_reason = "not an http(s) url"
        return cand
    cand.domain = domain_of(url)
    if is_blocked(url, cfg):
        cand.source_type = "blocked"
        cand.rejected_reason = f"blocklisted domain {cand.domain}"
        return cand
    cand.source_type = classify_source(url, claim, cfg)
    cand.rank = _rank(cand.source_type, origin, cfg)
    if cand.source_type == "unlisted" and origin != "links_txt":
        # Not in owners, not a filing host, not in reputable_press. The
        # blocklist cannot enumerate the whole content-farm internet, so an
        # unknown domain found by search is refused rather than trusted.
        cand.rejected_reason = (
            f"{cand.domain} is not an owner domain, an official filing host, "
            f"or in reputable_press")
    return cand


# --------------------------------------------------------------------------
# The swappable resolver (Stage 5 step 2)


SEARCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "title": {"type": "string"},
                    "why": {"type": "string"},
                },
                "required": ["url"],
            },
        }
    },
    "required": ["candidates"],
}

_SEARCH_SYSTEM = (
    "You find the ORIGINAL published page that proves a claim for a news video. "
    "Return only URLs you actually saw in search results; never construct or "
    "guess a URL. Prefer, in order: the entity's own site or official account, "
    "an official filing or regulator page, then a major news outlet. Return up "
    "to 6 candidates, best first."
)


def resolve_source(claim: Claim, cfg: dict, llm: LLM | None = None) -> Candidate:
    """Pick the page that must be screenshotted for `claim`.

    THIS IS THE SWAP POINT. `cfg` is `configs/sources.yaml` with one extra key
    the session injects: `links` -- the URLs from `links.txt`. Replacing Gemini
    search grounding with a search API means rewriting this function body only;
    callers depend on the `Candidate` return and nothing else.

    Always returns a `Candidate`: on failure it is an `unresolved` one carrying
    the audit trail, because "we found nothing usable" is a reportable result,
    not an exception. It never returns a URL the model merely invented: every
    candidate is one a human pasted or a grounded search returned.
    """
    considered: list[Candidate] = []

    # 1. links.txt -- the human already did the research.
    for url in _links_for_claim(claim, cfg):
        considered.append(make_candidate(url, claim, cfg, origin="links_txt",
                                         why="from links.txt"))

    # 2. The planner's own `prefer_source`, if it names a real URL. It is only a
    #    hint from the plan, so it is classified by the same domain policy.
    if claim.prefer_source and _URL_RE.match(claim.prefer_source.strip()):
        considered.append(make_candidate(claim.prefer_source.strip(), claim, cfg,
                                         origin="prefer_source",
                                         why="planner prefer_source"))

    chosen = _best(considered)
    if chosen is None:
        for cand in _search_candidates(claim, cfg, llm):
            considered.append(cand)
        chosen = _best(considered)

    if chosen is None:
        out = Candidate(source_type="unresolved",
                        rejected_reason="no allowed source carried this claim")
        out.considered = considered
        return out

    chosen.chosen = True
    out = Candidate(**{k: v for k, v in asdict(chosen).items() if k != "considered"})
    out.considered = considered
    return out


def _best(candidates: Iterable[Candidate]) -> Candidate | None:
    usable = [c for c in candidates if c.usable]
    if not usable:
        return None
    # Stable within a rank: earlier candidates are the better-ranked search hits.
    return min(enumerate(usable), key=lambda pair: (pair[1].rank, pair[0]))[1]


def _links_for_claim(claim: Claim, cfg: dict) -> list[str]:
    """Which pasted links plausibly belong to this claim.

    A links.txt with a single URL and several claims would otherwise attach that
    URL to everything, so a link is taken only when it is tied to the entity:
    by the entity's own domain, or by its name appearing in the URL. With no
    entity at all there is nothing to tie, and search decides instead.
    """
    links = [str(u) for u in (cfg.get("links") or [])]
    if not links or not claim.entity:
        return []
    _name, owner = owner_entry(claim.entity, cfg)
    owner_domains = [str(d) for d in (owner.get("domains") or [])]
    slug = re.sub(r"[^a-z0-9]", "", claim.entity.lower())
    out: list[str] = []
    for url in links:
        host = domain_of(url)
        blob = re.sub(r"[^a-z0-9]", "", url.lower())
        if any(_host_matches(host, d) for d in owner_domains) or (slug and slug in blob):
            out.append(url)
    return out


def _claude_search(prompt: str, scfg: dict) -> dict:
    """Web search through headless Claude Code, which bills Ali's subscription.

    Run from a temp dir with no setting sources so neither this repo's session
    hook nor any user plugin loads into a search call.
    """
    cmd = ["claude", "-p", prompt, "--output-format", "json",
           "--model", str(scfg.get("model") or "sonnet"),
           "--system-prompt", _SEARCH_SYSTEM, "--json-schema", json.dumps(SEARCH_SCHEMA),
           "--tools", "WebSearch", "--allowedTools", "WebSearch",
           "--setting-sources", "", "--strict-mcp-config", "--no-session-persistence"]
    try:
        run = subprocess.run(cmd, capture_output=True, text=True, cwd=tempfile.gettempdir(),
                             timeout=float(scfg.get("timeout_s") or 240))
    except FileNotFoundError as exc:
        raise RuntimeError("the `claude` CLI is not on PATH; install Claude Code or "
                           "set search.backend: gemini in configs/sources.yaml") from exc
    try:
        out = json.loads(run.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"claude -p exited {run.returncode}: "
                           f"{(run.stderr or run.stdout).strip()[:300]}") from None
    if out.get("is_error") or not isinstance(out.get("structured_output"), dict):
        raise RuntimeError(f"claude -p failed: {str(out.get('result'))[:300]}")
    return out["structured_output"]


def _search_candidates(claim: Claim, cfg: dict, llm: LLM | None) -> list[Candidate]:
    """Grounded search -> classified candidates. Never raises; logs instead."""
    prompt = (
        f"Claim to prove on screen: {claim.claim}\n"
        f"Entity: {claim.entity or 'unknown'}\n"
        f"Claim type: {claim.kind}\n"
        + (f"Figure that must appear: {claim.number}\n" if claim.number else "")
        + (f"Quote that must appear: {claim.quote}\n" if claim.quote else "")
        + (f"Date: {claim.date}\n" if claim.date else "")
        + f"Search query to start from: {claim.search_query()}\n"
        "Return the pages that best SHOW this on screen: the article itself, the "
        "official page or post, or the product page. Prefer the original over coverage."
    )
    scfg = cfg.get("search") or {}
    try:
        # Mock mode stays on the Gemini mock: tests never hit the network.
        if scfg.get("backend") == "claude_cli" and not os.environ.get("ZETA_LLM_MOCK"):
            out = _claude_search(prompt, scfg)
        else:
            out = (llm or LLM()).generate(prompt, system=_SEARCH_SYSTEM, schema=SEARCH_SCHEMA,
                                          tools=("google_search",))
    except Exception as exc:  # a dead search must flag the slot, not kill the run
        bad = Candidate(source_type="unresolved", origin="search",
                        rejected_reason=f"search unavailable: {exc}")
        return [bad]

    rows = (out or {}).get("candidates", []) if isinstance(out, dict) else []
    return [make_candidate(str(r.get("url", "")), claim, cfg, origin="search",
                           title=r.get("title"), why=r.get("why"))
            for r in rows if isinstance(r, dict)]


# --------------------------------------------------------------------------
# Slots, meta and shots.yml


def selector_preset(url: str | None, claim: Claim, cfg: dict) -> list[str]:
    host = domain_of(url or "")
    if host in ("x.com", "twitter.com") or host.endswith((".x.com", ".twitter.com")):
        return list(SELECTOR_PRESETS["x"])
    if classify_source(url or "", claim, cfg) == "official_filing":
        return list(SELECTOR_PRESETS["filing"])
    return list(SELECTOR_PRESETS["article"])


@dataclass
class Slot:
    """One screenshot slot: a claim, its chosen source, and where it lands."""

    claim: Claim
    candidate: Candidate
    selectors: list[str] = field(default_factory=list)
    dropped_reason: str | None = None

    @property
    def slot_id(self) -> str:
        return self.claim.slot_id

    @property
    def shippable(self) -> bool:
        return self.candidate.usable and self.dropped_reason is None

    def meta(self) -> dict:
        """The slot's `meta.json`: everything the report and the EDL need."""
        return {
            "slot_id": self.slot_id,
            "claim": self.claim.claim,
            "claim_detail": self.claim.to_dict(),
            "url": self.candidate.url,
            "source_type": self.candidate.source_type,
            "source_origin": self.candidate.origin,
            "selectors": self.selectors,
            "candidates": [c.to_dict() for c in self.candidate.considered],
            "verified": False,          # only verify_screenshot.py may set this
            "dropped_reason": self.dropped_reason,
        }


def build_slots(claims: Sequence[Claim], cfg: dict, llm: LLM | None = None) -> list[Slot]:
    from concurrent.futures import ThreadPoolExecutor

    # Each search is a ~20 s subprocess; one after another made research the
    # slowest stage of a 3 min edit.
    with ThreadPoolExecutor(max_workers=4) as pool:
        cands = list(pool.map(lambda c: resolve_source(c, cfg, llm), claims))
    slots: list[Slot] = []
    for claim, cand in zip(claims, cands):
        slot = Slot(claim=claim, candidate=cand,
                    selectors=selector_preset(cand.url, claim, cfg))
        if not cand.usable:
            slot.dropped_reason = cand.rejected_reason or "unresolved source"
        slots.append(slot)
    return slots


def write_slot_meta(slot: Slot, paths: EditPaths, extra: dict | None = None) -> Path:
    d = paths.slot(slot.slot_id)
    d.mkdir(parents=True, exist_ok=True)
    meta = slot.meta()
    if extra:
        meta.update(extra)
    p = d / "meta.json"
    p.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return p


def shots_entries(slots: Sequence[Slot], *, width: int, retina: bool,
                  timeout_ms: int, javascript: str | None = None,
                  wait_for: str | None = None, wait_ms: int = 0) -> list[dict]:
    """shot-scraper `multi` batch entries, one per shippable slot.

    Shape per https://shot-scraper.datasette.io `shot-scraper multi shots.yml`:
    a top-level YAML list of mappings with `url` and `output`. Dropped slots are
    omitted -- there is nothing to capture for a claim with no allowed source.
    """
    out: list[dict] = []
    for slot in slots:
        if not slot.shippable:
            continue
        entry: dict[str, Any] = {
            "url": slot.candidate.url,
            "output": f"{slot.slot_id}/shot.png",
            "width": int(width),
            "retina": bool(retina),
            "timeout": int(timeout_ms),
        }
        if slot.selectors:
            entry["selector"] = slot.selectors[0]
        if wait_for:
            entry["wait_for"] = wait_for
        if wait_ms:
            entry["wait"] = int(wait_ms)
        if javascript:
            entry["javascript"] = javascript
        out.append(entry)
    return out


def write_shots_yml(slots: Sequence[Slot], paths: EditPaths, *, aspect: str | None = None,
                    sources_cfg: dict | None = None, layout_cfg: dict | None = None) -> Path:
    """Emit `edit/shots.yml` in shot-scraper multi format."""
    from helpers.screenshot import (NETWORK_IDLE_JS, capture_settings,  # lazy: avoids a cycle
                                    cookie_banner_js)

    scfg = sources_cfg if sources_cfg is not None else cfgmod.load("sources")
    settings = capture_settings(aspect=aspect, sources_cfg=scfg, layout_cfg=layout_cfg)
    entries = shots_entries(
        slots,
        width=settings["width"], retina=settings["retina"],
        timeout_ms=settings["timeout_ms"],
        javascript=cookie_banner_js() if settings["dismiss_cookie_banners"] else None,
        wait_for=NETWORK_IDLE_JS if settings["wait_for_network_idle"] else None,
    )
    p = paths.edit / "shots.yml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(entries, sort_keys=False, allow_unicode=True),
                 encoding="utf-8")
    return p


def read_links(path: str | Path | None) -> list[str]:
    """`links.txt`: one URL per line, `#` comments and labels tolerated."""
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    urls: list[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        urls.extend(_URL_RE.findall(line))
    return urls


def research(plan: dict, videos_dir: str | Path, *, links: Sequence[str] = (),
             sources_cfg: dict | None = None, aspect: str | None = None,
             llm: LLM | None = None) -> dict:
    """Plan -> slots -> `meta.json` per slot -> `edit/shots.yml`."""
    cfg = dict(sources_cfg if sources_cfg is not None else cfgmod.load("sources"))
    cfg["links"] = list(links) + list(cfg.get("links") or [])
    paths = EditPaths.for_videos_dir(videos_dir)
    paths.ensure()

    claims = extract_claims(plan.get("kept_text", ""), plan.get("inserts", []) or [])
    slots = build_slots(claims, cfg, llm)
    for slot in slots:
        write_slot_meta(slot, paths)
    shots = write_shots_yml(slots, paths, aspect=aspect, sources_cfg=cfg)

    return {
        "slots": [s.meta() for s in slots],
        "shots_yml": str(shots),
        "shippable": [s.slot_id for s in slots if s.shippable],
        "dropped": {s.slot_id: s.dropped_reason for s in slots if not s.shippable},
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract claims from an edit plan and resolve their sources")
    ap.add_argument("plan", help="edit_plan.json with kept_text and inserts")
    ap.add_argument("-v", "--videos-dir", required=True, help="session videos dir")
    ap.add_argument("--links", help="links.txt with URLs Ali already has")
    ap.add_argument("--aspect", help="output aspect (default: configs/layout.yaml)")
    ap.add_argument("--json", action="store_true", help="print the result as JSON")
    args = ap.parse_args()

    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    out = research(plan, args.videos_dir, links=read_links(args.links), aspect=args.aspect)

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=1))
        return
    print(f"shots.yml: {out['shots_yml']}")
    for slot in out["slots"]:
        state = "DROPPED" if slot["dropped_reason"] else slot["source_type"]
        print(f"  {slot['slot_id']:10s} {state:16s} {slot['url'] or slot['dropped_reason']}")


if __name__ == "__main__":
    main()
