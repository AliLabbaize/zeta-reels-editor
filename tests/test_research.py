"""Source resolution is an editorial policy, so it is tested as one."""

from __future__ import annotations

import json

import pytest
import yaml

from helpers import gemini_client, research
from helpers.gemini_client import LLM
from helpers.paths import EditPaths

SOURCES = {
    "priority": ["links_txt", "owner_domain", "official_filing", "reputable_press"],
    "owners": {
        "OpenAI": {"domains": ["openai.com"], "handles": ["@OpenAI"]},
        "SpaceX": {"domains": ["spacex.com"], "handles": ["@SpaceX"]},
    },
    "reputable_press": ["reuters.com", "theverge.com"],
    "blocklist": ["facebook.com", "medium.com", "*.blogspot.com"],
    "capture": {"timeout_s": 45, "retries_per_slot": 1,
                "wait_for_network_idle": True, "dismiss_cookie_banners": True},
}


def _mock_search(urls, calls=None):
    """Install a search backend that returns `urls`, recording every call."""
    def handler(req):
        if calls is not None:
            calls.append(req)
        return {"candidates": [{"url": u, "title": "t", "why": "search hit"} for u in urls]}
    gemini_client.register_mock(handler)
    return LLM()


def _claim(entity="OpenAI", claim="OpenAI raised $40 billion", **kw):
    return research.Claim(slot_id="slot_01", claim=claim, entity=entity, **kw)


# -- claim extraction -------------------------------------------------------


def test_extract_claims_types_and_details():
    kept = ("هاد الشركة OpenAI جمعات 40 billion دولار "
            "و قال سام التمان \"this is the largest round ever\" نهار 2026-03-04")
    inserts = [
        {"after_text": "OpenAI جمعات", "claim": "OpenAI raised 40 billion",
         "entity": "OpenAI"},
        {"after_text": "قال سام التمان", "claim": "Altman said \"this is the largest round ever\"",
         "entity": "OpenAI"},
    ]
    claims = research.extract_claims(kept, inserts)

    assert [c.slot_id for c in claims] == ["slot_01", "slot_02"]
    assert claims[0].kind == "number"
    assert "40" in (claims[0].number or "")
    assert claims[1].kind == "quote"
    assert claims[1].quote == "this is the largest round ever"
    # The date is spoken after the anchor, not inside the claim text.
    assert claims[1].date == "2026-03-04"
    assert claims[0].trigger_word == "OpenAI"


def test_extract_claims_skips_markers_with_nothing_to_show():
    assert research.extract_claims("text", [{"after_text": "text"}]) == []


# -- priority ---------------------------------------------------------------


def test_links_txt_beats_search():
    calls: list[dict] = []
    llm = _mock_search(["https://openai.com/index/funding/"], calls)
    cfg = dict(SOURCES, links=["https://www.reuters.com/technology/openai-round-2026/"])

    chosen = research.resolve_source(_claim(), cfg, llm)

    assert chosen.origin == "links_txt"
    assert chosen.url == "https://www.reuters.com/technology/openai-round-2026/"
    # A link Ali pasted ends the resolution: no search is run at all.
    assert calls == []


def test_owner_domain_beats_reputable_press():
    llm = _mock_search(["https://www.reuters.com/technology/openai-round-2026/",
                        "https://openai.com/index/funding/"])

    chosen = research.resolve_source(_claim(), dict(SOURCES), llm)

    assert chosen.url == "https://openai.com/index/funding/"
    assert chosen.source_type == "owner"
    # The press hit is kept in the log even though it lost.
    logged = {c.url: c for c in chosen.considered}
    assert logged["https://www.reuters.com/technology/openai-round-2026/"].source_type == "press"
    assert sum(1 for c in chosen.considered if c.chosen) == 1


def test_official_filing_beats_press_when_no_owner_page():
    llm = _mock_search(["https://www.reuters.com/business/spacex-ipo/",
                        "https://www.sec.gov/Archives/edgar/data/spacex-s1.htm"])

    chosen = research.resolve_source(_claim(entity="SpaceX", claim="SpaceX S-1 filing"),
                                     dict(SOURCES), llm)

    assert chosen.source_type == "official_filing"
    assert "sec.gov" in chosen.url


def test_blocklisted_domain_is_never_chosen():
    llm = _mock_search(["https://medium.com/@someone/openai-round",
                        "https://openai.blogspot.com/2026/03/round.html",
                        "https://www.reuters.com/technology/openai-round-2026/"])

    chosen = research.resolve_source(_claim(), dict(SOURCES), llm)

    assert chosen.url == "https://www.reuters.com/technology/openai-round-2026/"
    rejected = {c.url: c.rejected_reason for c in chosen.considered if c.rejected_reason}
    assert "blocklisted" in rejected["https://medium.com/@someone/openai-round"]
    # The glob entry `*.blogspot.com` must match subdomains too.
    assert "blocklisted" in rejected["https://openai.blogspot.com/2026/03/round.html"]


def test_blocklist_applies_to_links_txt_too():
    llm = _mock_search(["https://openai.com/index/funding/"])
    cfg = dict(SOURCES, links=["https://www.facebook.com/openai/posts/123"])

    chosen = research.resolve_source(_claim(), cfg, llm)

    assert chosen.url == "https://openai.com/index/funding/"
    assert any("blocklisted" in (c.rejected_reason or "") for c in chosen.considered)


def test_unknown_domain_from_search_is_refused():
    llm = _mock_search(["https://ai-news-today.example/openai-round"])

    chosen = research.resolve_source(_claim(), dict(SOURCES), llm)

    assert chosen.url is None
    assert chosen.source_type == "unresolved"
    assert chosen.considered and chosen.considered[0].source_type == "unlisted"


def test_resolver_never_fabricates_when_search_is_down():
    def dead(_req):
        raise RuntimeError("no search backend")
    gemini_client.register_mock(dead)

    chosen = research.resolve_source(_claim(), dict(SOURCES), LLM())

    assert chosen.url is None
    assert "search unavailable" in (chosen.considered[-1].rejected_reason or "")


def test_links_txt_is_not_attached_to_an_unrelated_claim():
    llm = _mock_search(["https://spacex.com/updates/"])
    cfg = dict(SOURCES, links=["https://openai.com/index/funding/"])

    chosen = research.resolve_source(_claim(entity="SpaceX", claim="SpaceX launch"),
                                     cfg, llm)

    assert chosen.url == "https://spacex.com/updates/"


def test_read_links_tolerates_comments_and_labels(tmp_path):
    p = tmp_path / "links.txt"
    p.write_text("# sources\nOpenAI: https://openai.com/index/funding/  # blog\n\n"
                 "https://www.reuters.com/a\n", encoding="utf-8")
    assert research.read_links(p) == ["https://openai.com/index/funding/",
                                      "https://www.reuters.com/a"]


# -- slots, meta and shots.yml ----------------------------------------------


def _plan():
    return {"kept_text": "OpenAI raised 40 billion this week and Meta answered",
            "inserts": [{"after_text": "OpenAI raised", "claim": "OpenAI raised 40 billion",
                         "entity": "OpenAI"},
                        {"after_text": "and Meta answered", "claim": "Meta's reply post",
                         "entity": "Meta"}]}


def test_research_writes_meta_with_the_full_candidate_log(tmp_path):
    _mock_search(["https://medium.com/@x/openai", "https://openai.com/index/funding/"])
    out = research.research(_plan(), tmp_path, sources_cfg=dict(SOURCES), aspect="9:16")

    meta = json.loads((EditPaths.for_videos_dir(tmp_path).slot("slot_01")
                       / "meta.json").read_text(encoding="utf-8"))
    assert meta["url"] == "https://openai.com/index/funding/"
    assert meta["source_type"] == "owner"
    assert meta["verified"] is False          # only verify_screenshot.py flips this
    assert [c["url"] for c in meta["candidates"]] == [
        "https://medium.com/@x/openai", "https://openai.com/index/funding/"]
    assert meta["selectors"][0] == "article header"
    # Meta is not in `owners`, so openai.com is an unlisted domain for slot_02's
    # claim: the slot is dropped with a reason rather than quietly filled.
    assert out["shippable"] == ["slot_01"]
    assert out["dropped"]["slot_02"] == "no allowed source carried this claim"
    slot2 = [s for s in out["slots"] if s["slot_id"] == "slot_02"][0]
    assert all(c.get("rejected_reason") for c in slot2["candidates"])


def test_shots_yml_is_shot_scraper_multi_shape(tmp_path):
    _mock_search(["https://openai.com/index/funding/"])
    research.research(_plan(), tmp_path, sources_cfg=dict(SOURCES), aspect="9:16")

    path = EditPaths.for_videos_dir(tmp_path).edit / "shots.yml"
    entries = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert isinstance(entries, list) and entries
    for e in entries:
        assert set(e) <= {"url", "output", "selector", "width", "retina", "timeout",
                          "javascript", "wait", "wait_for"}
        assert e["url"].startswith("https://")
        assert e["output"].endswith("/shot.png")
        assert e["width"] == 800          # 9:16 capture width, from layout.yaml
        assert e["retina"] is True
        assert e["timeout"] == 45000      # sources.yaml capture.timeout_s
        assert "cookie" in e["javascript"] or "accept" in e["javascript"].lower()


def test_dropped_slots_are_absent_from_shots_yml(tmp_path):
    _mock_search(["https://medium.com/@x/openai"])
    research.research(_plan(), tmp_path, sources_cfg=dict(SOURCES))

    path = EditPaths.for_videos_dir(tmp_path).edit / "shots.yml"
    assert yaml.safe_load(path.read_text(encoding="utf-8")) in (None, [])


def test_x_post_gets_the_post_container_selector():
    claim = _claim()
    sels = research.selector_preset("https://x.com/OpenAI/status/1", claim, SOURCES)
    assert sels[0] == 'article[data-testid="tweet"]'
    assert len(sels) > 1        # verify_screenshot.py needs a different one to retry with
