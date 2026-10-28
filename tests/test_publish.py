"""`zeta publish` without a network: platforms are faked at PLATFORM_FUNCS."""

from __future__ import annotations

import datetime as dt
import hashlib
import json

import pytest

from helpers import gemini_client, publish

MB = publish.MB
LIMITS = {"caption_chars": 2200, "ig_hashtags": 30, "ig_mentions": 20}


def test_pkce_challenge_is_hex_sha256_of_verifier():
    verifier, challenge = publish.pkce_pair()
    assert 43 <= len(verifier) <= 128
    assert challenge == hashlib.sha256(verifier.encode()).hexdigest()


@pytest.mark.parametrize("size", [1, 5 * MB - 1, 64 * MB, 64 * MB + 1, 300 * MB + 123, 4096 * MB])
def test_chunks_cover_the_file_within_tiktok_limits(size):
    chunk, count = publish.chunk_plan(size)
    ranges = publish.chunk_ranges(size, chunk, count)
    assert ranges[0][0] == 0 and ranges[-1][1] == size - 1
    assert all(a[1] + 1 == b[0] for a, b in zip(ranges, ranges[1:]))
    sizes = [b - a + 1 for a, b in ranges]
    if count > 1:
        assert all(5 * MB <= s <= 64 * MB for s in sizes[:-1]) and sizes[-1] <= 128 * MB
    assert 1 <= count <= 1000


def test_rejects_empty_and_oversize():
    for size in (0, 4096 * MB + 1):
        with pytest.raises(publish.PublishError):
            publish.chunk_plan(size)


def test_privacy_has_no_default():
    opts = ["PUBLIC_TO_EVERYONE", "SELF_ONLY"]
    assert publish.check_privacy("SELF_ONLY", opts) == "SELF_ONLY"
    for bad in (None, "FOLLOWER_OF_CREATOR"):
        with pytest.raises(publish.PublishError, match="explicit choice"):
            publish.check_privacy(bad, opts)


def test_to_is_required_and_understands_aliases():
    assert publish.parse_to("fb,tiktok") == ["tiktok", "facebook"]
    assert publish.parse_to("IG") == ["instagram"]
    assert publish.parse_to("all") == list(publish.PLATFORMS)
    for bad in (None, "", "youtube"):
        with pytest.raises(publish.PublishError):
            publish.parse_to(bad)


def test_caption_limits():
    assert publish.caption_problems("tiktok", "x" * 2200, LIMITS) == []
    assert publish.caption_problems("tiktok", "x" * 2201, LIMITS)
    tags = " ".join(f"#t{i}" for i in range(31))
    assert publish.caption_problems("instagram", tags, LIMITS)
    assert publish.caption_problems("facebook", tags, LIMITS) == []


def test_at_must_be_future_local_time():
    now = dt.datetime(2026, 10, 28, 12, 0).astimezone()
    assert publish.parse_at("2026-10-29 18:00", now) > now.timestamp()
    for bad in ("2026-10-27 18:00", "tomorrow"):
        with pytest.raises(publish.PublishError):
            publish.parse_at(bad, now)


def test_srt_text_keeps_only_speech():
    srt = "1\n00:00:00,000 --> 00:00:01,000\nHello there\n\n2\n00:00:01,000 --> 00:00:02,000\nworld\n"
    assert publish.srt_text(srt) == "Hello there world"


# -- the run, with fake platforms ---------------------------------------------


@pytest.fixture
def session(tmp_path, monkeypatch):
    edit = tmp_path / "vid" / "edit"
    edit.mkdir(parents=True)
    video = edit / "final.mp4"
    video.write_bytes(b"\0" * 1000)
    monkeypatch.setattr(publish, "probe", lambda v: {"duration": 60.0, "width": 1080,
                                                     "height": 1920, "size": 1000})
    monkeypatch.setattr(publish, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(publish, "CAPTION_EDITS", tmp_path / "state" / "caption_edits.jsonl")
    monkeypatch.setattr(publish, "QUEUE_DIR", tmp_path / "state" / "queue")
    posted = []

    def fake(name, fail_pre=False, fail_post=False):
        def pre(job, media, limits):
            if fail_pre:
                raise publish.PublishError("not logged in")
            return {"account": name, "detail": "x"}

        def post(job, ctx, caption):
            if fail_post:
                raise publish.PublishError("boom")
            posted.append((name, caption))
            return {"id": "1", "url": f"https://{name}/1"}
        return pre, post

    funcs = {p: fake(p) for p in publish.PLATFORMS}
    monkeypatch.setattr(publish, "PLATFORM_FUNCS", funcs)
    return {"video": video, "edit": edit, "posted": posted, "funcs": funcs, "fake": fake}


def _job(video, to, **kw):
    return {"video": str(video), "to": to, "captions": {"default": "hi #news"}, **kw}


def test_one_platform_failing_does_not_stop_the_others(session):
    session["funcs"]["tiktok"] = session["fake"]("tiktok", fail_post=True)
    assert publish.run(_job(session["video"], ["tiktok", "facebook"]), yes=True) is False
    assert session["posted"] == [("facebook", "hi #news")]
    record = json.loads((session["edit"] / "publish.json").read_text())
    assert [(r["platform"], r["status"]) for r in record] == [("tiktok", "failed"), ("facebook", "ok")]


def test_a_preflight_failure_posts_nothing(session):
    session["funcs"]["instagram"] = session["fake"]("instagram", fail_pre=True)
    with pytest.raises(publish.PublishError, match="nothing was posted"):
        publish.run(_job(session["video"], ["tiktok", "instagram"]), yes=True)
    assert session["posted"] == []


def test_no_double_post_without_again(session):
    publish.run(_job(session["video"], ["facebook"]), yes=True)
    with pytest.raises(publish.PublishError, match="already posted"):
        publish.run(_job(session["video"], ["facebook"]), yes=True)
    assert publish.run(_job(session["video"], ["facebook"], again=True), yes=True)


def test_per_platform_caption_override(session):
    job = _job(session["video"], ["tiktok", "instagram"])
    job["captions"]["instagram"] = "ig only"
    publish.run(job, yes=True)
    assert session["posted"] == [("tiktok", "hi #news"), ("instagram", "ig only")]


def test_dry_run_and_no_confirmation_post_nothing(session, monkeypatch):
    publish.run(_job(session["video"], ["tiktok"]), dry_run=True)
    monkeypatch.setattr("builtins.input", lambda _: "n")
    with pytest.raises(publish.PublishError, match="cancelled"):
        publish.run(_job(session["video"], ["tiktok"]))
    assert session["posted"] == []


def test_scheduled_job_waits_then_posts(session, monkeypatch):
    monkeypatch.setattr(publish, "ensure_launchd", lambda: None)
    at = dt.datetime.now().timestamp() + 3600
    publish.run(_job(session["video"], ["facebook"], at=at), yes=True)
    assert session["posted"] == [] and publish.queued()[0]["status"] == "queued"
    publish.run_due(now=at - 1)
    assert session["posted"] == []
    assert publish.run_due(now=at + 1) == 0
    assert session["posted"] == [("facebook", "hi #news")]
    assert publish.queued()[0]["status"] == "done"


def test_suggestion_learns_from_alis_edits(session):
    (session["edit"] / "captions").mkdir()
    (session["edit"] / "captions" / "final.en.srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nSerbia students won\n")
    prompts = []
    gemini_client.register_mock(lambda req: prompts.append(req["prompt"]) or "DRAFT #a")
    try:
        llm = gemini_client.LLM()
        out = publish.suggest_caption(session["video"], llm=llm)
        assert out.read_text().strip() == "DRAFT #a"
        assert "Serbia students won" in prompts[0]
        out.write_text("Ali's version #zeta")
        publish.run(_job(session["video"], ["tiktok"],
                         captions={"default": out.read_text()}), yes=True)
        publish.suggest_caption(session["video"], llm=llm)
        assert "--- your draft:\nDRAFT #a\n--- Ali's final:\nAli's version #zeta" in prompts[1]
    finally:
        gemini_client.register_mock(None)


def test_fit_kbps_leaves_headroom():
    kbps = publish.fit_kbps(300 * MB, 172.0)
    assert (kbps + 192) * 1000 / 8 * 172.0 < 300 * MB
    assert kbps > 10_000          # a 3 min reel keeps a generous bitrate
