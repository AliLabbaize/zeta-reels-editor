"""Publish a finished video to TikTok, Instagram Reels and Facebook Reels.

    zeta publish --login tiktok | meta
    zeta publish final.mp4 --suggest-caption          # writes edit/caption.txt to edit
    zeta publish final.mp4 --to tiktok,ig,fb --privacy PUBLIC_TO_EVERYONE
    zeta publish final.mp4 --to all --privacy ... --at "2026-10-29 18:00"

Spec: docs/PUBLISH_SPEC.md. The order is always the same: preflight every chosen
platform, one confirmation, then post one platform at a time. A preflight
failure stops everything before anything exists; a failure while posting is
reported and the next platform still runs.

Uploads go straight from the Mac, nothing is hosted:
* TikTok: Content Posting API, chunked FILE_UPLOAD. Until the app passes
  TikTok's audit every direct post is forced private; `--draft` sends to the
  inbox instead. TikTok's sharing guidelines (what the audit checks) want the
  creator to see the account, choose the privacy level with no default, and
  consent before upload. The confirmation screen is exactly that.
* Instagram and Facebook: one Meta login, the Page token covers both. Binary
  goes to rupload.facebook.com.

App keys live in `.env`; user tokens in ~/.config/zeta/, mode 600, never in the
repo or a footage folder.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import http.server
import json
import os
import plistlib
import re
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path
from typing import Callable, Sequence

try:
    from . import config as configs
    from .env import load_env
    from .paths import REPO_ROOT
except ImportError:  # running as `python helpers/publish.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from helpers import config as configs
    from helpers.env import load_env
    from helpers.paths import REPO_ROOT

PLATFORMS = ("tiktok", "instagram", "facebook")
ALIASES = {"tt": "tiktok", "ig": "instagram", "fb": "facebook"}
CAPTION_FLAGS = {"tiktok": "caption_tiktok", "instagram": "caption_ig",
                 "facebook": "caption_fb"}

STATE_DIR = Path.home() / ".config" / "zeta"
QUEUE_DIR = STATE_DIR / "queue"
CAPTION_EDITS = STATE_DIR / "caption_edits.jsonl"
LAUNCHD_LABEL = "com.zeta.publish"

TIKTOK_API = "https://open.tiktokapis.com/v2"
TIKTOK_SCOPES = "user.info.basic,video.publish,video.upload"
# Both must be registered verbatim: TikTok Login Kit -> Desktop, and Meta's
# Valid OAuth Redirect URIs.
CALLBACK_PORT = 3455
TIKTOK_REDIRECT = f"http://127.0.0.1:{CALLBACK_PORT}/callback/"
META_REDIRECT = f"http://localhost:{CALLBACK_PORT}/callback/"
META_SCOPES = ("pages_show_list,pages_read_engagement,pages_manage_posts,"
               "business_management,instagram_basic,instagram_content_publish")

PRIVACY_LEVELS = ("PUBLIC_TO_EVERYONE", "MUTUAL_FOLLOW_FRIENDS",
                  "FOLLOWER_OF_CREATOR", "SELF_ONLY")
VIDEO_TYPES = {".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm"}
MB = 1024 * 1024


class PublishError(RuntimeError):
    pass


def _cfg() -> dict:
    return configs.load("publish")


# -- pure pieces (tested) -----------------------------------------------------


def parse_to(value: str | None) -> list[str]:
    """`--to tiktok,ig` -> ['tiktok', 'instagram']. Required: no default, so a
    typo never posts everywhere."""
    if not value:
        raise PublishError("say where: --to tiktok,ig,fb or --to all")
    names = [ALIASES.get(v.strip().lower(), v.strip().lower()) for v in value.split(",") if v.strip()]
    if names == ["all"]:
        return list(PLATFORMS)
    bad = [n for n in names if n not in PLATFORMS]
    if bad:
        raise PublishError(f"unknown platform {', '.join(bad)} (tiktok, ig, fb, all)")
    return [p for p in PLATFORMS if p in names]


def caption_problems(platform: str, text: str, limits: dict) -> list[str]:
    out = []
    if len(text) > limits["caption_chars"]:
        out.append(f"caption is {len(text)} characters, max {limits['caption_chars']}")
    if platform == "instagram":
        tags, mentions = len(re.findall(r"#\w", text)), len(re.findall(r"@\w", text))
        if tags > limits["ig_hashtags"]:
            out.append(f"{tags} hashtags, Instagram allows {limits['ig_hashtags']}")
        if mentions > limits["ig_mentions"]:
            out.append(f"{mentions} @mentions, Instagram allows {limits['ig_mentions']}")
    return out


def pkce_pair() -> tuple[str, str]:
    """(verifier, challenge). TikTok wants the challenge HEX-encoded, not the
    base64url of RFC 7636; the standard encoding is rejected."""
    verifier = secrets.token_urlsafe(64)[:96]
    return verifier, hashlib.sha256(verifier.encode()).hexdigest()


def chunk_plan(size: int, chunk: int = 32 * MB) -> tuple[int, int]:
    """TikTok (chunk_size, total_chunk_count): up to 64 MB in one piece, above
    that fixed chunks with the remainder folded into the last one."""
    if size <= 0:
        raise PublishError("empty video file")
    if size > 4096 * MB:
        raise PublishError("TikTok rejects files over 4 GB")
    if size <= 64 * MB:
        return size, 1
    return chunk, size // chunk


def chunk_ranges(size: int, chunk_size: int, count: int) -> list[tuple[int, int]]:
    """Inclusive byte ranges; the last absorbs the remainder."""
    return [(i * chunk_size, size - 1 if i == count - 1 else (i + 1) * chunk_size - 1)
            for i in range(count)]


def check_privacy(privacy: str | None, options: Sequence[str]) -> str:
    if privacy not in options:
        raise PublishError(f"choose --privacy from {', '.join(options)} "
                           "(TikTok requires an explicit choice, no default)")
    return privacy


def already_posted(record: list[dict], platform: str, video: Path) -> bool:
    size = video.stat().st_size
    return any(r.get("platform") == platform and r.get("status") == "ok"
               and r.get("video") == video.name and r.get("size") == size for r in record)


def parse_at(value: str, now: dt.datetime | None = None) -> float:
    """'2026-10-29 18:00' in local time -> epoch seconds, must be in the future."""
    try:
        when = dt.datetime.fromisoformat(value.strip()).astimezone()
    except ValueError:
        raise PublishError(f"--at {value!r}: use 'YYYY-MM-DD HH:MM' (local time)")
    if when <= (now or dt.datetime.now().astimezone()):
        raise PublishError(f"--at {value}: that is in the past")
    return when.timestamp()


def fit_kbps(max_bytes: int, duration: float, audio_kbps: int = 192) -> int:
    """Video bitrate that lands a file 5% under max_bytes (container overhead)."""
    return int(max_bytes * 8 * 0.95 / duration / 1000) - audio_kbps


def fit_size(video: Path, max_bytes: int, duration: float, out: Path) -> Path:
    """Two-pass x264 re-encode under max_bytes, audio copied untouched.

    Not lossless, but invisible: at 300 MB a 3 min reel still gets ~13 Mbps,
    and Instagram re-encodes every Reel to a few Mbps anyway. Two-pass so the
    size lands on target instead of being guessed. Cached next to the video.
    """
    if out.exists() and out.stat().st_mtime >= video.stat().st_mtime \
            and out.stat().st_size <= max_bytes:
        return out
    kbps = fit_kbps(max_bytes, duration)
    log = out.with_suffix(".passlog")
    base = ["ffmpeg", "-y", "-v", "error", "-i", str(video), "-c:v", "libx264",
            "-preset", "medium", "-b:v", f"{kbps}k", "-pix_fmt", "yuv420p", "-passlogfile", str(log)]
    try:
        subprocess.run(base + ["-pass", "1", "-an", "-f", "mp4", os.devnull], check=True)
        subprocess.run(base + ["-pass", "2", "-c:a", "copy", "-movflags", "+faststart",
                               str(out)], check=True)
    except subprocess.CalledProcessError as exc:
        raise PublishError(f"re-encode for size failed: {exc}")
    finally:
        for f in out.parent.glob(log.name + "*"):
            f.unlink()
    if out.stat().st_size > max_bytes:
        raise PublishError(f"re-encode still {out.stat().st_size / MB:.0f} MB")
    return out


def srt_text(srt: str) -> str:
    """Just the spoken lines of an .srt: no cue numbers, no timestamps."""
    lines = [l.strip() for l in srt.splitlines()]
    return " ".join(l for l in lines if l and not l.isdigit() and "-->" not in l)


# -- small plumbing -----------------------------------------------------------


def _edit_dir(video: Path) -> Path:
    """Hard rule 11: the session's edit/ dir. final.mp4 usually already sits in it."""
    return video.parent if video.parent.name == "edit" else video.parent / "edit"


def _load_json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _save_secret(name: str, data: dict) -> None:
    path = STATE_DIR / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    path.chmod(0o600)


def _load_secret(name: str) -> dict:
    path = STATE_DIR / f"{name}.json"
    if not path.exists():
        raise PublishError(f"not logged in to {name}: run `zeta publish --login {name}`")
    return json.loads(path.read_text())


def _env(*keys: str) -> list[str]:
    load_env()
    vals = [os.environ.get(k) for k in keys]
    if not all(vals):
        raise PublishError(f"set {' and '.join(keys)} in .env")
    return vals


def probe(video: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "format=duration:stream=width,height", "-of", "json", str(video)],
        capture_output=True, text=True)
    info = json.loads(out.stdout or "{}")
    stream = (info.get("streams") or [{}])[0]
    return {"duration": float(info.get("format", {}).get("duration", 0)),
            "width": stream.get("width", 0), "height": stream.get("height", 0),
            "size": video.stat().st_size}


def _request(url: str, *, data=None, headers: dict | None = None,
             method: str = "POST") -> tuple[int, dict]:
    """`data` may be bytes or an open file (streamed; Content-Length required)."""
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            body = resp.read()
            return resp.status, (json.loads(body) if body.strip() else {})
    except urllib.error.HTTPError as exc:
        body = exc.read()
        try:
            return exc.code, json.loads(body)
        except ValueError:
            return exc.code, {"error": {"code": str(exc.code),
                                        "message": body[:300].decode(errors="replace")}}


def _poll(check: Callable[[], bool], *, every: float, timeout: float, what: str) -> None:
    end = time.time() + timeout
    while time.time() < end:
        if check():
            return
        time.sleep(every)
    raise PublishError(f"{what}: still processing after {timeout / 60:.0f} min")


def _browser_code(url: str, state: str) -> dict:
    """Open the consent page; a one-shot 127.0.0.1 server catches the redirect."""
    got: dict = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            got.update({k: v[0] for k, v in q.items()})
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"zeta: connected, you can close this tab.")

        def log_message(self, *a):
            pass

    with http.server.HTTPServer(("127.0.0.1", CALLBACK_PORT), Handler) as srv:
        print(f"Opening the login page. If no browser opens, visit:\n{url}")
        webbrowser.open(url)
        srv.timeout = 10
        deadline = time.time() + 300
        while not got and time.time() < deadline:
            srv.handle_request()
    if not got:
        raise PublishError("no answer from the login page within 5 minutes")
    # The state check is what stops another page from feeding us its own code.
    if got.get("state") != state:
        raise PublishError("login state mismatch, try again")
    if "code" not in got:
        raise PublishError(f"login refused: {got.get('error_description') or got.get('error')}")
    return got


# -- TikTok -------------------------------------------------------------------


def _tiktok_token_request(fields: dict) -> dict:
    key, secret = _env("TIKTOK_CLIENT_KEY", "TIKTOK_CLIENT_SECRET")
    _, body = _request(f"{TIKTOK_API}/oauth/token/", data=urllib.parse.urlencode(
        {"client_key": key, "client_secret": secret, **fields}).encode(),
        headers={"Content-Type": "application/x-www-form-urlencoded"})
    if "access_token" not in body:
        raise PublishError(f"TikTok token request failed: {body.get('error')} "
                           f"{body.get('error_description', '')}".strip())
    body["expires_at"] = time.time() + int(body.get("expires_in", 0)) - 60
    _save_secret("tiktok", body)
    return body


def login_tiktok() -> None:
    key, _ = _env("TIKTOK_CLIENT_KEY", "TIKTOK_CLIENT_SECRET")
    verifier, challenge = pkce_pair()
    state = secrets.token_urlsafe(16)
    got = _browser_code("https://www.tiktok.com/v2/auth/authorize/?" + urllib.parse.urlencode({
        "client_key": key, "scope": TIKTOK_SCOPES, "response_type": "code",
        "redirect_uri": TIKTOK_REDIRECT, "state": state,
        "code_challenge": challenge, "code_challenge_method": "S256"}), state)
    tok = _tiktok_token_request({"code": got["code"], "grant_type": "authorization_code",
                                 "redirect_uri": TIKTOK_REDIRECT, "code_verifier": verifier})
    print(f"TikTok connected. Scopes: {tok.get('scope')}")


def _tiktok_access() -> str:
    tok = _load_secret("tiktok")
    if time.time() >= tok.get("expires_at", 0):
        tok = _tiktok_token_request({"grant_type": "refresh_token",
                                     "refresh_token": tok["refresh_token"]})
    return tok["access_token"]


def _tiktok(path: str, token: str, payload: dict) -> dict:
    _, body = _request(f"{TIKTOK_API}{path}", data=json.dumps(payload).encode(), headers={
        "Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=UTF-8"})
    err = body.get("error", {})
    if err.get("code", "ok") != "ok":
        raise PublishError(f"TikTok {path} -> {err.get('code')}: {err.get('message')} "
                           f"(log_id {err.get('log_id', '?')})")
    return body.get("data", {})


def preflight_tiktok(job: dict, media: dict, limits: dict) -> dict:
    token = _tiktok_access()
    info = _tiktok("/post/publish/creator_info/query/", token, {})
    cap = info.get("max_video_post_duration_sec", 0)
    if cap and media["duration"] > cap:
        raise PublishError(f"video is {media['duration']:.0f}s, this account can post {cap}s")
    if job.get("draft"):
        detail = "draft to inbox"
    else:
        detail = check_privacy(job.get("privacy"), info.get("privacy_level_options", []))
    return {"token": token, "info": info, "detail": detail,
            "account": f"@{info.get('creator_username')} ({info.get('creator_nickname')})"}


def post_tiktok(job: dict, ctx: dict, caption: str) -> dict:
    video, token, info = Path(job["video"]), ctx["token"], ctx["info"]
    size = video.stat().st_size
    chunk_size, count = chunk_plan(size)
    source = {"source": "FILE_UPLOAD", "video_size": size,
              "chunk_size": chunk_size, "total_chunk_count": count}
    if job.get("draft"):
        data = _tiktok("/post/publish/inbox/video/init/", token, {"source_info": source})
    else:
        post = {"title": caption, "privacy_level": job["privacy"],
                # A setting the creator turned off in TikTok cannot be re-enabled here.
                "disable_comment": job.get("no_comments") or info.get("comment_disabled", False),
                "disable_duet": job.get("no_duet") or info.get("duet_disabled", False),
                "disable_stitch": job.get("no_stitch") or info.get("stitch_disabled", False)}
        if job.get("cover_ms") is not None:
            post["video_cover_timestamp_ms"] = job["cover_ms"]
        data = _tiktok("/post/publish/video/init/", token, {"post_info": post, "source_info": source})

    ctype = VIDEO_TYPES[video.suffix.lower()]
    with video.open("rb") as fh:
        for i, (first, last) in enumerate(chunk_ranges(size, chunk_size, count), 1):
            chunk = fh.read(last - first + 1)
            status, body = _request(data["upload_url"], data=chunk, method="PUT", headers={
                "Content-Type": ctype, "Content-Length": str(len(chunk)),
                "Content-Range": f"bytes {first}-{last}/{size}"})
            if status not in (200, 201, 206):
                raise PublishError(f"chunk {i}/{count} upload failed: HTTP {status} {body}")
            print(f"  tiktok: uploaded {i}/{count}")

    done = "SEND_TO_USER_INBOX" if job.get("draft") else "PUBLISH_COMPLETE"
    st: dict = {}

    def check() -> bool:
        st.update(_tiktok("/post/publish/status/fetch/", token, {"publish_id": data["publish_id"]}))
        if st.get("status") == "FAILED":
            raise PublishError(f"TikTok rejected the video: {st.get('fail_reason')}")
        return st.get("status") == done

    _poll(check, every=5, timeout=600, what="TikTok")
    # TikTok's own field name, misspelling included. Empty while posts are private.
    ids = st.get("publicaly_available_post_id") or []
    if job.get("draft"):
        return {"id": data["publish_id"], "url": "", "note": "sent to your TikTok inbox (can take ~10 min to show up as a notification); add the caption in the app and post"}
    if not ids:
        return {"id": data["publish_id"], "url": "",
                "note": "posted private (app not audited by TikTok yet)"}
    return {"id": str(ids[0]),
            "url": f"https://www.tiktok.com/@{info.get('creator_username')}/video/{ids[0]}"}


# -- Meta (Facebook + Instagram) ---------------------------------------------


def _graph(method: str, path: str, token: str, host: str = "graph.facebook.com",
           **params) -> dict:
    url = f"https://{host}/{_cfg()['graph_version']}/{path}"
    if token:   # the oauth/access_token exchange authenticates with the app secret instead
        params["access_token"] = token
    encoded = urllib.parse.urlencode(params)
    if method == "GET":
        _, body = _request(f"{url}?{encoded}", method="GET")
    else:
        _, body = _request(url, data=encoded.encode(),
                           headers={"Content-Type": "application/x-www-form-urlencoded"})
    if "error" in body:
        err = body["error"]
        raise PublishError(f"Meta {path.split('?')[0]} -> {err.get('code')}/"
                           f"{err.get('error_subcode', '-')}: {err.get('message')} "
                           f"(fbtrace {err.get('fbtrace_id', '?')})")
    return body


def _rupload(url: str, token: str, video: Path) -> None:
    size = video.stat().st_size
    with video.open("rb") as fh:
        status, body = _request(url, data=fh, headers={
            "Authorization": f"OAuth {token}", "offset": "0", "file_size": str(size),
            "Content-Length": str(size), "Content-Type": "application/octet-stream"})
    if status != 200 or not body.get("success"):
        raise PublishError(f"upload failed: HTTP {status} {body.get('debug_info') or body.get('error')}")


def login_meta(page: str | None = None) -> None:
    app_id, secret = _env("META_APP_ID", "META_APP_SECRET")
    state = secrets.token_urlsafe(16)
    ver = _cfg()["graph_version"]
    got = _browser_code(f"https://www.facebook.com/{ver}/dialog/oauth?" + urllib.parse.urlencode({
        "client_id": app_id, "redirect_uri": META_REDIRECT, "state": state,
        "response_type": "code",
        # Facebook Login for Business apps may require a login configuration id
        # instead of a scope list; use it when one is set.
        **({"config_id": os.environ["META_CONFIG_ID"]} if os.environ.get("META_CONFIG_ID")
           else {"scope": META_SCOPES})}), state)
    short = _graph("GET", "oauth/access_token", "", client_id=app_id, client_secret=secret,
                   redirect_uri=META_REDIRECT, code=got["code"])["access_token"]
    # A Page token minted from a long-lived user token does not expire.
    exchanged = _graph("GET", "oauth/access_token", "", grant_type="fb_exchange_token",
                       client_id=app_id, client_secret=secret, fb_exchange_token=short)
    long = exchanged["access_token"]
    pages = _graph("GET", "me/accounts", long, fields=(
        "id,name,access_token,instagram_business_account{id,username}"))["data"]
    if page:
        pages = [p for p in pages if page.lower() in (p["name"].lower(), p["id"])]
    if not pages:
        raise PublishError("no matching Facebook Page; you must be admin of the Page")
    if len(pages) > 1:
        raise PublishError("several Pages, pick one with --page: "
                           + ", ".join(p["name"] for p in pages))
    p = pages[0]
    ig = p.get("instagram_business_account") or {}
    _save_secret("meta", {"page_id": p["id"], "page_name": p["name"],
                          "page_token": p["access_token"],
                          # Only long Facebook videos need it (upload sessions are
                          # per user); it lasts ~60 days, the Page token does not expire.
                          "user_token": long, "user_token_expires_at":
                              time.time() + int(exchanged.get("expires_in", 60 * 86400)),
                          "ig_id": ig.get("id"), "ig_username": ig.get("username")})
    print(f"Facebook Page: {p['name']}\nInstagram: "
          + (f"@{ig['username']}" if ig else "none linked to this Page (link it in Page settings)"))


def _duration_problem(media: dict, limits: dict, max_key: str, name: str) -> None:
    if media["duration"] < limits["min_duration_s"]:
        raise PublishError(f"{name} needs at least {limits['min_duration_s']}s")
    if media["duration"] > limits[max_key]:
        raise PublishError(f"video is {media['duration']:.0f}s, {name} Reels take at most "
                           f"{limits[max_key]}s via the API (configs/publish.yaml)")


def preflight_instagram(job: dict, media: dict, limits: dict) -> dict:
    meta = _load_secret("meta")
    if not meta.get("ig_id"):
        raise PublishError("no Instagram account linked to the Facebook Page")
    _duration_problem(media, limits, "instagram_max_s", "Instagram")
    ctx = {}
    if media["size"] > limits["instagram_max_mb"] * MB:
        print(f"instagram: {media['size'] / MB:.0f} MB is over {limits['instagram_max_mb']} MB, "
              "re-encoding a copy to fit (a few minutes)...")
        video = Path(job["video"])
        ctx["file"] = str(fit_size(video, limits["instagram_max_mb"] * MB, media["duration"],
                                   _edit_dir(video) / f"{video.stem}.instagram.mp4"))
    quota = _graph("GET", f"{meta['ig_id']}/content_publishing_limit", meta["page_token"],
                   fields="quota_usage,config").get("data", [{}])[0]
    if quota.get("quota_usage", 0) >= quota.get("config", {}).get("quota_total", 1e9):
        raise PublishError("Instagram daily API post limit reached")
    detail = "Reel only" if job.get("no_feed") else "Reel + feed"
    if ctx:
        detail += f", {Path(ctx['file']).stat().st_size / MB:.0f} MB copy"
    return {**ctx, "meta": meta, "account": f"@{meta['ig_username']}", "detail": detail}


def post_instagram(job: dict, ctx: dict, caption: str) -> dict:
    meta, video = ctx["meta"], Path(ctx.get("file") or job["video"])
    token, ig = meta["page_token"], meta["ig_id"]
    params = {"media_type": "REELS", "upload_type": "resumable", "caption": caption}
    if job.get("cover_ms") is not None:
        params["thumb_offset"] = job["cover_ms"]
    if job.get("no_feed"):
        params["share_to_feed"] = "false"   # Reels go to the feed by default
    container = _graph("POST", f"{ig}/media", token, **params)
    _rupload(container["uri"], token, video)
    print("  instagram: uploaded, processing")

    def check() -> bool:
        st = _graph("GET", container["id"], token, fields="status_code,status")
        if st.get("status_code") in ("ERROR", "EXPIRED"):
            raise PublishError(f"Instagram rejected the video: {st.get('status')}")
        return st.get("status_code") == "FINISHED"

    # Meta asks for at most one status call a minute; 10 s is fine for one container.
    _poll(check, every=10, timeout=600, what="Instagram")
    media_id = _graph("POST", f"{ig}/media_publish", token, creation_id=container["id"])["id"]
    link = _graph("GET", media_id, token, fields="permalink").get("permalink", "")
    return {"id": media_id, "url": link}


def preflight_facebook(job: dict, media: dict, limits: dict) -> dict:
    """Up to 90 s: the Reels API. Longer: a regular Page video, which Facebook
    has shown as a Reel since June 2025 ("all videos are reels")."""
    meta = _load_secret("meta")
    if media["duration"] < limits["min_duration_s"]:
        raise PublishError(f"Facebook needs at least {limits['min_duration_s']}s")
    long = media["duration"] > limits["facebook_max_s"]
    if long and time.time() > meta.get("user_token_expires_at", 0) - 86400:
        raise PublishError("long videos need a fresh Meta login: `zeta publish --login meta`")
    detail = "draft" if job.get("draft") else "published"
    return {"meta": meta, "long": long, "account": f"{meta['page_name']} (Page)",
            "detail": detail + (", as video" if long else "")}


def _post_facebook_long(job: dict, meta: dict, video: Path, caption: str) -> str:
    """Resumable Upload API: session and bytes with the user token, publish with
    the Page token. Returns the video id."""
    app_id, _ = _env("META_APP_ID", "META_APP_SECRET")
    user, size = meta["user_token"], video.stat().st_size
    session = _graph("POST", f"{app_id}/uploads", user, file_name=video.name,
                     file_length=size, file_type="video/mp4")["id"]
    with video.open("rb") as fh:
        status, body = _request(
            f"https://graph.facebook.com/{_cfg()['graph_version']}/{session}", data=fh,
            headers={"Authorization": f"OAuth {user}", "file_offset": "0",
                     "Content-Length": str(size)})
    if "h" not in body:
        raise PublishError(f"upload failed: HTTP {status} {body.get('error')}")
    params = {"fbuploader_video_file_chunk": body["h"], "description": caption}
    if job.get("draft"):
        params["published"] = "false"
    return _graph("POST", f"{meta['page_id']}/videos", meta["page_token"],
                  host="graph-video.facebook.com", **params)["id"]


def post_facebook(job: dict, ctx: dict, caption: str) -> dict:
    meta, video = ctx["meta"], Path(job["video"])
    token, page = meta["page_token"], meta["page_id"]
    if ctx.get("long"):
        vid = _post_facebook_long(job, meta, video, caption)
    else:
        start = _graph("POST", f"{page}/video_reels", token, upload_phase="start")
        _rupload(start["upload_url"], token, video)
        vid = start["video_id"]
        _graph("POST", f"{page}/video_reels", token, upload_phase="finish", video_id=vid,
               video_state="DRAFT" if job.get("draft") else "PUBLISHED", description=caption)
    print("  facebook: uploaded, processing")

    def check() -> bool:
        st = _graph("GET", vid, token, fields="status").get("status", {})
        if st.get("video_status") in ("error", "expired", "upload_failed"):
            raise PublishError(f"Facebook rejected the video: {json.dumps(st)[:300]}")
        pub = st.get("publishing_phase", {})
        return pub.get("publish_status") in ("published", "draft") or \
            pub.get("status") in ("complete", "completed")

    _poll(check, every=10, timeout=600, what="Facebook")
    return {"id": vid, "url": f"https://www.facebook.com/reel/{vid}"}


# Tests swap entries here instead of faking HTTP for every platform.
PLATFORM_FUNCS: dict[str, tuple[Callable, Callable]] = {
    "tiktok": (preflight_tiktok, post_tiktok),
    "instagram": (preflight_instagram, post_instagram),
    "facebook": (preflight_facebook, post_facebook),
}


# -- captions: suggest, and learn from Ali's edits ----------------------------


def caption_for(job: dict, platform: str) -> str:
    caps = job.get("captions", {})
    return caps.get(platform) or caps.get("default", "")


def _caption_examples(n: int) -> list[dict]:
    if not CAPTION_EDITS.exists():
        return []
    rows = [json.loads(l) for l in CAPTION_EDITS.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows[-n:]


def caption_prompt(transcript: str, examples: list[dict]) -> str:
    parts = ["Write the social media caption for a Zeta news video (TikTok, Instagram "
             "Reels, Facebook Reels; one caption for all three). Max 2200 characters, "
             "max 30 hashtags. Return only the caption text."]
    if examples:
        parts.append("Ali edits every caption you suggest. Learn from his edits: match the "
                     "language, length, tone, hook, emoji and hashtag habits of his FINAL "
                     "versions, and avoid what he keeps removing.")
        for ex in examples:
            parts.append(f"--- your draft:\n{ex['suggested']}\n--- Ali's final:\n{ex['final']}")
    else:
        parts.append("Write in English: a one-line hook, two or three short lines on the "
                     "story, then a few relevant hashtags.")
    parts.append(f"--- what is said in the video:\n{transcript[:12000]}")
    return "\n\n".join(parts)


def suggest_caption(video: Path, llm=None) -> Path:
    edit = _edit_dir(video)
    en_srt, kept = edit / "captions" / "final.en.srt", edit / "kept_text.txt"
    if en_srt.exists():
        transcript = srt_text(en_srt.read_text(encoding="utf-8"))
    elif kept.exists():
        transcript = kept.read_text(encoding="utf-8")
    else:
        raise PublishError(f"no captions/final.en.srt or kept_text.txt in {edit}; "
                           "write the caption yourself with --caption")
    cfg = _cfg()["caption"]
    if llm is None:
        from helpers.gemini_client import LLM   # lazy: pulls google-genai
        llm = LLM(model=cfg["model"], cache_dir=edit / "llm_cache")
    llm.require()
    text = llm.text(caption_prompt(transcript, _caption_examples(cfg["examples"])),
                    temperature=0.4).strip()
    (edit / "caption.suggested.txt").write_text(text + "\n", encoding="utf-8")
    out = edit / "caption.txt"
    out.write_text(text + "\n", encoding="utf-8")
    return out


def remember_caption(edit: Path, final: str) -> None:
    """Pair the model's suggestion with what Ali actually posted. An unchanged
    suggestion is a signal too (he approved it), so it is kept."""
    suggested = edit / "caption.suggested.txt"
    if not suggested.exists() or not final.strip():
        return
    row = {"video": edit.parent.name, "suggested": suggested.read_text(encoding="utf-8").strip(),
           "final": final.strip(), "at": dt.datetime.now().isoformat(timespec="seconds")}
    if any(r["video"] == row["video"] and r["final"] == row["final"]
           for r in _caption_examples(10_000)):
        return
    CAPTION_EDITS.parent.mkdir(parents=True, exist_ok=True)
    with CAPTION_EDITS.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# -- the run ------------------------------------------------------------------


def preflight(job: dict) -> tuple[dict, dict]:
    """Every check for every platform; raises with ALL problems, posts nothing."""
    video = Path(job["video"])
    if not video.is_file() or video.suffix.lower() not in VIDEO_TYPES:
        raise PublishError(f"{video}: need an existing .mp4, .mov or .webm")
    limits = _cfg()["limits"]
    media = probe(video)
    record = _load_json(_edit_dir(video) / "publish.json", [])
    ctxs, problems = {}, []
    for p in job["to"]:
        if not job.get("again") and already_posted(record, p, video):
            problems.append(f"{p}: this video is already posted there (--again to repost)")
            continue
        errs = caption_problems(p, caption_for(job, p), limits)
        try:
            ctxs[p] = PLATFORM_FUNCS[p][0](job, media, limits)
        except PublishError as exc:
            errs.append(str(exc))
        problems += [f"{p}: {e}" for e in errs]
    if problems:
        raise PublishError("nothing was posted:\n  " + "\n  ".join(problems))
    return media, ctxs


def summary(job: dict, media: dict, ctxs: dict) -> str:
    video = Path(job["video"])
    m, s = divmod(round(media["duration"]), 60)
    lines = [f"Video: {video.name}  {m}:{s:02d}  {media['width']}x{media['height']}  "
             f"{media['size'] / MB:.0f} MB"]
    if job.get("at"):
        lines.append("Scheduled: " + dt.datetime.fromtimestamp(job["at"]).strftime("%Y-%m-%d %H:%M"))
    for p in job["to"]:
        cap = caption_for(job, p)
        lines.append(f"{p:<10} {ctxs[p]['account']:<28} {ctxs[p]['detail']:<20} "
                     f"caption {len(cap)} chars")
    caps = {caption_for(job, p) for p in job["to"]}
    for cap in caps:
        lines.append("Caption:\n  " + (cap.replace("\n", "\n  ") if cap else "(none)"))
    if "tiktok" in job["to"]:
        lines.append("By posting, you agree to TikTok's Music Usage Confirmation.")
    return "\n".join(lines)


def _confirm(question: str) -> bool:
    try:
        return input(f"{question} [y/N] ").strip().lower() == "y"
    except EOFError:
        return False


def post_all(job: dict, ctxs: dict) -> bool:
    """One platform at a time; a failure is recorded and the next still runs."""
    video = Path(job["video"])
    edit = _edit_dir(video)
    edit.mkdir(parents=True, exist_ok=True)
    record_path = edit / "publish.json"
    ok = True
    for p in job["to"]:
        print(f"{p}: posting...")
        row = {"platform": p, "video": video.name, "size": video.stat().st_size,
               "at": dt.datetime.now().isoformat(timespec="seconds"),
               "caption": caption_for(job, p)}
        try:
            row.update(PLATFORM_FUNCS[p][1](job, ctxs[p], caption_for(job, p)), status="ok")
            print(f"{p}: done {row.get('url') or row.get('note', '')}")
        except (PublishError, OSError) as exc:
            ok = False
            row.update(status="failed", error=str(exc))
            print(f"{p}: FAILED {exc}", file=sys.stderr)
        # Written after every platform, so a crash midway still leaves the truth.
        record_path.write_text(json.dumps(_load_json(record_path, []) + [row], indent=2,
                                          ensure_ascii=False), encoding="utf-8")
    return ok


def run(job: dict, *, yes: bool = False, dry_run: bool = False) -> bool:
    media, ctxs = preflight(job)
    print(summary(job, media, ctxs))
    if dry_run:
        print("(dry run: nothing posted)")
        return True
    verb = "Schedule" if job.get("at") else "Post to"
    if not yes and not _confirm(f"{verb} {len(job['to'])} platform(s)?"):
        raise PublishError("cancelled")
    remember_caption(_edit_dir(Path(job["video"])), job.get("captions", {}).get("default", ""))
    if job.get("at"):
        return schedule(job)
    return post_all(job, ctxs)


# -- scheduling: a queue file per post, launchd checks it every 5 min ----------


def schedule(job: dict) -> bool:
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    job_id = dt.datetime.fromtimestamp(job["at"]).strftime("%Y%m%d-%H%M-") + secrets.token_hex(2)
    (QUEUE_DIR / f"{job_id}.json").write_text(json.dumps({**job, "id": job_id, "status": "queued"},
                                                         indent=2, ensure_ascii=False))
    ensure_launchd()
    print(f"Scheduled as {job_id}. `zeta publish --queue` lists, `--cancel {job_id}` removes.\n"
          "The Mac must be awake (or wake) around that time; a sleeping Mac posts on wake.")
    return True


def ensure_launchd() -> None:
    # ponytail: polls every 5 min and needs the Mac awake; `pmset schedule wake`
    # (needs sudo) or Facebook's native scheduling if posts must land on the minute.
    plist = Path.home() / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"
    if plist.exists():
        return
    plist.parent.mkdir(parents=True, exist_ok=True)
    log = str(STATE_DIR / "publish.log")
    plist.write_bytes(plistlib.dumps({
        "Label": LAUNCHD_LABEL,
        "ProgramArguments": [sys.executable, str(REPO_ROOT / "cli.py"), "publish", "--run-due"],
        "StartInterval": 300, "RunAtLoad": True,
        "StandardOutPath": log, "StandardErrorPath": log}))
    subprocess.run(["launchctl", "bootstrap", f"gui/{os.getuid()}", str(plist)], check=False)


def queued() -> list[dict]:
    return [json.loads(f.read_text()) for f in sorted(QUEUE_DIR.glob("*.json"))] \
        if QUEUE_DIR.exists() else []


def run_due(now: float | None = None) -> int:
    failed = 0
    for job in queued():
        if job["status"] != "queued" or job["at"] > (now or time.time()):
            continue
        path = QUEUE_DIR / f"{job['id']}.json"
        path.write_text(json.dumps({**job, "status": "running"}, indent=2, ensure_ascii=False))
        print(f"[{dt.datetime.now():%Y-%m-%d %H:%M}] posting {job['id']}")
        try:
            media, ctxs = preflight({**job, "at": None})
            ok = post_all(job, ctxs)
        except PublishError as exc:
            print(exc, file=sys.stderr)
            ok = False
        failed += not ok
        path.write_text(json.dumps({**job, "status": "done" if ok else "failed"},
                                   indent=2, ensure_ascii=False))
    return 1 if failed else 0


# -- command line -------------------------------------------------------------


def _accounts() -> None:
    try:
        tok = _load_secret("tiktok")
        info = _tiktok("/post/publish/creator_info/query/", _tiktok_access(), {})
        days = (tok.get("expires_at", 0) - time.time() + int(tok.get("refresh_expires_in", 0))) / 86400
        print(f"tiktok     @{info.get('creator_username')}  (login valid ~{days:.0f} more days)")
    except PublishError as exc:
        print(f"tiktok     {exc}")
    try:
        meta = _load_secret("meta")
        print(f"facebook   {meta['page_name']} (Page)")
        print("instagram  " + (f"@{meta['ig_username']}" if meta.get("ig_id") else "not linked"))
    except PublishError as exc:
        print(f"facebook   {exc}\ninstagram  {exc}")


def build_job(args) -> dict:
    video = Path(args.video).resolve()
    to = parse_to(args.to)
    if args.draft and "instagram" in to:
        print("instagram: no drafts via the API, skipped (post it without --draft)")
        to.remove("instagram")
        if not to:
            raise PublishError("nothing left to post")
    default = args.caption
    if args.caption_file:
        default = Path(args.caption_file).read_text(encoding="utf-8").strip()
    elif default is None and (_edit_dir(video) / "caption.txt").exists():
        default = (_edit_dir(video) / "caption.txt").read_text(encoding="utf-8").strip()
        print(f"caption: from {_edit_dir(video) / 'caption.txt'}")
    captions = {"default": default or ""}
    for p, flag in CAPTION_FLAGS.items():
        if getattr(args, flag):
            captions[p] = getattr(args, flag)
    return {"video": str(video), "to": to, "captions": captions, "privacy": args.privacy,
            "draft": args.draft, "cover_ms": args.cover_ms, "no_feed": args.no_feed,
            "no_comments": args.no_comments, "no_duet": args.no_duet,
            "no_stitch": args.no_stitch, "again": args.again,
            "at": parse_at(args.at) if args.at else None}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="zeta publish", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", nargs="?")
    ap.add_argument("--to", help="tiktok,ig,fb or all (required)")
    g = ap.add_argument_group("caption")
    g.add_argument("--caption", help="same caption everywhere (default: edit/caption.txt)")
    g.add_argument("--caption-file")
    g.add_argument("--caption-tiktok")
    g.add_argument("--caption-ig")
    g.add_argument("--caption-fb")
    g.add_argument("--suggest-caption", action="store_true",
                   help="write a suggested caption to edit/caption.txt for you to edit")
    g = ap.add_argument_group("platform options")
    g.add_argument("--privacy", choices=PRIVACY_LEVELS, help="TikTok, required")
    g.add_argument("--draft", action="store_true", help="TikTok inbox / Facebook draft")
    g.add_argument("--no-comments", action="store_true", help="TikTok")
    g.add_argument("--no-duet", action="store_true", help="TikTok")
    g.add_argument("--no-stitch", action="store_true", help="TikTok")
    g.add_argument("--no-feed", action="store_true", help="Instagram: Reels tab only")
    g.add_argument("--cover-ms", type=int, help="cover frame, ms into the video")
    g = ap.add_argument_group("run")
    g.add_argument("--at", help="schedule: 'YYYY-MM-DD HH:MM' local time")
    g.add_argument("--again", action="store_true", help="allow reposting the same video")
    g.add_argument("--dry-run", action="store_true", help="checks and summary, post nothing")
    g.add_argument("-y", "--yes", action="store_true", help="skip the confirmation")
    g = ap.add_argument_group("accounts and queue")
    g.add_argument("--login", choices=["tiktok", "meta"])
    g.add_argument("--page", help="with --login meta: which Facebook Page")
    g.add_argument("--accounts", action="store_true")
    g.add_argument("--queue", action="store_true", help="list scheduled posts")
    g.add_argument("--cancel", metavar="ID", help="remove a scheduled post")
    g.add_argument("--run-due", action="store_true", help=argparse.SUPPRESS)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    try:
        if args.login == "tiktok":
            login_tiktok()
        elif args.login == "meta":
            login_meta(args.page)
        elif args.accounts:
            _accounts()
        elif args.queue:
            for j in queued():
                when = dt.datetime.fromtimestamp(j["at"]).strftime("%Y-%m-%d %H:%M")
                print(f"{j['id']}  {when}  {j['status']:<8} {','.join(j['to'])}  {Path(j['video']).name}")
        elif args.cancel:
            path = QUEUE_DIR / f"{Path(args.cancel).name}.json"
            if not path.exists():
                raise PublishError(f"no scheduled post {args.cancel}")
            path.unlink()
            print(f"cancelled {args.cancel}")
        elif args.run_due:
            return run_due()
        elif not args.video:
            ap.error("give a video (or --login, --accounts, --queue)")
        elif args.suggest_caption:
            out = suggest_caption(Path(args.video).resolve())
            print(out.read_text(encoding="utf-8"))
            print(f"Edit {out}, then publish; the caption is read from there.")
        else:
            return 0 if run(build_job(args), yes=args.yes, dry_run=args.dry_run) else 1
        return 0
    except PublishError as exc:
        print(f"zeta publish: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
