# `zeta publish` - feature spec (approved 2026-10-28)

Post a finished video to TikTok, Instagram Reels and Facebook Reels from one
command, the way Buffer does, but for Ali's own accounts only.

Status: built (`helpers/publish.py`, `configs/publish.yaml`), unit-tested
offline, never run against a real platform yet. See docs/STATUS.md.

## 1. Goals and non-goals

**Goals**
- One command, one caption, any subset of TikTok / Instagram / Facebook.
- Upload straight from the Mac, no hosting, no server.
- Nothing posts without a summary on screen and a "y" (unless `-y`).
- One platform failing never blocks or undoes the others.
- Every post recorded, so the same video is not posted twice by accident.

**Non-goals (v1)**
- Multi-user / multi-account SaaS. One account per platform.
- A web dashboard or a queue. It is a CLI.
- Analytics, comments, replies.
- YouTube Shorts, X, LinkedIn (easy to add later as another platform).

## 2. Command line

```
zeta publish <video> --to <platforms> [caption options] [platform options] [-y]
zeta publish --login tiktok | meta
zeta publish --accounts
```

### Choosing platforms: `--to`

Comma list, any of `tiktok`, `instagram` (alias `ig`), `facebook` (alias `fb`),
or `all`.

```
zeta publish final.mp4 --to tiktok            --caption "..."
zeta publish final.mp4 --to ig,fb             --caption "..."
zeta publish final.mp4 --to all               --caption "..."
```

`--to` is **required**. No default, so a typo never posts everywhere.
(Open question Q1.)

### Caption

| Option | Meaning |
| --- | --- |
| `--caption "text"` | same caption on every platform |
| `--caption-file caption.txt` | same, from a file (multi-line, emojis, Darija/Arabic safe) |
| `--caption-tiktok`, `--caption-ig`, `--caption-fb` | override for one platform; the others keep `--caption` |

Hashtags and @mentions go inside the caption text, as you would type them in the
app. No caption at all is allowed but asked about in the confirmation.

Checked before anything uploads:
- TikTok: max 2200 characters.
- Instagram: max 2200 characters, max 30 hashtags, max 20 @mentions.
- Facebook: max 2200 characters (kept equal to IG so one caption fits all).

### Platform options

| Option | Platform | Meaning |
| --- | --- | --- |
| `--privacy LEVEL` | TikTok | required when posting to TikTok (TikTok rule: no default). `PUBLIC_TO_EVERYONE`, `MUTUAL_FOLLOW_FRIENDS`, `FOLLOWER_OF_CREATOR`, `SELF_ONLY` |
| `--no-comments`, `--no-duet`, `--no-stitch` | TikTok | interaction toggles |
| `--draft` | TikTok, Facebook | TikTok: to the inbox, tap Post in the app. Facebook: saved as a draft Reel. Instagram has no draft via API; with `--draft` IG is skipped with a warning |
| `--cover-ms N` | all | cover frame, N ms into the video |
| `--no-feed` | Instagram | Reel only in the Reels tab, not the main grid (default: also in feed) |
| `--again` | all | allow posting a video that was already posted to that platform |
| `-y` | all | skip the confirmation |

### Accounts

- `zeta publish --login tiktok` - browser login (done).
- `zeta publish --login meta` - one Facebook login covers both the Facebook Page
  and the Instagram account linked to it. Lists the Pages you manage and asks
  which one, if more than one.
- `zeta publish --accounts` - shows what is connected, the account names, and
  when each login expires.

## 3. What happens, in order

1. **Preflight, all platforms, before any upload.** File exists and is
   mp4/mov; duration, size, aspect, codec (ffprobe) within each platform's
   limits; caption limits; account connected and token valid (refresh if
   needed); TikTok privacy chosen and allowed for this account. Any failure
   stops everything, nothing has been posted.
2. **One confirmation screen:**
   ```
   Video: final.mp4  2:41  9:16  238 MB
   TikTok     @zeta.news      PUBLIC_TO_EVERYONE   caption 412 chars
   Instagram  @zeta.news      Reel + feed          caption 412 chars
   Facebook   Zeta (Page)     published            caption 412 chars
   By posting, you agree to TikTok's Music Usage Confirmation.
   Post to 3 platforms? [y/N]
   ```
3. **Upload one platform at a time** (TikTok, then Instagram, then Facebook),
   with progress. A failure is reported and the next platform still runs.
4. **Wait for each platform to finish processing** (each has a status call),
   max 10 minutes per platform.
5. **Summary** with the post link per platform, or the error.
6. **Record** to `<videos_dir>/edit/publish.json` (hard rule 11): platform,
   time, caption, post id, link, status. This is also what `--again` checks.
   Exit code 0 only if every chosen platform succeeded.

## 4. Platform details

### TikTok (Content Posting API) - built
- Login Kit for Desktop, PKCE, redirect `http://127.0.0.1:3455/callback/`.
- Chunked FILE_UPLOAD, status polling.
- **Until TikTok audits the app, every post is private.** The audit needs a
  privacy policy URL, a terms URL and a screen recording of this command.

### Instagram Reels (Instagram Graph API via Facebook Login)
- Needs: IG professional account linked to a Facebook Page.
- Create container (`media_type=REELS`, `upload_type=resumable`), upload the
  file to `rupload.facebook.com` (no hosting needed), poll `status_code`,
  then `media_publish`.
- Limit: 100 API posts / 24 h, checked with `content_publishing_limit`.
- Max Reel length via API: unclear (one source says 90 s). **Test with a
  3 min video first**; if capped, preflight blocks with a clear message.

### Facebook Reels (Page `video_reels`)
- `upload_phase=start` -> upload to rupload -> `upload_phase=finish` with
  `video_state=PUBLISHED` (or `DRAFT`), poll status.
- Posts to the Page, not a personal profile (Meta does not allow API posting
  to personal profiles).

### Meta app
- One app at developers.facebook.com, owned by Ali, left in **development
  mode**. Ali is admin of the app, so it can post to his own Page and IG
  without Meta app review.
- Permissions: `pages_show_list`, `pages_read_engagement`,
  `pages_manage_posts`, `instagram_basic`, `instagram_content_publish`.
- Long-lived Page token (does not expire while the app has access).

## 5. Security
- App keys only in `.env` (`TIKTOK_CLIENT_KEY/SECRET`, `META_APP_ID/SECRET`),
  `.env` already gitignored.
- User tokens in `~/.config/zeta/<platform>.json`, mode 600, outside the repo
  and outside footage folders.
- Tokens never printed or logged; errors show the platform's error code and
  log id only.
- Login checks the OAuth `state`; the callback server listens on 127.0.0.1
  only and shuts down after one request.

## 6. Code shape
- `helpers/publish.py`: CLI, preflight, confirmation, record, the loop over
  platforms. Each platform is three functions: `preflight`, `upload`,
  `wait`. Split into per-platform files only if it passes ~600 lines.
- stdlib only (urllib, http.server); ffprobe for the media checks.
- Tests offline: caption limits, `--to` parsing, preflight blocking,
  one-failure-continues, `publish.json` and `--again`, chunk math (done),
  PKCE (done). HTTP replaced by a fake in tests.

## 7. Build order
1. Reshape TikTok code into this shape: `--to`, caption options,
   preflight, confirmation, `publish.json`. Test on a real private TikTok
   post.
2. Meta login + Facebook Reels. Test as `--draft` first.
3. Instagram Reels. Test the 3 min length question.
4. TikTok audit pack: privacy policy, terms, demo recording.

## 8. Decisions (Ali, 2026-10-28)
- **Q1.** `--to` is required. No default.
- **Q2.** Captions are suggested, Ali edits, the tool learns from the edits:
  `zeta publish final.mp4 --suggest-caption` writes `edit/caption.suggested.txt`
  and `edit/caption.txt`. Ali edits `caption.txt`; publish reads it when no
  `--caption` is given. On confirmation the (suggested, final) pair is appended
  to `~/.config/zeta/caption_edits.jsonl`; the next suggestion shows the model
  the last 8 pairs. Source text: `edit/captions/final.en.srt`, else
  `edit/kept_text.txt`. Model in `configs/publish.yaml`.
- **Q3.** Scheduling is in v1: `--at "YYYY-MM-DD HH:MM"` (local time). Checks
  and confirmation happen when scheduling; the job is saved to
  `~/.config/zeta/queue/`, and a launchd agent (`com.zeta.publish`, installed on
  first use) runs `zeta publish --run-due` every 5 minutes, re-checks and posts.
  The Mac must be awake; a sleeping Mac posts when it wakes. `--queue` lists,
  `--cancel ID` removes. Log: `~/.config/zeta/publish.log`.
- **Q4.** Instagram Reels also go to the main grid (`--no-feed` to opt out).
- **Q5.** Same file on every platform.

Also (Ali, same day): Facebook videos over 90 s go through the regular Page
video API (`/{page}/videos`, Resumable Upload API), which Facebook shows as a
Reel; needs the Meta login to be under ~60 days old. Instagram files over the
size limit are re-encoded (two-pass x264, audio copied) to
`edit/<name>.instagram.mp4` instead of being refused.

Also built: `--dry-run` (all checks and the summary, posts nothing).
