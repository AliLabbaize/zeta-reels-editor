"""Ali approves the screenshots before the render: a one-page local review.

`zeta edit --review-shots` stops after research, opens this page in the browser,
and waits. Each verified screenshot is shown with the moment of the script it
illustrates; Ali clicks Keep or Drop, then Done, and only kept inserts render.
Stdlib only: a localhost http.server that lives for exactly one submission.
"""

from __future__ import annotations

import argparse
import html
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Sequence
from urllib.parse import urlparse


def _mmss(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


def page(items: Sequence[dict]) -> str:
    """The review page. `items`: slot_id, time, said, claim, url."""
    cards = []
    for n, it in enumerate(items, start=1):
        sid = html.escape(it["slot_id"])
        cards.append(f"""
<div class="card" id="{sid}">
  <div class="meta"><b>#{n}</b> at <b>{_mmss(it['time'])}</b> &middot;
    <span class="src">{html.escape(urlparse(it.get('url') or '').netloc)}</span></div>
  <div class="said" dir="auto">&ldquo;{html.escape(it.get('said') or '')}&rdquo;</div>
  <div class="claim">{html.escape(it.get('claim') or '')}</div>
  <img src="/img/{sid}" alt="">
  <div class="btns">
    <button class="keep on" onclick="pick('{sid}',true)">&#10003; Keep</button>
    <button class="drop" onclick="pick('{sid}',false)">&#10007; Drop</button>
  </div>
  <textarea placeholder="Feedback (optional): what should this show instead?"></textarea>
  <input type="url" placeholder="Or paste a link to use instead">
</div>""")
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Screenshot review</title><style>
body{{font:16px -apple-system,system-ui,sans-serif;background:#111;color:#eee;margin:0;padding:16px}}
h1{{font-size:20px;margin:0 0 4px}} p.hint{{color:#aaa;margin:0 0 16px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:16px}}
.card{{background:#1c1c1e;border-radius:12px;padding:12px;border:3px solid #2e7d32}}
.card.dropped{{border-color:#c62828;opacity:.55}}
.meta{{color:#bbb;font-size:14px}} .said{{margin:6px 0;font-size:17px}}
.claim{{color:#9ecbff;font-size:14px;margin-bottom:8px}}
img{{width:100%;border-radius:6px;background:#fff}}
.btns{{display:flex;gap:8px;margin-top:8px}} button{{flex:1;padding:10px;border:0;border-radius:8px;
font-size:16px;background:#333;color:#eee;cursor:pointer}}
button.keep.on{{background:#2e7d32}} button.drop.on{{background:#c62828}}
textarea,input{{width:100%;box-sizing:border-box;margin-top:8px;padding:8px;border-radius:8px;
border:1px solid #444;background:#111;color:#eee;font:15px -apple-system,system-ui,sans-serif}}
textarea{{height:56px}}
#done{{position:sticky;bottom:0;width:100%;margin-top:16px;padding:16px;font-size:18px;background:#0a84ff}}
</style></head><body>
<h1>{len(items)} screenshot(s) for this video</h1>
<p class="hint">Everything is kept unless you click Drop. Paste a link to swap a screenshot;
write feedback and the video waits so it can be redone. Click Done when finished.</p>
<div class="grid">{''.join(cards)}</div>
<button id="done" onclick="done()">Done &mdash; render the video</button>
<script>
const keep = {{}};
{"".join(f"keep['{html.escape(it['slot_id'])}']=true;" for it in items)}
function pick(id, v) {{
  keep[id] = v; const c = document.getElementById(id);
  c.classList.toggle('dropped', !v);
  c.querySelector('.keep').classList.toggle('on', v);
  c.querySelector('.drop').classList.toggle('on', !v);
}}
function done() {{
  const out = {{}};
  for (const id in keep) {{
    const c = document.getElementById(id);
    out[id] = {{keep: keep[id], note: c.querySelector('textarea').value.trim(),
               url: c.querySelector('input').value.trim()}};
  }}
  fetch('/submit', {{method: 'POST', body: JSON.stringify(out)}})
    .then(() => {{ document.body.innerHTML = '<h1>Thanks. Rendering now; you can close this tab.</h1>'; }});
}}
</script></body></html>"""


def review(items: Sequence[dict], *, open_browser: bool = True, port: int = 0) -> dict[str, dict]:
    """Serve the page, block until Ali clicks Done.

    Returns {slot_id: {"keep": bool, "note": str, "url": str}}.
    """
    body = page(items).encode("utf-8")
    images = {it["slot_id"]: Path(it["image"]) for it in items}
    result: dict[str, dict] = {}
    finished = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, code: int, data: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.startswith("/img/"):
                img = images.get(self.path[5:])
                if img and img.exists():
                    return self._send(200, img.read_bytes(), "image/png")
                return self._send(404, b"", "text/plain")
            self._send(200, body, "text/html; charset=utf-8")

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                got = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                got = {}
            for k, v in got.items():
                if k in images:
                    v = v if isinstance(v, dict) else {"keep": bool(v)}
                    result[k] = {"keep": bool(v.get("keep", True)),
                                 "note": str(v.get("note") or ""), "url": str(v.get("url") or "")}
            self._send(200, b"ok", "text/plain")
            finished.set()

    server = HTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"zeta: review the screenshots at {url}", flush=True)
    if open_browser:
        webbrowser.open(url)
    finished.wait()
    server.shutdown()
    # Anything not answered stays in: the page defaults every card to Keep.
    return {sid: result.get(sid, {"keep": True, "note": "", "url": ""}) for sid in images}


def main() -> None:
    ap = argparse.ArgumentParser(description="Review screenshot inserts in the browser")
    ap.add_argument("items_json", help='JSON list of {slot_id, time, said, claim, url, image}')
    args = ap.parse_args()
    print(json.dumps(review(json.loads(Path(args.items_json).read_text()))))


if __name__ == "__main__":
    main()
