"""The review page serves the shots and hands back exactly what Ali chose."""
import json
import socket
import threading
import urllib.request

from helpers import review_shots


def test_review_returns_keep_and_drop_and_defaults_to_keep(tmp_path):
    for n in (1, 2, 3):
        (tmp_path / f"s{n}.png").write_bytes(b"\x89PNG")
    items = [{"slot_id": f"slot_0{n}", "time": 10.0 * n, "said": "هاد ال agents",
              "claim": "c", "url": "https://x.test/a", "image": str(tmp_path / f"s{n}.png")}
             for n in (1, 2, 3)]
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    out = {}
    t = threading.Thread(target=lambda: out.update(
        review_shots.review(items, open_browser=False, port=port)))
    t.start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(50):
        try:
            page = urllib.request.urlopen(base + "/").read().decode()
            break
        except OSError:
            threading.Event().wait(0.05)
    assert "slot_02" in page and "هاد ال agents" in page
    assert urllib.request.urlopen(base + "/img/slot_01").read() == b"\x89PNG"
    urllib.request.urlopen(urllib.request.Request(
        base + "/submit", data=json.dumps({
            "slot_02": {"keep": False, "note": "show the OpenAI report", "url": ""},
            "slot_03": {"keep": True, "note": "", "url": "https://cdn.openai.com/r.pdf"}}).encode(),
        method="POST"))
    t.join(5)
    assert out["slot_01"] == {"keep": True, "note": "", "url": ""}
    assert out["slot_02"]["keep"] is False and out["slot_02"]["note"] == "show the OpenAI report"
    assert out["slot_03"]["url"] == "https://cdn.openai.com/r.pdf"
