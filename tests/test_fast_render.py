"""The render reuses the cut segments when only cards or captions changed."""
import json

from helpers import fast_render


def _edl(tmp_path, ranges, overlays=()):
    p = tmp_path / "edl.json"
    p.write_text(json.dumps({"sources": {"a": "/tmp/a.mp4"}, "aspect": "9:16",
                             "ranges": ranges, "overlays": list(overlays)}))
    return p


def test_a_second_render_with_new_cards_reuses_the_base(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(fast_render.vendored, "extract_all_segments",
                        lambda *a, **k: calls.append("extract") or [])
    monkeypatch.setattr(fast_render.vendored, "concat_segments",
                        lambda segs, base, d: base.write_bytes(b"base"))
    monkeypatch.setattr(fast_render.vendored, "build_final_composite",
                        lambda *a, **k: calls.append("composite"))
    ranges = [{"source": "a", "start": 0.0, "end": 2.0}]
    edl = _edl(tmp_path, ranges)
    fast_render.render(edl, tmp_path / "out.mp4", no_loudnorm=True, quiet=True)
    edl = _edl(tmp_path, ranges, [{"file": "x.mov", "start_in_output": 1.0, "duration": 2.0}])
    fast_render.render(edl, tmp_path / "out.mp4", no_loudnorm=True, quiet=True)
    assert calls == ["extract", "composite", "composite"]   # cut once, composited twice


def test_changing_the_cut_re_cuts(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(fast_render.vendored, "extract_all_segments",
                        lambda *a, **k: calls.append("extract") or [])
    monkeypatch.setattr(fast_render.vendored, "concat_segments",
                        lambda segs, base, d: base.write_bytes(b"base"))
    monkeypatch.setattr(fast_render.vendored, "build_final_composite", lambda *a, **k: None)
    fast_render.render(_edl(tmp_path, [{"source": "a", "start": 0.0, "end": 2.0}]),
                       tmp_path / "o.mp4", no_loudnorm=True, quiet=True)
    fast_render.render(_edl(tmp_path, [{"source": "a", "start": 0.0, "end": 3.0}]),
                       tmp_path / "o.mp4", no_loudnorm=True, quiet=True)
    assert calls == ["extract", "extract"]
