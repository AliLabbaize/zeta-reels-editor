"""Captions: output-timeline correctness, chunking limits, Arabic integrity."""

from __future__ import annotations

import re

import pytest

from helpers import captions, config as configs
from helpers.edl import EDL, Range
from helpers.paths import EditPaths
from helpers.words import Word, WordsDoc

ASPECTS = ["9:16", "4:5", "1:1", "16:9"]


def w(text, start, end, display=None):
    return Word(word=text, start=start, end=end, display=display)


@pytest.fixture
def edl():
    # Two kept ranges with a cut between them: output is 0-2s then 2-5s.
    return EDL(
        sources={"raw01": "/tmp/raw01.mp4"},
        ranges=[Range("raw01", 1.0, 3.0, cut_before={"class": "intro_trim", "text": "salam"}),
                Range("raw01", 10.0, 13.0, cut_before={"class": "filler", "text": "يعني"})],
        subtitles="edit/captions/final.ass",
    )


@pytest.fixture
def doc():
    return WordsDoc(
        source={"name": "raw01"},
        words=[
            w("الشركة", 1.0, 1.30, display="الشَّرِكة"),
            w("ديال", 1.35, 1.60),
            w("SpaceX", 1.65, 2.10),
            w("قالت.", 2.20, 2.60),
            # inside the cut: 3.0 - 10.0 does not exist in the output
            w("يعني", 5.00, 5.40),
            w("زعما", 6.00, 6.40),
            # second range
            w("هادي", 10.00, 10.40),
            w("خمسة", 10.50, 10.90),
            w("ملايير", 11.00, 11.60),
            w("دولار", 11.70, 12.20),
        ],
    )


def test_output_times_come_from_the_edl(edl, doc):
    cues = captions.build_cues(doc, edl, aspect="9:16")
    assert cues, "expected cues"
    # First range: source 1.0 -> output 0.0
    assert cues[0].start == pytest.approx(edl.to_output_time("raw01", 1.0))
    assert cues[0].start == pytest.approx(0.0)

    # A word in the SECOND range carries the accumulated offset, not its source time.
    second = [c for c in cues if c.range_index == 1]
    assert second, "expected cues from the second range"
    assert second[0].start == pytest.approx(edl.to_output_time("raw01", 10.0))
    assert second[0].start == pytest.approx(2.0)
    assert second[0].start != pytest.approx(10.0)

    # Every word, everywhere, sits where the EDL says it does.
    for cue in cues:
        for cw in cue.words:
            src = [x for x in doc.words if (x.display or x.word) == cw.text][0]
            assert cw.start == pytest.approx(edl.to_output_time("raw01", src.start))


def test_cut_words_produce_no_cue(edl, doc):
    cues = captions.build_cues(doc, edl, aspect="9:16")
    text = " ".join(c.text for c in cues)
    assert "يعني" not in text
    assert "زعما" not in text
    assert "هادي" in text


def test_no_cue_crosses_a_segment_boundary(edl, doc):
    cues = captions.build_cues(doc, edl, aspect="9:16")
    offsets = edl.offsets()
    for cue in cues:
        seg_start = offsets[cue.range_index]
        seg_end = seg_start + edl.ranges[cue.range_index].duration
        assert cue.start >= seg_start - 1e-6
        assert cue.end <= seg_end + 1e-6
    # The boundary at 2.0s must fall between cues, never inside one.
    boundary = offsets[1]
    assert not any(c.start < boundary < c.end for c in cues)


def test_short_cue_is_not_extended_across_the_boundary(edl):
    # A single word ending right at the end of range 0: min_cue_duration_s would
    # push it past the cut if the clamp were missing.
    doc = WordsDoc(source={"name": "raw01"},
                   words=[w("صافي", 2.85, 2.95), w("هادي", 10.0, 10.4)])
    cues = captions.build_cues(doc, edl, aspect="9:16")
    first = cues[0]
    assert first.range_index == 0
    assert first.end <= 2.0 + 1e-6


@pytest.mark.parametrize("aspect", ASPECTS)
def test_chunk_band_and_char_limit_hold(aspect, edl):
    style = captions.resolve_style(aspect)
    words, t = [], 10.0
    for i in range(40):
        words.append(w(f"kalima{i % 7}", t, t + 0.25))
        t += 0.3
    doc = WordsDoc(source={"name": "raw01"},
                   words=words,
                   )
    big = EDL(sources={"raw01": "/tmp/raw01.mp4"}, ranges=[Range("raw01", 9.0, 30.0)])
    cues = captions.build_cues(doc, big, style=style)
    lo, hi = style.words_per_line
    assert cues
    for cue in cues:
        assert len(cue.words) <= hi
        assert len(cue.text) <= style.max_chars_per_line
        assert cue.duration <= float(style.chunking["max_cue_duration_s"]) + 1e-6
    # Only the tail may fall below the band.
    assert all(len(c.words) >= lo for c in cues[:-1])


def test_gap_and_punctuation_break_cues(edl):
    doc = WordsDoc(source={"name": "raw01"}, words=[
        w("واحد", 10.0, 10.2), w("جوج.", 10.25, 10.45),
        w("تلاتة", 10.5, 10.7),            # after a strong stop
        w("ربعة", 11.4, 11.6),             # after a 700 ms gap
    ])
    cues = captions.build_cues(doc, edl, aspect="9:16")
    texts = [c.text for c in cues]
    assert texts == ["واحد جوج.", "تلاتة", "ربعة"]


def test_min_cue_duration_is_enforced(edl):
    doc = WordsDoc(source={"name": "raw01"}, words=[w("آه", 10.0, 10.08)])
    cues = captions.build_cues(doc, edl, aspect="9:16")
    assert cues[0].duration >= 0.30 - 1e-6


def test_arabic_display_survives_and_is_not_uppercased(edl, doc):
    caps = configs.load("captions")
    caps["style"]["uppercase_latin"] = True  # the risky setting, on purpose
    style = captions.resolve_style("9:16", captions_cfg=caps)
    cues = captions.build_cues(doc, edl, style=style)
    ass = captions.render_ass(cues, style)

    # The vocalised display form, not the matching form, and not mangled.
    assert "الشَّرِكة" in ass
    mixed = [c for c in cues if "SpaceX" in c.text][0]
    assert "SpaceX" in captions.cue_text_for_display(mixed, style)
    assert "SPACEX" not in captions.cue_text_for_display(mixed, style)

    # A Latin-only line may be uppercased; an Arabic or mixed line may not.
    latin = captions.Cue(start=0.0, end=1.0, words=[captions.CueWord("hello", 0.0, 0.5, "lat"),
                                                   captions.CueWord("world", 0.5, 1.0, "lat")])
    assert captions.cue_text_for_display(latin, style) == "HELLO WORLD"


@pytest.mark.parametrize("aspect,res", [("9:16", (1080, 1920)), ("4:5", (1080, 1350)),
                                        ("1:1", (1080, 1080)), ("16:9", (1920, 1080))])
def test_ass_header_playres_and_margins(aspect, res, edl, doc):
    style = captions.resolve_style(aspect)
    assert style.play_res == res
    head = captions.ass_header(style)
    assert f"PlayResX: {res[0]}" in head
    assert f"PlayResY: {res[1]}" in head
    assert "[V4+ Styles]" in head and "[Events]" in head
    assert style.font in head
    # MarginV is the last-but-one style field; it must clear the platform UI.
    style_line = [l for l in head.splitlines() if l.startswith(f"Style: {captions.STYLE_PLAIN},")][0]
    assert style_line.split(",")[-2] == str(style.margin_v)


def test_margin_v_clears_the_instagram_ui_band():
    layout = configs.load("layout")
    style = captions.resolve_style("9:16")
    ui_band = layout["aspects"]["9:16"]["safe_area"]["bottom"]
    assert style.margin_v >= ui_band, "captions would render under the Reels UI"
    # The inline position (which force_style cannot override) agrees.
    prefix = captions._inline_prefix(style, karaoke=False)
    assert f"\\pos(540,{1920 - style.margin_v})" in prefix


def test_karaoke_degrades_when_a_word_has_no_timing(edl):
    style = captions.resolve_style("9:16")
    timed = WordsDoc(source={"name": "raw01"},
                     words=[w("واحد", 10.0, 10.3), w("جوج", 10.35, 10.6)])
    ass = captions.render_ass(captions.build_cues(timed, edl, style=style), style)
    assert "\\k" in ass and captions.STYLE_KARAOKE in ass

    untimed = WordsDoc(source={"name": "raw01"},
                       words=[w("واحد", 10.0, 10.3), Word(word="جوج")])
    cues = captions.build_cues(untimed, edl, style=style)
    assert any(c.has_untimed for c in cues)
    plain = captions.render_ass(cues, style)
    dialogue = [l for l in plain.splitlines() if l.startswith("Dialogue:")]
    assert all("\\k" not in l for l in dialogue)
    assert "جوج" in plain  # the word still reads, it just is not highlighted


def test_srt_is_byte_valid_and_time_ordered(edl, doc, tmp_path):
    style = captions.resolve_style("9:16")
    cues = captions.build_cues(doc, edl, style=style)
    srt = captions.render_srt(cues, style)
    p = tmp_path / "final.srt"
    p.write_bytes(srt.encode("utf-8"))
    blocks = [b for b in p.read_text(encoding="utf-8").split("\n\n") if b.strip()]
    assert len(blocks) == len(cues)

    pattern = re.compile(r"^(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.+)$",
                         re.S)

    def to_s(v):
        hh, mm, rest = v.split(":")
        ss, ms = rest.split(",")
        return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000

    last_end = -1.0
    for i, block in enumerate(blocks, start=1):
        m = pattern.match(block.strip())
        assert m, f"malformed SRT block: {block!r}"
        assert int(m.group(1)) == i
        start, end = to_s(m.group(2)), to_s(m.group(3))
        assert end > start
        assert start >= last_end - 1e-6
        last_end = end


def test_check_bidi_finds_a_mixed_line(edl, doc):
    cues = captions.build_cues(doc, edl, aspect="9:16")
    findings = captions.check_bidi(cues)
    assert findings, "a Darija line containing SpaceX must be reported"
    assert any("SpaceX" in f.text for f in findings)
    assert all(set(f.kinds) & {"ary"} for f in findings)

    # A pure-Arabic line is not a bidi risk and must not be reported.
    assert not any("SpaceX" not in f.text and f.kinds == ["ary"] for f in findings)


def test_check_bidi_marks_an_enclosed_latin_run():
    cues = [captions.Cue(start=0.0, end=1.0, index=0, words=[
        captions.CueWord("قال", 0.0, 0.2, "ary"),
        captions.CueWord("Tesla", 0.2, 0.6, "lat"),
        captions.CueWord("اليوم", 0.6, 1.0, "ary")])]
    findings = captions.check_bidi(cues)
    assert findings[0].enclosed_run is True


def test_write_captions_lands_under_edit(tmp_path, edl, doc):
    paths = EditPaths.for_videos_dir(tmp_path)
    out = captions.write_captions(doc, edl, paths, aspect="9:16")
    assert out["ass"] == tmp_path / "edit" / "captions" / "final.ass"
    assert out["ass"].exists() and out["srt"].exists()
    assert out["ass"].read_text(encoding="utf-8").startswith("[Script Info]")


def test_subtitles_field_is_relative_to_the_edit_dir(tmp_path, edl, doc):
    # render.py resolves a relative EDL subtitles path against the edl.json's
    # own directory, so this is the spelling that actually burns.
    paths = EditPaths.for_videos_dir(tmp_path)
    out = captions.write_captions(doc, edl, paths, aspect="9:16")
    assert out["subtitles_field"] == "captions/final.ass"
    assert (paths.edit / out["subtitles_field"]).exists()


def test_missing_font_is_detected_rather_than_silently_substituted():
    """libass substitutes instead of failing, so nothing else would notice."""
    import subprocess

    from helpers import captions as cap

    installed = subprocess.CompletedProcess(
        [], 0, "Noto Sans Arabic\nDejaVu Sans\n", "")
    assert cap.font_available("Noto Sans Arabic", runner=lambda *a, **k: installed) is True
    assert cap.font_available("IBM Plex Sans Arabic", runner=lambda *a, **k: installed) is False

    # fontconfig absent or broken is "unknown", not "missing": a wrong warning
    # on every render would train people to ignore it.
    failed = subprocess.CompletedProcess([], 1, "", "no fontconfig")
    assert cap.font_available("Noto Sans Arabic", runner=lambda *a, **k: failed) is None
