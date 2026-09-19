"""Geometry invariants: never cropped, never under the platform UI, never on
top of the card -- for every aspect Zeta publishes to.
"""

from __future__ import annotations

import pytest

from helpers import build_overlay as bo
from helpers import config as cfgmod
from helpers.edl import EDL, EDLError, Range
from helpers.paths import EditPaths

ASPECTS = ["9:16", "4:5", "1:1", "16:9"]
# Shapes a real capture produces: an article header, a whole tall article, a
# square post card, a wide banner.
IMAGE_SIZES = [(1600, 900), (1600, 2400), (1200, 1200), (2400, 600), (800, 1800)]

LAYOUT = cfgmod.load("layout")


def _acfg(aspect):
    return cfgmod.aspect_config(aspect)[1]


def _verified_meta(**kw):
    meta = {
        "slot_id": "slot_01",
        "claim": "OpenAI raised 40 billion",
        "url": "https://openai.com/index/funding/",
        "source_type": "owner",
        "source_origin": "search",
        "verified": True,
        "evidence": "headline and the figure 40 billion are legible",
        "claim_detail": {"trigger_word": "OpenAI"},
        "capture_source_type": "html",
    }
    meta.update(kw)
    return meta


# -- fit_card: the default, and the one that may never crop ------------------


@pytest.mark.parametrize("aspect", ASPECTS)
@pytest.mark.parametrize("size", IMAGE_SIZES)
def test_fit_card_never_crops_and_stays_inside_the_safe_area(aspect, size):
    acfg = _acfg(aspect)
    geom = bo.compute_geometry(size, acfg, LAYOUT, layout="fit_card")

    card, image, pip, safe = geom["card"], geom["image"], geom["pip"], geom["safe"]
    iw, ih = size

    assert geom["fit"] == "contain" and geom["cropped"] is False
    # Aspect ratio preserved to within rounding: nothing is squeezed or trimmed.
    assert image.aspect == pytest.approx(iw / ih, rel=0.01)
    assert image.w <= iw * geom["scale"] + 1 and image.h <= ih * geom["scale"] + 1
    assert safe.contains(card) and card.contains(image)
    assert safe.contains(pip)
    assert not pip.intersects(card)


@pytest.mark.parametrize("aspect", ASPECTS)
def test_nothing_lands_in_the_platform_ui_bands(aspect):
    acfg = _acfg(aspect)
    w, h = acfg["resolution"]
    s = acfg["safe_area"]
    geom = bo.compute_geometry((1600, 900), acfg, LAYOUT, layout="fit_card")

    for name in ("card", "image", "pip"):
        r = geom[name]
        assert r.y >= s["top"], f"{name} is under the top chrome"
        assert r.bottom <= h - s["bottom"], f"{name} is under the caption band"
        assert r.x >= s["left"]
        assert r.right <= w - s["right"], f"{name} is under the action rail"


def test_fit_card_reserves_the_pip_band_even_for_a_very_tall_screenshot():
    acfg = _acfg("9:16")
    geom = bo.compute_geometry((800, 6000), acfg, LAYOUT, layout="fit_card")
    # The tall image is height-limited, so the card grows until it touches the
    # PiP band -- and must stop there, not under it.
    assert geom["card"].bottom <= geom["pip"].y
    assert not geom["pip"].intersects(geom["card"])


def test_card_hugs_the_image_so_there_is_no_empty_white_margin():
    geom = bo.compute_geometry((1600, 900), _acfg("9:16"), LAYOUT, layout="fit_card")
    pad = LAYOUT["inserts"]["layouts"]["fit_card"]["card_padding_px"]
    assert geom["card"].w == geom["image"].w + 2 * pad
    assert geom["card"].h == geom["image"].h + 2 * pad


def test_center_anchor_keeps_the_card_above_the_pip():
    cfg = {"inserts": {**LAYOUT["inserts"]}}
    cfg["inserts"]["layouts"] = {**LAYOUT["inserts"]["layouts"]}
    cfg["inserts"]["layouts"]["fit_card"] = {
        **LAYOUT["inserts"]["layouts"]["fit_card"], "card_anchor": "center"}
    geom = bo.compute_geometry((1600, 900), _acfg("9:16"), cfg, layout="fit_card")
    assert geom["safe"].contains(geom["card"])
    assert not geom["pip"].intersects(geom["card"])


def test_an_impossible_pip_scale_is_refused_rather_than_silently_overlapped():
    cfg = {"inserts": {**LAYOUT["inserts"], "layouts": {
        **LAYOUT["inserts"]["layouts"],
        "fit_card": {**LAYOUT["inserts"]["layouts"]["fit_card"], "facecam_scale": 0.95}}}}
    with pytest.raises(bo.GeometryError, match="facecam_scale"):
        bo.compute_geometry((1600, 900), _acfg("9:16"), cfg, layout="fit_card")


# -- the alternatives --------------------------------------------------------


@pytest.mark.parametrize("aspect", ASPECTS)
@pytest.mark.parametrize("size", IMAGE_SIZES)
def test_split_also_contains_the_screenshot(aspect, size):
    geom = bo.compute_geometry(size, _acfg(aspect), LAYOUT, layout="split")
    assert geom["cropped"] is False
    assert geom["image"].aspect == pytest.approx(size[0] / size[1], rel=0.01)
    assert geom["safe"].contains(geom["card"]) and geom["card"].contains(geom["image"])
    assert geom["safe"].contains(geom["pip"])
    assert not geom["pip"].intersects(geom["card"])


def test_fullframe_pip_is_the_only_layout_allowed_to_crop():
    geom = bo.compute_geometry((1600, 900), _acfg("9:16"), LAYOUT, layout="fullframe_pip")
    assert geom["fit"] == "cover" and geom["cropped"] is True
    # The facecam still has to stay clear of the UI even here.
    assert geom["safe"].contains(geom["pip"])


def test_cropping_requires_the_explicit_opt_in():
    cfg = {"inserts": {**LAYOUT["inserts"], "layouts": {
        **LAYOUT["inserts"]["layouts"],
        "fullframe_pip": {**LAYOUT["inserts"]["layouts"]["fullframe_pip"],
                          "allow_crop": False}}}}
    with pytest.raises(bo.GeometryError, match="allow_crop"):
        bo.compute_geometry((1600, 900), _acfg("9:16"), cfg, layout="fullframe_pip")


def test_unknown_layout_is_an_error():
    with pytest.raises(bo.GeometryError, match="unknown insert layout"):
        bo.compute_geometry((100, 100), _acfg("9:16"), LAYOUT, layout="mosaic")


def test_validate_geometry_catches_a_card_pushed_out_of_the_safe_area():
    geom = bo.compute_geometry((1600, 900), _acfg("9:16"), LAYOUT, layout="fit_card")
    shift = geom["card"].y            # slide the whole card up into the top chrome
    geom["card"] = bo.Rect(geom["card"].x, 0, geom["card"].w, geom["card"].h)
    geom["image"] = bo.Rect(geom["image"].x, geom["image"].y - shift,
                            geom["image"].w, geom["image"].h)
    with pytest.raises(bo.GeometryError, match="safe area"):
        bo.validate_geometry(geom)


def test_geometry_is_json_serialisable_for_the_report():
    import json
    geom = bo.compute_geometry((1600, 900), _acfg("9:16"), LAYOUT)
    blob = json.loads(json.dumps(bo.geometry_to_dict(geom)))
    assert blob["layout"] == LAYOUT["inserts"]["default_layout"]
    assert set(blob["card"]) == {"x", "y", "w", "h"}


# -- compositing and encoding ------------------------------------------------


def test_render_frame_produces_a_full_resolution_frame(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    shot = tmp_path / "shot.png"
    face = tmp_path / "face.png"
    Image.new("RGB", (1600, 900), (255, 0, 0)).save(shot)
    Image.new("RGB", (1080, 1920), (0, 0, 255)).save(face)

    geom = bo.compute_geometry((1600, 900), _acfg("9:16"), LAYOUT, layout="fit_card")
    out = bo.render_frame(shot, tmp_path / "frame.png", geom, face)

    with Image.open(out) as im:
        assert im.size == (1080, 1920)
        # Centre of the card is the screenshot, untouched by the background.
        assert im.getpixel((geom["image"].x + geom["image"].w // 2,
                            geom["image"].y + geom["image"].h // 2)) == (255, 0, 0)
        # The blurred facecam background is dimmed, so it is darker than the source.
        bg = im.getpixel((5, 5))
        assert bg[2] < 255 and sum(bg) < 3 * 200


def test_encode_cmd_carries_fps_fades_and_the_ken_burns_zoom():
    cmd = bo.encode_cmd("frame.png", "overlay.mp4", duration_s=4.5, fps=30,
                        size=(1080, 1920), fade_in_s=0.15, fade_out_s=0.15,
                        ken_burns=0.03)
    vf = cmd[cmd.index("-vf") + 1]
    assert "zoompan" in vf and "1.0300" in vf
    assert "fade=t=in:st=0:d=0.150" in vf
    assert "fade=t=out:st=4.350" in vf
    assert f"s=1080x1920" in vf
    assert cmd[cmd.index("-r") + 1] == "30"
    assert "-an" in cmd            # the overlay never carries audio


def test_encode_cmd_without_ken_burns_is_a_plain_scale():
    vf = bo.encode_cmd("f.png", "o.mp4", duration_s=3, fps=30, size=(1080, 1920),
                       ken_burns=0)[bo.encode_cmd("f.png", "o.mp4", duration_s=3, fps=30,
                                                  size=(1080, 1920),
                                                  ken_burns=0).index("-vf") + 1]
    assert "zoompan" not in vf and vf.startswith("scale=1080:1920")


# -- EDL placement -----------------------------------------------------------


def test_overlay_for_slot_starts_one_lead_in_before_the_trigger():
    ov = bo.overlay_for_slot("slot_01", _verified_meta(), trigger_time_output=14.5,
                             total_duration_s=87.4, layout_cfg=LAYOUT, aspect="9:16")
    lead_in = LAYOUT["inserts"]["timing"]["lead_in_s"]
    assert ov.start_in_output == pytest.approx(14.5 - lead_in)
    assert ov.duration == pytest.approx(LAYOUT["inserts"]["timing"]["default_duration_s"])
    assert ov.meta["verified"] is True
    assert ov.meta["url"].startswith("https://openai.com/")
    assert ov.meta["trigger_word"] == "OpenAI"
    assert ov.meta["trigger_time_output"] == 14.5
    assert ov.meta["layout"] == LAYOUT["inserts"]["default_layout"]
    # Relative to edl.json's own directory (`<videos_dir>/edit/`), which is how
    # the vendored render.py resolves it -- no repeated `edit/` segment. A float
    # card carries alpha, so it is a .mov.
    assert ov.file == "screenshots/slot_01/overlay.mov"


def test_overlay_is_clamped_to_the_cut_at_both_ends():
    head = bo.overlay_for_slot("slot_01", _verified_meta(), trigger_time_output=0.1,
                               total_duration_s=20.0, layout_cfg=LAYOUT)
    assert head.start_in_output == 0.0

    tail = bo.overlay_for_slot("slot_01", _verified_meta(), trigger_time_output=19.9,
                               total_duration_s=20.0, layout_cfg=LAYOUT)
    assert tail.start_in_output + tail.duration <= 20.0 + 1e-6

    tiny = bo.overlay_for_slot("slot_01", _verified_meta(), trigger_time_output=1.0,
                               total_duration_s=2.0, layout_cfg=LAYOUT)
    assert tiny.start_in_output == 0.0 and tiny.duration == pytest.approx(2.0)


def test_duration_is_clamped_to_the_profile_bounds():
    long = bo.overlay_for_slot("slot_01", _verified_meta(), trigger_time_output=10.0,
                               total_duration_s=100.0, duration_s=30.0, layout_cfg=LAYOUT)
    assert long.duration == pytest.approx(LAYOUT["inserts"]["timing"]["max_duration_s"])
    short = bo.overlay_for_slot("slot_01", _verified_meta(), trigger_time_output=10.0,
                                total_duration_s=100.0, duration_s=0.4, layout_cfg=LAYOUT)
    assert short.duration == pytest.approx(LAYOUT["inserts"]["timing"]["min_duration_s"])


def test_an_unverified_slot_never_becomes_an_overlay():
    meta = _verified_meta(verified=False, dropped_reason="vision check found no evidence")
    with pytest.raises(ValueError, match="unverified"):
        bo.overlay_for_slot("slot_01", meta, trigger_time_output=10.0,
                            total_duration_s=60.0, layout_cfg=LAYOUT)


def test_a_slot_with_no_url_never_becomes_an_overlay():
    with pytest.raises(ValueError, match="no source url"):
        bo.overlay_for_slot("slot_01", _verified_meta(url=None), trigger_time_output=10.0,
                            total_duration_s=60.0, layout_cfg=LAYOUT)


def test_overlay_passes_the_edl_validator():
    ov = bo.overlay_for_slot("slot_01", _verified_meta(), trigger_time_output=14.5,
                             total_duration_s=40.0, layout_cfg=LAYOUT, aspect="9:16")
    edl = EDL(sources={"raw01": "/abs/raw01.mp4"},
              ranges=[Range(source="raw01", start=2.0, end=42.0, beat="HOOK")],
              overlays=[ov], aspect="9:16")
    assert edl.validate() is edl

    # And the validator is what stops an unverified one, belt and braces.
    ov.meta["verified"] = False
    with pytest.raises(EDLError, match="verified"):
        edl.validate()


def test_two_slots_do_not_collide_on_the_timeline():
    a = bo.overlay_for_slot("slot_01", _verified_meta(), trigger_time_output=10.0,
                            total_duration_s=60.0, layout_cfg=LAYOUT)
    b = bo.overlay_for_slot("slot_02", _verified_meta(slot_id="slot_02"),
                            trigger_time_output=30.0, total_duration_s=60.0,
                            layout_cfg=LAYOUT)
    edl = EDL(sources={"raw01": "/abs/raw01.mp4"},
              ranges=[Range(source="raw01", start=0.0, end=60.0)], overlays=[a, b])
    edl.validate()
    assert b.file != a.file


def test_overlay_file_path_matches_the_slot_directory(tmp_path):
    paths = EditPaths.for_videos_dir(tmp_path)
    ov = bo.overlay_for_slot("slot_07", _verified_meta(slot_id="slot_07"),
                             trigger_time_output=5.0, total_duration_s=30.0,
                             layout_cfg=LAYOUT)
    assert ov.file == "screenshots/slot_07/overlay.mov"
    # ... and that is exactly the slot directory EditPaths creates, seen from edit/.
    assert paths.slot("slot_07") == paths.edit / ov.file.rsplit("/", 1)[0]


def test_build_overlay_composites_then_encodes_at_the_aspect_fps(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    shot = tmp_path / "shot.png"
    face = tmp_path / "face.png"
    Image.new("RGB", (1600, 900), (12, 200, 40)).save(shot)
    Image.new("RGB", (1080, 1920), (30, 30, 30)).save(face)
    seen: list = []

    class Proc:
        returncode = 0
        stderr = ""

    def runner(cmd, **kw):
        seen.append(cmd)
        return Proc()

    out = bo.build_overlay(shot, tmp_path / "overlay.mp4", facecam_still=face,
                           aspect="9:16", layout="fit_card", duration_s=30.0,
                           runner=runner)

    # The still frame is composited before ffmpeg is ever invoked.
    assert (tmp_path / "overlay_frame.png").exists()
    assert out["duration_s"] == pytest.approx(LAYOUT["inserts"]["timing"]["max_duration_s"])
    assert out["geometry"]["layout"] == "fit_card"
    assert seen[0][seen[0].index("-r") + 1] == "30"


def test_build_overlay_surfaces_an_encoder_failure(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    shot = tmp_path / "shot.png"
    Image.new("RGB", (800, 600), (255, 255, 255)).save(shot)

    class Proc:
        returncode = 1
        stderr = "Unknown encoder 'libx264'"

    with pytest.raises(RuntimeError, match="ffmpeg failed"):
        bo.build_overlay(shot, tmp_path / "o.mp4", aspect="9:16",
                         runner=lambda cmd, **kw: Proc())


def test_defaults_come_from_configs_layout_yaml_when_nothing_is_passed():
    ov = bo.overlay_for_slot("slot_01", _verified_meta(), trigger_time_output=9.0,
                             total_duration_s=60.0)
    assert ov.start_in_output == pytest.approx(9.0 - LAYOUT["inserts"]["timing"]["lead_in_s"])
    assert ov.meta["layout"] == LAYOUT["inserts"]["default_layout"] == "cutout"


def test_overlay_path_resolves_the_way_the_vendored_renderer_resolves_it(tmp_path):
    from helpers import render                      # vendored, untouched
    paths = EditPaths.for_videos_dir(tmp_path)
    ov = bo.overlay_for_slot("slot_01", _verified_meta(), trigger_time_output=5.0,
                             total_duration_s=30.0, layout_cfg=LAYOUT)
    # render.py resolves overlay files against the directory holding edl.json.
    assert render.resolve_path(ov.file, paths.edit) == paths.slot("slot_01") / "overlay.mov"


def test_a_float_card_stays_out_of_the_top_band_and_the_caption_band():
    geom = bo.compute_geometry((1600, 1200), _acfg("9:16"), LAYOUT, layout="float")
    card = geom["card"]
    assert card.y >= 1920 * LAYOUT["inserts"]["top_reserved"]   # Ali's title space
    assert card.y + card.h <= round(1920 * LAYOUT["inserts"]["layouts"]["float"]["card_bottom"])
    assert geom["cropped"] is False and geom["pip"] is None


def test_a_float_overlay_is_encoded_with_alpha(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    shot = tmp_path / "shot.png"
    Image.new("RGB", (1600, 800), (255, 255, 255)).save(shot)
    seen: list = []

    class Proc:
        returncode = 0
        stderr = ""

    out = bo.build_overlay(shot, tmp_path / "overlay.mp4", aspect="9:16", layout="float",
                           runner=lambda cmd, **kw: seen.append(cmd) or Proc())
    assert out["file"].endswith(".mov")
    assert "qtrle" in seen[0] and "alpha=1" in " ".join(seen[0])
    with Image.open(tmp_path / "overlay_frame.png") as frame:
        assert frame.mode == "RGBA" and frame.getpixel((5, 5))[3] == 0   # see-through
