"""PNG -> `overlay.mp4`, and the EDL overlay entry that places it.

Stage 5 step 5. Two halves, deliberately separated:

  compute_geometry()  pure arithmetic. No PIL, no ffmpeg, no files. It answers
                      where the card, the screenshot and the facecam PiP land in
                      OUTPUT pixels, and it is what the invariants are tested
                      against.
  build_overlay()     the compositing: blurred facecam background, rounded card,
                      PiP, Ken Burns, fades, encode.

THE SCREENSHOT IS NEVER CROPPED (except in the opt-in `fullframe_pip` layout).
A cropped headline can change what the source appears to say, so `fit_card` and
`split` always `contain` the image: aspect ratio preserved, whole image visible.
Everything is also kept inside the aspect's safe area, because a card under the
Instagram action rail or behind the caption band is evidence nobody can read.

Overlay clips are built at the OUTPUT resolution and placed on the OUTPUT
timeline; helpers/render.py does the `setpts=PTS-STARTPTS+T/TB` shift (hard
rule 4), so nothing here touches PTS.

    python helpers/build_overlay.py /videos/edit/screenshots/slot_01 \
        --facecam still.png --aspect 9:16 --geometry
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # `python helpers/build_overlay.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helpers import config as cfgmod
from helpers.edl import Overlay
from helpers.screenshot import resolve_aspect


def overlay_file(slot_id: str, layout: str | None = None) -> str:
    """The EDL `file` for a slot, relative to the EDL itself.

    `render.py:resolve_path` resolves a relative overlay path against the
    directory holding `edl.json`, which IS `<videos_dir>/edit/`. So the path
    stored here must NOT repeat the `edit/` segment the spec's example shows,
    or the render resolves `<videos_dir>/edit/edit/screenshots/...` and fails.
    """
    return f"screenshots/{slot_id}/overlay.{'mov' if layout in ('float', 'cutout') else 'mp4'}"


class GeometryError(ValueError):
    """A layout that would crop the source or hide it under platform UI."""


@dataclass(frozen=True)
class Rect:
    """Integer pixel rect in output space, `(x, y)` = top-left."""

    x: int
    y: int
    w: int
    h: int

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h

    @property
    def aspect(self) -> float:
        return self.w / self.h if self.h else 0.0

    def box(self) -> tuple[int, int, int, int]:
        """PIL-style `(left, upper, right, lower)`."""
        return (self.x, self.y, self.right, self.bottom)

    def contains(self, other: "Rect", tol: int = 0) -> bool:
        return (other.x >= self.x - tol and other.y >= self.y - tol
                and other.right <= self.right + tol and other.bottom <= self.bottom + tol)

    def intersects(self, other: "Rect") -> bool:
        return not (other.x >= self.right or other.right <= self.x
                    or other.y >= self.bottom or other.bottom <= self.y)

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}


# --------------------------------------------------------------------------
# Pure geometry


def _inserts_block(layout_cfg: dict | None) -> dict:
    """Accept either the whole layout.yaml or just its `inserts` block."""
    if layout_cfg is None:
        layout_cfg = cfgmod.load("layout")
    return layout_cfg.get("inserts", layout_cfg) or {}


def safe_rect(aspect_cfg: dict) -> Rect:
    w, h = (int(v) for v in aspect_cfg["resolution"])
    s = aspect_cfg.get("safe_area", {}) or {}
    x0, y0 = int(s.get("left", 0)), int(s.get("top", 0))
    return Rect(x0, y0, w - int(s.get("right", 0)) - x0, h - int(s.get("bottom", 0)) - y0)


def _contain(iw: int, ih: int, box_w: int, box_h: int) -> tuple[int, int, float]:
    """Largest `w, h` with the source's aspect ratio that fits in the box."""
    scale = min(box_w / iw, box_h / ih)
    return max(1, round(iw * scale)), max(1, round(ih * scale)), scale


def _corner_rect(corner: str, w: int, h: int, inside: Rect, margin: int) -> Rect:
    right = "right" in corner
    bottom = "bottom" in corner
    x = (inside.right - margin - w) if right else (inside.x + margin)
    y = (inside.bottom - margin - h) if bottom else (inside.y + margin)
    return Rect(int(x), int(y), int(w), int(h))


def compute_geometry(image_size: tuple[int, int], aspect_cfg: dict,
                     layout_cfg: dict | None = None, *, layout: str | None = None) -> dict:
    """Where everything lands, in output pixels. Pure: no PIL, no ffmpeg, no IO.

    `image_size` is the captured PNG's `(width, height)`. Returns a dict with
    `card`, `image` and `pip` as `Rect`s plus the styling the compositor needs.
    The returned `image` rect is the SCALED SCREENSHOT: in `fit_card` and
    `split` it is a `contain` fit (nothing cropped) and it sits inside `card`;
    in `fullframe_pip` it is a `cover` fit and may extend past the frame, which
    is the crop that layout opts into.
    """
    iw, ih = (int(v) for v in image_size)
    if iw <= 0 or ih <= 0:
        raise GeometryError(f"bad image size {image_size}")

    inserts = _inserts_block(layout_cfg)
    name = layout or inserts.get("default_layout", "fit_card")
    lcfg = ((inserts.get("layouts") or {}).get(name)) or {}
    out_w, out_h = (int(v) for v in aspect_cfg["resolution"])
    frame = Rect(0, 0, out_w, out_h)
    safe = safe_rect(aspect_cfg)
    if safe.w <= 0 or safe.h <= 0:
        raise GeometryError(f"safe area is empty for resolution {out_w}x{out_h}")

    geom: dict[str, Any] = {
        "layout": name,
        "frame": frame,
        "safe": safe,
        "fps": int(aspect_cfg.get("fps", 30)),
        "timing": dict(inserts.get("timing") or {}),
        "style": dict(lcfg),
    }

    if name == "fit_card":
        pad = int(lcfg.get("card_padding_px", 24))
        pip: Rect | None = None
        reserved = 0
        if lcfg.get("facecam_pip", True):
            # The PiP keeps the output's aspect ratio: it is a scaled copy of the
            # facecam frame, not a crop of it.
            pip_w = round(float(lcfg.get("facecam_scale", 0.3)) * out_w)
            pip_h = round(pip_w * out_h / out_w)
            pip = _corner_rect(str(lcfg.get("facecam_corner", "bottom_right")),
                               pip_w, pip_h, safe, margin=0)
            # Reserve the PiP's band so the card can never end up under it.
            reserved = pip_h + pad

        avail_h = safe.h - reserved
        if avail_h < 4 * pad:
            raise GeometryError(
                f"fit_card: only {avail_h}px left for the card after reserving the "
                f"facecam PiP; lower inserts.layouts.fit_card.facecam_scale")
        img_w, img_h, scale = _contain(iw, ih, safe.w - 2 * pad, avail_h - 2 * pad)

        card = Rect(safe.x + (safe.w - (img_w + 2 * pad)) // 2, 0,
                    img_w + 2 * pad, img_h + 2 * pad)
        if str(lcfg.get("card_anchor", "upper")) == "center":
            card = Rect(card.x, safe.y + (avail_h - card.h) // 2, card.w, card.h)
        else:
            card = Rect(card.x, safe.y, card.w, card.h)
        image = Rect(card.x + pad, card.y + pad, img_w, img_h)

        geom.update({"card": card, "image": image, "pip": pip, "fit": "contain",
                     "cropped": False, "scale": scale,
                     "card_radius_px": int(lcfg.get("card_radius_px", 28)),
                     "pip_radius_px": int(lcfg.get("facecam_radius_px", 24)),
                     "card_background": lcfg.get("card_background", "#FFFFFF"),
                     "background": lcfg.get("frame_background", "blurred_facecam"),
                     "background_solid": lcfg.get("frame_background_solid", "#0B0B0D"),
                     "background_blur": int(lcfg.get("frame_background_blur", 28)),
                     "background_dim": float(lcfg.get("frame_background_dim", 0.45))})

    elif name in ("float", "cutout"):
        # A card floating over the LIVE video: no blurred still, no face PiP.
        # It sits in a band below the face and above the captions, and never
        # enters the top band Ali keeps for his title text.
        pad = int(lcfg.get("card_padding_px", 16))
        mx = int(lcfg.get("margin_x_px", 60))
        top = round(out_h * max(float(lcfg.get("card_top", 0.45)),
                                float(inserts.get("top_reserved", 0.0))))
        bottom = round(out_h * float(lcfg.get("card_bottom", 0.73)))
        box = Rect(mx, top, out_w - 2 * mx, bottom - top)
        img_w, img_h, scale = _contain(iw, ih, box.w - 2 * pad, box.h - 2 * pad)
        # "top": hang from the band's top (Ali: screens at the top, not over my
        # mouth); default: hug the band's bottom.
        y = box.y if lcfg.get("card_anchor") == "top" else box.y + box.h - (img_h + 2 * pad)
        card = Rect(box.x + (box.w - img_w - 2 * pad) // 2, y,
                    img_w + 2 * pad, img_h + 2 * pad)
        image = Rect(card.x + pad, card.y + pad, img_w, img_h)
        geom.update({"card": card, "image": image, "pip": None, "fit": "contain",
                     "cropped": False, "scale": scale, "box": box,
                     "card_radius_px": int(lcfg.get("card_radius_px", 24)),
                     "card_background": lcfg.get("card_background", "#FFFFFF"),
                     "shadow_px": int(lcfg.get("shadow_px", 24)),
                     "background": "transparent", "background_solid": "#000000",
                     "background_blur": 0, "background_dim": 0.0, "pip_radius_px": 0})

    elif name == "split":
        gap = int(lcfg.get("gap_px", 16))
        share = float(lcfg.get("screenshot_share", 0.5))
        top_h = int(round((safe.h - gap) * share))
        bottom_h = safe.h - gap - top_h
        if top_h <= 0 or bottom_h <= 0:
            raise GeometryError("split: screenshot_share leaves an empty block")
        block = Rect(safe.x, safe.y, safe.w, top_h)
        img_w, img_h, scale = _contain(iw, ih, block.w, block.h)
        image = Rect(block.x + (block.w - img_w) // 2,
                     block.y + (block.h - img_h) // 2, img_w, img_h)
        pip = Rect(safe.x, safe.y + top_h + gap, safe.w, bottom_h)

        geom.update({"card": block, "image": image, "pip": pip, "fit": "contain",
                     "cropped": False, "scale": scale, "card_radius_px": 0,
                     "pip_radius_px": 0,
                     "card_background": lcfg.get("background", "#0B0B0D"),
                     "background": "solid",
                     "background_solid": lcfg.get("background", "#0B0B0D"),
                     "background_blur": 0, "background_dim": 0.0})

    elif name == "fullframe_pip":
        if not lcfg.get("allow_crop", False):
            raise GeometryError(
                "fullframe_pip crops the source; set allow_crop: true to opt in")
        # cover: fill the frame, overflow off the edges. Only for images whose
        # edges carry no information -- that is the layout's whole contract.
        scale = max(out_w / iw, out_h / ih)
        img_w, img_h = round(iw * scale), round(ih * scale)
        image = Rect((out_w - img_w) // 2, (out_h - img_h) // 2, img_w, img_h)
        pip_w = round(float(lcfg.get("facecam_scale", 0.28)) * out_w)
        pip_h = round(pip_w * out_h / out_w)
        pip = _corner_rect(str(lcfg.get("facecam_corner", "bottom_right")),
                           pip_w, pip_h, safe, margin=0)

        geom.update({"card": frame, "image": image, "pip": pip, "fit": "cover",
                     "cropped": (img_w > out_w or img_h > out_h), "scale": scale,
                     "card_radius_px": 0, "pip_radius_px": 24,
                     "card_background": "#0B0B0D", "background": "solid",
                     "background_solid": "#0B0B0D", "background_blur": 0,
                     "background_dim": 0.0})
    else:
        raise GeometryError(f"unknown insert layout {name!r}")

    validate_geometry(geom, source_size=(iw, ih))
    return geom


def validate_geometry(geom: dict, *, source_size: tuple[int, int] | None = None,
                      tol: int = 1) -> dict:
    """The invariants, enforced at build time and asserted in the tests."""
    layout = geom["layout"]
    frame, safe = geom["frame"], geom["safe"]
    card, image, pip = geom["card"], geom["image"], geom.get("pip")

    if layout in ("fit_card", "split"):
        if geom.get("cropped") or geom.get("fit") != "contain":
            raise GeometryError(f"{layout}: screenshot must be contained, not cropped")
        if not card.contains(image, tol=tol):
            raise GeometryError(f"{layout}: screenshot escapes its card")
        if not safe.contains(card, tol=tol):
            raise GeometryError(
                f"{layout}: card {card.to_dict()} leaves the safe area "
                f"{safe.to_dict()} -- it would sit under the Instagram UI")
        if source_size:
            iw, ih = source_size
            # Aspect preserved to within a pixel of rounding.
            if abs(image.aspect - iw / ih) > 0.01:
                raise GeometryError(
                    f"{layout}: screenshot aspect {image.aspect:.4f} != source "
                    f"{iw / ih:.4f}; it is being squeezed")
        if pip is not None:
            if not safe.contains(pip, tol=tol):
                raise GeometryError(f"{layout}: facecam PiP leaves the safe area")
            if pip.intersects(card):
                raise GeometryError(
                    f"{layout}: facecam PiP {pip.to_dict()} covers the card "
                    f"{card.to_dict()}")
    else:  # fullframe_pip: the image may overflow by design, nothing else may
        if pip is not None and not safe.contains(pip, tol=tol):
            raise GeometryError("fullframe_pip: facecam PiP leaves the safe area")

    for name in ("card", "pip"):
        r = geom.get(name)
        if r is not None and not frame.contains(r, tol=tol):
            raise GeometryError(f"{name} {r.to_dict()} falls outside the {frame.w}x"
                                f"{frame.h} frame")
    return geom


def geometry_to_dict(geom: dict) -> dict:
    """JSON-able copy (for `meta.json` and the decision report)."""
    return {k: (v.to_dict() if isinstance(v, Rect) else v) for k, v in geom.items()}


# --------------------------------------------------------------------------
# Compositing (PIL) and encoding (ffmpeg)


def _hex_rgb(value: str) -> tuple[int, int, int]:
    s = str(value).lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


def _pil():
    try:
        from PIL import Image, ImageDraw, ImageFilter
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is not installed: uv pip install -e '.' (pillow is a base "
            "dependency)") from exc
    return Image, ImageDraw, ImageFilter


def _cover(img, w: int, h: int):
    """Resize to fill `w x h`, centre-cropping the overflow. Facecam only."""
    Image, _d, _f = _pil()
    scale = max(w / img.width, h / img.height)
    resized = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))),
                         Image.LANCZOS)
    left = (resized.width - w) // 2
    top = (resized.height - h) // 2
    return resized.crop((left, top, left + w, top + h))


def _rounded_mask(size: tuple[int, int], radius: int):
    Image, ImageDraw, _f = _pil()
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, size[0] - 1, size[1] - 1),
                                           radius=max(0, radius), fill=255)
    return mask


def render_float_frame(screenshot_png: str | Path, out_png: str | Path, geom: dict) -> Path:
    """Transparent full frame: a white rounded card with a soft shadow, nothing else."""
    Image, ImageDraw, ImageFilter = _pil()
    frame, card, image = geom["frame"], geom["card"], geom["image"]
    r, sh = geom["card_radius_px"], geom["shadow_px"]
    canvas = Image.new("RGBA", (frame.w, frame.h), (0, 0, 0, 0))
    shadow = Image.new("RGBA", (frame.w, frame.h), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle(
        (card.x, card.y + sh // 3, card.x + card.w, card.y + card.h + sh // 3),
        radius=r, fill=(0, 0, 0, 110))
    canvas = Image.alpha_composite(canvas, shadow.filter(ImageFilter.GaussianBlur(sh)))
    ImageDraw.Draw(canvas).rounded_rectangle(
        (card.x, card.y, card.x + card.w, card.y + card.h), radius=r,
        fill=_hex_rgb(geom["card_background"]) + (255,))
    with Image.open(screenshot_png) as shot:
        shot = shot.convert("RGBA").resize((image.w, image.h), Image.LANCZOS)
    mask = _rounded_mask((image.w, image.h), max(0, r - 8))
    canvas.paste(shot, (image.x, image.y), mask)
    out = Path(out_png)
    canvas.save(out)
    return out


def render_frame(screenshot_png: str | Path, out_png: str | Path, geom: dict,
                 facecam_still: str | Path | None = None) -> Path:
    """Composite one still overlay frame at the output resolution."""
    Image, _draw, ImageFilter = _pil()
    frame: Rect = geom["frame"]
    card: Rect = geom["card"]
    image: Rect = geom["image"]
    pip: Rect | None = geom.get("pip")

    canvas = Image.new("RGB", (frame.w, frame.h), _hex_rgb(geom.get("background_solid",
                                                                   "#0B0B0D")))
    face = None
    if facecam_still and Path(facecam_still).exists():
        face = Image.open(facecam_still).convert("RGB")

    if geom.get("background") == "blurred_facecam" and face is not None:
        # Blur + dim, so the eye goes to the card and the frame still feels like
        # the same shot rather than a hard cut to a slide.
        bg = _cover(face, frame.w, frame.h).filter(
            ImageFilter.GaussianBlur(int(geom.get("background_blur", 28))))
        dim = float(geom.get("background_dim", 0.45))
        if dim > 0:
            bg = Image.blend(bg, Image.new("RGB", bg.size, (0, 0, 0)), dim)
        canvas.paste(bg, (0, 0))

    shot = Image.open(screenshot_png).convert("RGB")
    if geom.get("fit") == "cover":
        canvas.paste(_cover(shot, frame.w, frame.h), (0, 0))
    else:
        card_img = Image.new("RGB", (card.w, card.h),
                             _hex_rgb(geom.get("card_background", "#FFFFFF")))
        card_img.paste(shot.resize((image.w, image.h), Image.LANCZOS),
                       (image.x - card.x, image.y - card.y))
        radius = int(geom.get("card_radius_px", 0))
        canvas.paste(card_img, (card.x, card.y),
                     _rounded_mask((card.w, card.h), radius) if radius else None)

    if pip is not None and face is not None:
        pip_img = _cover(face, pip.w, pip.h)
        radius = int(geom.get("pip_radius_px", 0))
        canvas.paste(pip_img, (pip.x, pip.y),
                     _rounded_mask((pip.w, pip.h), radius) if radius else None)

    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return out


def encode_cmd(frame_png: str | Path, out_mp4: str | Path, *, duration_s: float,
               fps: int, size: tuple[int, int], fade_in_s: float = 0.15,
               fade_out_s: float = 0.15, ken_burns: float = 0.03) -> list[str]:
    """The ffmpeg argv. Pure, so the tests can read it without ffmpeg installed."""
    w, h = size
    frames = max(1, int(round(duration_s * fps)))
    filters: list[str] = []
    if ken_burns and ken_burns > 0:
        # zoompan samples its input, so upscale first: zooming the delivery-size
        # frame directly is what makes Ken Burns look like it is stepping.
        filters.append(f"scale={w * 2}:{h * 2}:flags=lanczos")
        filters.append(
            f"zoompan=z='min(1+{ken_burns:.4f}*on/{frames},{1 + ken_burns:.4f})'"
            f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":d={frames}:s={w}x{h}:fps={fps}")
    else:
        filters.append(f"scale={w}:{h}")
    if fade_in_s > 0:
        filters.append(f"fade=t=in:st=0:d={fade_in_s:.3f}")
    if fade_out_s > 0:
        filters.append(f"fade=t=out:st={max(0.0, duration_s - fade_out_s):.3f}"
                       f":d={fade_out_s:.3f}")
    filters.append("format=yuv420p")

    # zoompan emits d frames PER INPUT frame, so it gets the still once; a looped
    # input made a 4.5 s insert encode as minutes of video.
    src = (["-i", str(frame_png)] if ken_burns and ken_burns > 0 else
           ["-loop", "1", "-framerate", str(fps), "-t", f"{duration_s:.3f}", "-i", str(frame_png)])
    return ["ffmpeg", "-y", *src,
            "-vf", ",".join(filters), "-an",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-r", str(fps), str(out_mp4)]


def build_overlay(screenshot_png: str | Path, out_mp4: str | Path, *,
                  facecam_still: str | Path | None = None, aspect: str | None = None,
                  layout: str | None = None, layout_cfg: dict | None = None,
                  duration_s: float | None = None, image_size: tuple[int, int] | None = None,
                  runner=subprocess.run) -> dict:
    """Screenshot PNG -> `overlay.mp4` at the aspect's resolution and fps."""
    _key, acfg = resolve_aspect(aspect, layout_cfg if layout_cfg
                                and "aspects" in layout_cfg else None)
    if image_size is None:
        Image, _d, _f = _pil()
        with Image.open(screenshot_png) as im:
            image_size = im.size
    geom = compute_geometry(image_size, acfg, layout_cfg, layout=layout)

    timing = geom["timing"]
    dur = float(duration_s if duration_s is not None
                else timing.get("default_duration_s", 4.5))
    dur = max(float(timing.get("min_duration_s", 2.5)),
              min(float(timing.get("max_duration_s", 8.0)), dur))

    out = Path(out_mp4)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame_png = out.with_name("overlay_frame.png")
    if shutil.which("ffmpeg") is None and runner is subprocess.run:
        raise RuntimeError("ffmpeg is not installed; it is required to encode overlays")
    if geom["layout"] in ("float", "cutout"):
        # Alpha survives only in a codec that has it: qtrle in .mov. render.py's
        # overlay filter honours the alpha plane, so the live video shows through.
        out = out.with_suffix(".mov")
        render_float_frame(screenshot_png, frame_png, geom)
        fi, fo = float(timing.get("fade_in_s", 0.15)), float(timing.get("fade_out_s", 0.15))
        cmd = ["ffmpeg", "-y", "-loop", "1", "-framerate", str(geom["fps"]),
               "-t", f"{dur:.3f}", "-i", str(frame_png), "-vf",
               f"format=rgba,fade=t=in:st=0:d={fi:.3f}:alpha=1,"
               f"fade=t=out:st={max(0.0, dur - fo):.3f}:d={fo:.3f}:alpha=1",
               "-an", "-c:v", "qtrle", "-pix_fmt", "argb", "-r", str(geom["fps"]), str(out)]
        proc = runner(cmd, capture_output=True, text=True)
        if getattr(proc, "returncode", 1) != 0:
            raise RuntimeError(f"ffmpeg failed encoding {out}: "
                               f"{(getattr(proc, 'stderr', '') or '')[-400:]}")
        return {"file": str(out), "frame": str(frame_png), "duration_s": dur,
                "geometry": geometry_to_dict(geom), "command": cmd}
    render_frame(screenshot_png, frame_png, geom, facecam_still)

    cmd = encode_cmd(frame_png, out, duration_s=dur, fps=geom["fps"],
                     size=(geom["frame"].w, geom["frame"].h),
                     fade_in_s=float(timing.get("fade_in_s", 0.15)),
                     fade_out_s=float(timing.get("fade_out_s", 0.15)),
                     ken_burns=float(timing.get("ken_burns_zoom", 0.03)))
    proc = runner(cmd, capture_output=True, text=True)
    if getattr(proc, "returncode", 1) != 0:
        raise RuntimeError(f"ffmpeg failed encoding {out}: "
                           f"{(getattr(proc, 'stderr', '') or '')[-400:]}")

    return {"file": str(out), "frame": str(frame_png), "duration_s": dur,
            "geometry": geometry_to_dict(geom), "command": cmd}


def extract_facecam_still(video: str | Path, at_s: float, out_png: str | Path,
                          *, runner=subprocess.run) -> Path:
    """One frame of the facecam, used for the blurred background and the PiP."""
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-ss", f"{max(0.0, at_s):.3f}", "-i", str(video),
           "-frames:v", "1", str(out)]
    proc = runner(cmd, capture_output=True, text=True)
    if getattr(proc, "returncode", 1) != 0:
        raise RuntimeError(f"ffmpeg could not grab a facecam still at {at_s:.2f}s")
    return out


# --------------------------------------------------------------------------
# EDL placement


def overlay_for_slot(slot_id: str, meta: dict, *, trigger_time_output: float,
                     total_duration_s: float, file: str | None = None,
                     duration_s: float | None = None, layout_cfg: dict | None = None,
                     layout: str | None = None, aspect: str | None = None) -> Overlay:
    """The EDL entry for a verified slot, on the OUTPUT timeline.

    `start_in_output = trigger_time_output - lead_in_s`, clamped so the clip
    neither starts before the cut nor runs past its end -- render.py shifts PTS
    to that start, and an overlay hanging off either end of the timeline is an
    EDL the validator rejects.

    Refuses unverified slots outright (Zeta hard rule 12): the drop happens here
    as well as in the validator, so a caller that skips validation still cannot
    ship one.
    """
    if not meta.get("verified"):
        raise ValueError(
            f"{slot_id}: refusing to build an overlay for an unverified slot "
            f"({meta.get('dropped_reason') or 'no verification recorded'})")
    if not meta.get("url"):
        raise ValueError(f"{slot_id}: refusing to build an overlay with no source url")

    inserts = _inserts_block(layout_cfg)
    timing = inserts.get("timing") or {}
    lead_in = float(timing.get("lead_in_s", 0.3))
    dur = float(duration_s if duration_s is not None
                else meta.get("duration_s", timing.get("default_duration_s", 4.5)))
    dur = max(float(timing.get("min_duration_s", 2.5)),
              min(float(timing.get("max_duration_s", 8.0)), dur))
    dur = min(dur, max(0.0, total_duration_s))
    if dur <= 0:
        raise ValueError(f"{slot_id}: no room on a {total_duration_s:.2f}s timeline")

    start = max(0.0, float(trigger_time_output) - lead_in)
    start = min(start, max(0.0, total_duration_s - dur))

    layout_name = layout or meta.get("layout") or inserts.get("default_layout", "fit_card")
    overlay_meta = {
        "slot_id": slot_id,
        "claim": meta.get("claim", ""),
        "story": meta.get("story", ""),
        "url": meta.get("url"),
        "source_type": meta.get("source_type", "unknown"),
        "verified": True,
        "evidence": meta.get("evidence") or
                    ((meta.get("verification") or {}).get("final") or {}).get("evidence", ""),
        "trigger_word": (meta.get("claim_detail") or {}).get("trigger_word")
                        or meta.get("trigger_word"),
        "trigger_time_output": round(float(trigger_time_output), 3),
        "layout": layout_name,
    }
    if aspect:
        overlay_meta["aspect"] = aspect
    if meta.get("capture_source_type"):
        overlay_meta["capture"] = meta["capture_source_type"]
    if meta.get("source_origin"):
        overlay_meta["source_origin"] = meta["source_origin"]

    return Overlay(file=file or meta.get("overlay") or overlay_file(slot_id, layout_name),
                   start_in_output=start, duration=dur, meta=overlay_meta)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build an overlay clip (or just its geometry) for a slot")
    ap.add_argument("slot", help="slot directory holding shot.png and meta.json")
    ap.add_argument("--facecam", help="facecam still PNG for the background and PiP")
    ap.add_argument("--aspect", help="output aspect (default: configs/layout.yaml)")
    ap.add_argument("--layout", help="fit_card | split | fullframe_pip")
    ap.add_argument("--duration", type=float, help="clip duration in seconds")
    ap.add_argument("--image-size", help="WxH, to compute geometry without Pillow")
    ap.add_argument("--geometry", action="store_true",
                    help="print the computed geometry and exit (no render)")
    args = ap.parse_args()

    d = Path(args.slot)
    meta_path = d / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    shot = Path(meta.get("image") or (d / "shot.png"))

    if args.geometry:
        if args.image_size:
            size = tuple(int(v) for v in args.image_size.lower().split("x"))
        else:
            from PIL import Image  # lazy: --image-size avoids needing it
            with Image.open(shot) as im:
                size = im.size
        _key, acfg = resolve_aspect(args.aspect)
        print(json.dumps(geometry_to_dict(
            compute_geometry(size, acfg, layout=args.layout)), indent=1))
        return

    out = build_overlay(shot, d / "overlay.mp4", facecam_still=args.facecam,
                        aspect=args.aspect, layout=args.layout, duration_s=args.duration)
    # Stored EDL-relative, not absolute, so the edit directory stays portable.
    meta["overlay"] = overlay_file(d.name)
    meta["geometry"] = out["geometry"]
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("file", "duration_s")}, indent=1))


if __name__ == "__main__":
    main()
