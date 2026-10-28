"""YouTube facecam thumbnail, Ali's approved layout (facecam2, 2026-10-12).

Face on the right third, headline stacked on a dark left gradient, red + white
only, Zeta logo small bottom-right above YouTube's duration badge.

    python scripts/thumbnail.py VIDEO TIME_S FACE_X FACE_Y OUT.jpg "NO|r" "CAPITAL" "=|r" "NO|r" "FREEDOM"

FACE_X/FACE_Y: face centre as a fraction of the source frame. "|r" = red word.
"""
import subprocess, sys, tempfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageFont

LOGO = Path(__file__).resolve().parent.parent / "assets/zeta_logo.png"  # keyed from the intro at 0:21
FONT = "/System/Library/Fonts/Supplemental/Arial Black.ttf"
RED, WHITE = (225, 30, 30), "white"


def frame(video, t):
    out = Path(tempfile.mkdtemp()) / "f.png"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(t), "-i", video, "-frames:v", "1", str(out)], check=True)
    return Image.open(out).convert("RGB")


def face_crop(im, fx, fy, z=1.55):
    # Zoom so the face lands on the right third (x 0.70) with headroom.
    W, H = im.size; w, h = W / z, W / z * 9 / 16
    cx = min(max(fx - 0.20 / z, 0.5 / z), 1 - 0.5 / z); cy = min(max(fy + 0.12 / z, 0.5 / z), 1 - 0.5 / z)
    return im.crop((int(cx*W - w/2), int(cy*H - h/2), int(cx*W + w/2), int(cy*H + h/2))).resize((1280, 720), Image.LANCZOS)


def make(video, t, fx, fy, out, lines):
    im = face_crop(frame(video, t), fx, fy)
    g = Image.linear_gradient("L").rotate(90).resize((1280, 720))  # 0 at left -> 255 at right
    g = g.point(lambda v: int(235 * min(1, max(0, 1.25 - v / 255 * 1280 / 620))))
    im.paste(Image.new("RGB", im.size, (8, 8, 12)), (0, 0), g)
    d = ImageDraw.Draw(im); f = ImageFont.truetype(FONT, 100)
    for i, s in enumerate(lines):
        word, _, col = s.partition("|")
        d.text((80, 40 + i * 105), word, font=f, fill=RED if col == "r" else WHITE, stroke_width=8, stroke_fill="black")
    lg = Image.open(LOGO); lg = lg.resize((140, int(140 * lg.height / lg.width)), Image.LANCZOS)
    pos = (1280 - 40 - lg.width, 720 - 80 - lg.height)
    sh = Image.new("L", im.size); sh.paste(lg.getchannel("A").point(lambda v: min(255, v * 4)), pos)
    sh = sh.filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.GaussianBlur(12))  # shadow: cream logo on light clothes
    im.paste(Image.new("RGB", im.size, (0, 0, 0)), (0, 0), sh); im.paste(lg, pos, lg)
    im.save(out, quality=92)


if __name__ == "__main__":
    if len(sys.argv) < 7 or sys.argv[1] in ("-h", "--help"):
        sys.exit(__doc__)
    v, t, fx, fy, out, *lines = sys.argv[1:]
    make(v, float(t), float(fx), float(fy), out, lines)
