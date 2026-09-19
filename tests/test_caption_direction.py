"""Mixed Darija/Latin caption lines must burn right to left.

Found on the first real take: libass laid "ليكم واحد part 2" out LTR, so the
line read backwards. This burns one karaoke line and checks where the words land.
"""
from __future__ import annotations

import shutil
import subprocess

import pytest

from helpers import captions
from helpers.captions import Cue, CueWord

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


def test_arabic_first_word_lands_right_of_the_latin_one(tmp_path):
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    style = captions.resolve_style("9:16")
    cue = Cue(start=0.0, end=1.0, words=[CueWord("السلام", 0.0, 0.5),
                                         CueWord("NVIDIA", 0.5, 1.0, lang="lat")])
    ass = tmp_path / "c.ass"
    ass.write_text(captions.render_ass([cue], style), encoding="utf-8")
    png = tmp_path / "f.png"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
                    "-i", "color=black:s=1080x1920:d=1", "-vf", f"ass={ass}",
                    "-ss", "0.25", "-frames:v", "1", str(png)], check=True)

    rgb = np.asarray(Image.open(png).convert("RGB")).astype(int)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    # At 0.25 s the sung Arabic word is the highlight (yellow), NVIDIA still white.
    yellow = (r > 180) & (g > 150) & (b < 90)
    white = (r > 200) & (g > 200) & (b > 200)
    assert yellow.any() and white.any(), "caption did not burn"
    assert np.nonzero(yellow)[1].mean() > np.nonzero(white)[1].mean(), \
        "Arabic word is left of the Latin one: the line was laid out LTR"
