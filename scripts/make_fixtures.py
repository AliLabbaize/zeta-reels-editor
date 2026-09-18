"""Generate the test fixtures: a 30 s take with known cuts and a known insert.

Real Zeta footage is not in the repo, so the fixtures are synthesised from a
script: each "word" is a tone burst at a known time and each silence is a real
silence of a known length. That makes the cut engine's rules testable as
arithmetic - a 120 ms gap really is 120 ms - which is the part that matters.
What it deliberately does NOT test is anything acoustic or facial: alignment
quality and the face-presence check in learn mode need real footage.

The media files are gitignored; the JSON fixtures are committed. Regenerate:

    python scripts/make_fixtures.py
"""

from __future__ import annotations

import json
import math
import struct
import subprocess
import sys
import wave
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
SAMPLE_RATE = 16000

# (text, word_seconds, silence_after_seconds). The silences are chosen to hit
# every branch of the cut-point rule: >= 400 ms is a clean cut point, 150-400 ms
# needs a visual check, < 150 ms is never a cut point.
SCRIPT: list[tuple[str, float, float]] = [
    ("salam",      0.45, 0.12),   # intro, tight
    ("a",          0.20, 0.10),
    ("khoya",      0.50, 0.85),   # clean boundary after the greeting
    ("euh",        0.30, 0.16),   # filler
    ("OpenAI",     0.70, 0.14),
    ("جابت",       0.45, 0.10),
    ("مليار",      0.55, 0.45),   # clean boundary
    ("دولار",      0.60, 0.20),   # 200 ms: needs a visual check
    ("euh",        0.28, 0.13),   # filler
    ("donc",       0.32, 0.11),   # filler
    ("السوق",      0.50, 0.10),
    ("طلع",        0.40, 0.55),   # clean boundary
    ("بزاف",       0.45, 0.13),
    ("Nvidia",     0.65, 0.12),
    ("زادت",       0.45, 0.42),   # clean boundary
    ("عشرة",       0.42, 0.11),
    ("بالمية",     0.60, 0.90),   # clean boundary before the outro
    ("صافي",       0.40, 0.14),
    ("شكرا",       0.50, 1.20),   # outro tail
]

# What a good edit removes: the three fillers. Learn-mode fixtures also drop the
# outro tail, so a raw/published pair has an intro/outro trim to recover.
FILLER_INDICES = [3, 8, 9]
PUBLISHED_DROP = FILLER_INDICES + [17, 18]

# The insert: a source screenshot for the "milliard dollars" claim, triggered on
# the first mention of OpenAI.
INSERT = {
    "slot_id": "slot_01",
    "trigger_word_index": 4,
    "claim": "OpenAI raised one billion dollars",
    "entity": "OpenAI",
    "url": "https://openai.com/index/fixture-announcement/",
    "duration_s": 4.0,
}


def build_words() -> tuple[list[dict], float]:
    words, t = [], 1.00   # 1 s of room tone before the first word
    for text, dur, gap in SCRIPT:
        words.append({"word": text, "display": text,
                      "start": round(t, 3), "end": round(t + dur, 3),
                      "score": 0.93, "src": "fixture"})
        t += dur + gap
    return words, round(t, 3)


def write_tone_wav(words: list[dict], total_s: float, path: Path) -> None:
    """One sine burst per word, silence elsewhere, with 5 ms edge ramps.

    The ramps matter: a hard-edged burst has broadband click energy that the
    self-eval waveform check would read as an audio pop at a cut.
    """
    n = int(total_s * SAMPLE_RATE)
    samples = [0.0] * n
    for i, w in enumerate(words):
        freq = 180.0 + 40.0 * (i % 5)          # vary pitch so bursts are distinguishable
        a, b = int(w["start"] * SAMPLE_RATE), int(w["end"] * SAMPLE_RATE)
        ramp = int(0.005 * SAMPLE_RATE)
        for k in range(a, min(b, n)):
            env = 1.0
            if k - a < ramp:
                env = (k - a) / ramp
            elif b - k < ramp:
                env = max(0.0, (b - k) / ramp)
            samples[k] = 0.35 * env * math.sin(2 * math.pi * freq * (k - a) / SAMPLE_RATE)

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(b"".join(struct.pack("<h", int(s * 32767)) for s in samples))


def have_ffmpeg() -> bool:
    return subprocess.run(["which", "ffmpeg"], capture_output=True).returncode == 0


def mux_video(wav: Path, out: Path, total_s: float, insert: tuple[float, float] | None) -> None:
    """A 1080x1920 clip: moving test pattern, optionally covered by a flat card.

    The card stands in for a full-frame screenshot insert so scene detection has
    a real cut to find. It is flat on purpose: PySceneDetect should see the
    content change, not a dissolve.
    """
    vf = "drawtext=text='ZETA FIXTURE':fontcolor=white:fontsize=48:x=(w-tw)/2:y=120"
    if insert:
        a, b = insert
        vf += (f",drawbox=x=0:y=0:w=1080:h=1920:color=white@1.0:t=fill:"
               f"enable='between(t,{a:.3f},{b:.3f})'")
    cmd = [
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"testsrc2=size=1080x1920:rate=30:duration={total_s:.3f}",
        "-i", str(wav), "-vf", vf,
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-shortest", str(out),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def main() -> int:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    words, total = build_words()

    raw_doc = {
        "schema": "zeta.words.v1",
        "source": {"name": "raw01", "path": str(FIXTURES / "raw01.mp4"), "duration_s": total},
        "language": "ary", "backend": "fixture", "aligner": "fixture",
        "words": words,
        "meta": {"script": [list(s) for s in SCRIPT]},
    }
    (FIXTURES / "raw01.words.json").write_text(
        json.dumps(raw_doc, ensure_ascii=False, indent=1), encoding="utf-8")

    kept = [w for i, w in enumerate(words) if i not in FILLER_INDICES]
    published = [w for i, w in enumerate(words) if i not in PUBLISHED_DROP]
    expected = {
        "kept_text": " ".join(w["display"] for w in kept),
        "published_text": " ".join(w["display"] for w in published),
        "filler_indices": FILLER_INDICES,
        "published_drop_indices": PUBLISHED_DROP,
        "cut_ratio_filler_only": round(len(FILLER_INDICES) / len(words), 4),
        "insert": INSERT | {"trigger_time_source": words[INSERT["trigger_word_index"]]["start"]},
        "total_duration_s": total,
    }
    (FIXTURES / "expected.json").write_text(
        json.dumps(expected, ensure_ascii=False, indent=1), encoding="utf-8")
    (FIXTURES / "links.txt").write_text(INSERT["url"] + "\n", encoding="utf-8")

    if not have_ffmpeg():
        print("ffmpeg not found: wrote JSON fixtures only")
        return 0

    wav = FIXTURES / "raw01.wav"
    write_tone_wav(words, total, wav)
    mux_video(wav, FIXTURES / "raw01.mp4", total, insert=None)

    # The published cut: the same take with the dropped spans removed and a
    # four second insert card over the OpenAI mention.
    pub_wav = FIXTURES / "published01.wav"
    pub_words, t = [], 0.0
    for i, w in enumerate(words):
        if i in PUBLISHED_DROP:
            continue
        dur = w["end"] - w["start"]
        pub_words.append({**w, "start": round(t, 3), "end": round(t + dur, 3)})
        t += dur + 0.12
    write_tone_wav(pub_words, round(t, 3), pub_wav)
    ins_start = pub_words[2]["start"]
    mux_video(pub_wav, FIXTURES / "published01.mp4", round(t, 3),
              insert=(ins_start, ins_start + INSERT["duration_s"]))

    print(f"fixtures in {FIXTURES}:")
    for p in sorted(FIXTURES.iterdir()):
        print(f"  {p.name}  ({p.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
