"""Shared test setup.

The repo is not installed during tests, and `helpers/render.py` is vendored
code that expects its own directory on `sys.path`, so both the repo root and
`helpers/` go on the path here rather than in every test file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (REPO_ROOT, REPO_ROOT / "helpers"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from helpers import gemini_client  # noqa: E402
from helpers.edl import EDL, Range  # noqa: E402
from helpers.words import Word, WordsDoc  # noqa: E402


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """No test may reach the network, with or without a key in the environment."""
    monkeypatch.setenv("ZETA_LLM_MOCK", "1")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    gemini_client.register_mock(None)
    yield
    gemini_client.register_mock(None)


@pytest.fixture
def mock_llm():
    """`mock_llm({"substring in prompt": response})` -> an LLM serving them."""

    def _install(routes: dict[str, object]):
        def handler(req: dict):
            for needle, response in routes.items():
                if needle in (req.get("prompt") or "") or needle in (req.get("system") or ""):
                    return response
            return None

        gemini_client.register_mock(handler)
        return gemini_client.LLM(cache_dir=None)

    return _install


def make_doc(text: str, *, start: float = 0.0, word_s: float = 0.40,
             gap_s: float = 0.10, name: str = "raw01") -> WordsDoc:
    """A WordsDoc with evenly spaced words. `word_s` speech, `gap_s` silence."""
    words, t = [], start
    for tok in text.split():
        words.append(Word(word=tok, start=round(t, 3), end=round(t + word_s, 3), score=0.9))
        t += word_s + gap_s
    return WordsDoc(words=words, source={"name": name, "path": f"/tmp/{name}.mp4"},
                    backend="test", aligner="test")


@pytest.fixture
def doc_factory():
    return make_doc


@pytest.fixture
def simple_edl(tmp_path):
    """Two ranges from one source: 2.0-6.0 then 10.0-14.0 (8 s of output)."""
    return EDL(sources={"raw01": str(tmp_path / "raw01.mp4")},
               ranges=[Range("raw01", 2.0, 6.0), Range("raw01", 10.0, 14.0)])
