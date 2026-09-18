"""`.env` has to actually be read, or the README is a lie."""

from __future__ import annotations

import os

import pytest

from helpers import env


def test_parses_comments_quotes_and_export():
    parsed = env.parse_env(
        "# a comment\n\nGEMINI_API_KEY=abc123\n"
        "export COHERE_API_KEY='xyz'\n"
        'QUOTED="with spaces"\nnot a pair\n')
    assert parsed == {"GEMINI_API_KEY": "abc123", "COHERE_API_KEY": "xyz",
                      "QUOTED": "with spaces"}


def test_a_real_environment_variable_wins(tmp_path, monkeypatch):
    """An exported key or one injected by a cloud environment is more specific
    than a file on disk; silently overriding it would be very hard to debug."""
    dotenv = tmp_path / ".env"
    dotenv.write_text("GEMINI_API_KEY=from_file\n", encoding="utf-8")
    monkeypatch.setenv("GEMINI_API_KEY", "from_environment")

    env.load_env(dotenv, force=True)
    assert os.environ["GEMINI_API_KEY"] == "from_environment"


def test_the_file_fills_an_unset_key(tmp_path, monkeypatch):
    dotenv = tmp_path / ".env"
    dotenv.write_text("GEMINI_API_KEY=from_file\n", encoding="utf-8")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    applied = env.load_env(dotenv, force=True)
    assert applied == {"GEMINI_API_KEY": "from_file"}
    assert os.environ["GEMINI_API_KEY"] == "from_file"


def test_a_missing_file_is_not_an_error(tmp_path):
    assert env.load_env(tmp_path / "nope.env", force=True) == {}


def test_the_llm_picks_the_key_up_from_the_file(tmp_path, monkeypatch):
    from helpers.gemini_client import LLM

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(env, "_LOADED", False)
    monkeypatch.setattr("helpers.env.REPO_ROOT", tmp_path)
    (tmp_path / ".env").write_text("GEMINI_API_KEY=k-from-dotenv\n", encoding="utf-8")

    assert LLM().api_key == "k-from-dotenv"


class _Part:
    def __init__(self, text=None, audio_transcription=None):
        self.text = text
        self.audio_transcription = audio_transcription


class _Resp:
    def __init__(self, text=None, parts=(), finish_reason="STOP"):
        self.text = text
        candidate = type("C", (), {"content": type("Ct", (), {"parts": list(parts)})(),
                                   "finish_reason": finish_reason})()
        self.candidates = [candidate]


class _FakeClient:
    """Stands in for google.genai's Client; counts calls so retries are visible."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0
        self.models = self
        self.files = self

    def generate_content(self, **kw):
        self.calls += 1
        out = self._responses[min(self.calls - 1, len(self._responses) - 1)]
        if isinstance(out, Exception):
            raise out
        return out


def test_transcription_models_answer_in_a_part_that_is_not_text(monkeypatch):
    """gemini_transcribe replies with an audio_transcription part, not `.text`.

    A reader that only knows `resp.text` sees a perfectly good transcript as an
    empty response, and the whole backend looks broken.
    """
    from helpers.gemini_client import LLM

    monkeypatch.delenv("ZETA_LLM_MOCK", raising=False)
    resp = _Resp(text=None, parts=[_Part(audio_transcription={"text": "مرحبا بكم في زيتا"})])
    llm = LLM(api_key="k", _client=_FakeClient([resp]))

    assert llm.generate("transcribe", use_cache=False) == "مرحبا بكم في زيتا"


def test_an_unusable_answer_is_not_retried(monkeypatch):
    """Four backoffs on a non-transient failure only delay the real reason."""
    from helpers.gemini_client import LLM, LLMError

    monkeypatch.delenv("ZETA_LLM_MOCK", raising=False)
    slept = []
    monkeypatch.setattr("helpers.gemini_client.time.sleep", slept.append)
    client = _FakeClient([_Resp(text=None, parts=[], finish_reason="SAFETY")])
    llm = LLM(api_key="k", _client=client)

    with pytest.raises(LLMError, match="no usable content"):
        llm.generate("hello", use_cache=False)
    assert client.calls == 1, "an unusable answer must not be retried"
    assert slept == [], "and must not back off"


def test_a_transient_failure_still_retries(monkeypatch):
    from helpers.gemini_client import LLM

    monkeypatch.delenv("ZETA_LLM_MOCK", raising=False)
    slept = []
    monkeypatch.setattr("helpers.gemini_client.time.sleep", slept.append)
    client = _FakeClient([RuntimeError("503 overloaded"), _Resp(text="OK")])
    llm = LLM(api_key="k", _client=client)

    assert llm.generate("hello", use_cache=False) == "OK"
    assert client.calls == 2 and len(slept) == 1


def test_the_fallback_model_gets_a_turn_after_an_unusable_answer(monkeypatch):
    from helpers.gemini_client import LLM

    monkeypatch.delenv("ZETA_LLM_MOCK", raising=False)
    monkeypatch.setattr("helpers.gemini_client.time.sleep", lambda _s: None)
    client = _FakeClient([_Resp(text=None, parts=[]), _Resp(text="OK")])
    llm = LLM(api_key="k", fallback_model="gemini-3.1-flash-lite", _client=client)

    assert llm.generate("hello", use_cache=False) == "OK"
    assert client.calls == 2
