"""`.env` has to actually be read, or the README is a lie."""

from __future__ import annotations

import os

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
