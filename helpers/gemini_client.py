"""One door to the model.

Every LLM call in this project goes through `LLM` so that retries, JSON-schema
enforcement, response caching and offline mocking exist once. Modules never
import `google.genai` directly.

Offline / CI:
    ZETA_LLM_MOCK=1                 -> refuse to call the network
    ZETA_LLM_FIXTURES=<dir>         -> serve responses from <dir>/<sha1>.json
    LLM.register_mock(fn)           -> in-process handler (used by the tests)

The cache is keyed by the full request (model + prompt + files + schema), so a
re-run of `zeta plan` on an unchanged transcript costs nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from .env import load_env

DEFAULT_TEXT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_VISION_MODEL = "gemini-3.5-flash-lite"

_MOCK_HANDLER: Callable[[dict], Any] | None = None


class LLMError(RuntimeError):
    pass


class LLMResponseError(LLMError):
    """The model answered and the answer was not usable.

    Separate from a transport failure because retrying it is pure cost: the same
    request returns the same unusable shape four times, each after a longer
    backoff, and the real reason arrives minutes late.
    """


class LLMUnavailable(LLMError):
    """No API key, or mock mode is on and nothing answered the request."""


def register_mock(fn: Callable[[dict], Any] | None) -> None:
    """Install an in-process handler: `fn(request_dict) -> str | dict`."""
    global _MOCK_HANDLER
    _MOCK_HANDLER = fn


def _request_key(req: dict) -> str:
    blob = json.dumps(req, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


@dataclass
class LLM:
    """Thin, retrying, cacheable Gemini wrapper."""

    api_key: str | None = None
    model: str = DEFAULT_TEXT_MODEL
    fallback_model: str | None = None
    cache_dir: Path | None = None
    max_retries: int = 4
    timeout_s: float = 120.0
    _client: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.api_key is None:
            load_env()
            self.api_key = os.environ.get("GEMINI_API_KEY") or None
        if self.cache_dir:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

    # -- availability -------------------------------------------------------
    @property
    def mock_mode(self) -> bool:
        return os.environ.get("ZETA_LLM_MOCK", "").lower() in ("1", "true", "yes")

    @property
    def available(self) -> bool:
        return bool(self.api_key) or self.mock_mode or _MOCK_HANDLER is not None

    def require(self) -> None:
        if not self.available:
            raise LLMUnavailable(
                "GEMINI_API_KEY is not set. Put it in the environment or in .env "
                "at the repo root (see .env.example). Set ZETA_LLM_MOCK=1 to run "
                "the offline paths instead.")

    def _genai(self):
        if self._client is None:
            try:
                from google import genai  # lazy: only real calls pay for it
            except ImportError as exc:  # pragma: no cover
                raise LLMUnavailable(
                    "google-genai is not installed: uv pip install -e '.[llm]'") from exc
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    # -- cache --------------------------------------------------------------
    def _cached(self, key: str) -> Any | None:
        for d in filter(None, [self.cache_dir, os.environ.get("ZETA_LLM_FIXTURES")]):
            p = Path(d) / f"{key}.json"
            if p.exists():
                return json.loads(p.read_text(encoding="utf-8")).get("response")
        return None

    def _store(self, key: str, req: dict, response: Any) -> None:
        if not self.cache_dir:
            return
        p = Path(self.cache_dir) / f"{key}.json"
        p.write_text(json.dumps({"request": req, "response": response},
                                ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    # -- core ---------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        schema: dict | None = None,
        model: str | None = None,
        images: Sequence[str | Path] = (),
        files: Sequence[str | Path] = (),
        temperature: float = 0.0,
        tools: Sequence[str] = (),
        use_cache: bool = True,
    ) -> Any:
        """Return parsed JSON when `schema` is given, otherwise raw text."""
        req = {
            "model": model or self.model,
            "system": system,
            "prompt": prompt,
            "schema": schema,
            "images": [str(Path(p).name) for p in images],
            "files": [str(Path(p).name) for p in files],
            "temperature": temperature,
            "tools": list(tools),
        }
        key = _request_key(req)

        if use_cache:
            hit = self._cached(key)
            if hit is not None:
                return hit

        if _MOCK_HANDLER is not None:
            out = _MOCK_HANDLER(req)
            if out is not None:
                return self._coerce(out, schema)

        if self.mock_mode:
            raise LLMUnavailable(
                f"ZETA_LLM_MOCK is on and no fixture answered request {key[:12]} "
                f"(model={req['model']})")

        self.require()
        raw = self._call_with_retry(req, images=images, files=files)
        out = self._coerce(raw, schema)
        if use_cache:
            self._store(key, req, out)
        return out

    def _coerce(self, raw: Any, schema: dict | None) -> Any:
        if schema is None or not isinstance(raw, str):
            return raw
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            start, end = raw.find("{"), raw.rfind("}")
            if start != -1 and end > start:
                return json.loads(raw[start:end + 1])
            raise LLMError(f"model did not return JSON: {raw[:200]!r}")

    def _call_with_retry(self, req: dict, images: Sequence, files: Sequence) -> str:
        models = [req["model"]] + ([self.fallback_model] if self.fallback_model else [])
        last: Exception | None = None
        for model_name in models:
            for attempt in range(self.max_retries):
                try:
                    return self._call_once(model_name, req, images, files)
                except LLMResponseError as exc:
                    # The model replied; a backoff changes nothing. Move on to
                    # the fallback model, which is a different model and may
                    # answer in a shape we can read.
                    last = exc
                    break
                except Exception as exc:  # transient: rate limit, 5xx, timeout
                    last = exc
                    if attempt == self.max_retries - 1:
                        break
                    delay = (2 ** attempt) + random.uniform(0, 0.5)
                    time.sleep(delay)
        raise LLMError(f"all attempts failed for {models}: {last}") from last

    @staticmethod
    def _response_text(resp: Any) -> str | None:
        """Pull the answer out, whatever part shape the model used.

        `resp.text` is a convenience over text parts only. The transcription
        models answer with an `audio_transcription` part instead, so a reader
        that only knows `.text` sees a valid response as an empty one.
        """
        text = getattr(resp, "text", None)
        if text:
            return text
        for candidate in (getattr(resp, "candidates", None) or []):
            content = getattr(candidate, "content", None)
            for part in (getattr(content, "parts", None) or []):
                transcription = getattr(part, "audio_transcription", None)
                if transcription is not None:
                    value = getattr(transcription, "text", None) or (
                        transcription.get("text") if isinstance(transcription, dict) else None)
                    if value:
                        return value
                if getattr(part, "text", None):
                    return part.text
        return None

    def _call_once(self, model_name: str, req: dict, images: Sequence, files: Sequence) -> str:
        client = self._genai()
        parts: list[Any] = [req["prompt"]]
        for p in list(images) + list(files):
            parts.append(client.files.upload(file=str(p)))

        cfg: dict[str, Any] = {"temperature": req["temperature"]}
        if req["system"]:
            cfg["system_instruction"] = req["system"]
        if req["schema"]:
            cfg["response_mime_type"] = "application/json"
            cfg["response_schema"] = req["schema"]
        if req["tools"]:
            # e.g. ("google_search",) for grounded source resolution
            cfg["tools"] = [{t: {}} for t in req["tools"]]

        resp = client.models.generate_content(model=model_name, contents=parts, config=cfg)
        text = self._response_text(resp)
        if not text:
            finish = None
            for candidate in (getattr(resp, "candidates", None) or []):
                finish = getattr(candidate, "finish_reason", None) or finish
            raise LLMResponseError(
                f"no usable content in the response from {model_name}"
                + (f" (finish_reason={finish})" if finish else ""))
        return text

    # -- conveniences -------------------------------------------------------
    def json(self, prompt: str, schema: dict, **kw: Any) -> Any:
        return self.generate(prompt, schema=schema, **kw)

    def text(self, prompt: str, **kw: Any) -> str:
        out = self.generate(prompt, **kw)
        return out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)

    def vision_json(self, prompt: str, image: str | Path, schema: dict, **kw: Any) -> Any:
        kw.setdefault("model", DEFAULT_VISION_MODEL)
        return self.generate(prompt, schema=schema, images=[image], **kw)


def default_llm(cfg: dict | None = None, cache_dir: Path | None = None) -> LLM:
    """Build an `LLM` from a `configs/transcribe.yaml`-shaped backend block."""
    cfg = cfg or {}
    return LLM(
        model=cfg.get("model", DEFAULT_TEXT_MODEL),
        fallback_model=cfg.get("fallback_model"),
        cache_dir=cache_dir,
    )
