"""Text-LLM adapter (Gemini). Returns schema-validated pydantic objects.

All provider specifics + retry/backoff + JSON re-ask live here ONCE; agents call
``complete_json`` and get a typed object back. In mock mode (no API key) the
agent-supplied ``mock_factory`` produces a deterministic object so the pipeline
runs offline.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError
import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.adapters.errors import PermanentProviderError, TransientProviderError
from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.observability import usage

log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


def _is_transient(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in ("429", "resource_exhausted", "rate", "503", "500", "unavailable"))


# Gemini 429s carry their own back-off hint in TWO spots of the error string:
# the human message ("Please retry in 3.017691049s") and the structured
# RetryInfo detail ("'retryDelay': '3s'"). Match either; .search returns the
# first (the precise message value), so we wait what the server actually asked.
_RETRY_AFTER_RE = re.compile(
    r"(?:retry in\s+|retryDelay['\"]?\s*[:=]\s*['\"]?)(\d+(?:\.\d+)?)\s*s",
    re.IGNORECASE,
)


def parse_retry_after(exc: Exception) -> float | None:
    """Pull the provider's suggested back-off (seconds) out of a 429, or None."""
    m = _RETRY_AFTER_RE.search(str(exc))
    if not m:
        return None
    try:
        return float(m.group(1))
    except (TypeError, ValueError):
        return None


def wait_retry_after(fallback: Callable[[object], float], *, cap: float) -> Callable[[object], float]:
    """tenacity ``wait`` strategy: honor a ``TransientProviderError.retry_after``
    hint (CAPPED) when the provider gave one, else defer to ``fallback``.

    Capping is load-bearing, not cosmetic: a DAILY-quota 429 (100 RPD) can carry
    a huge ``retryDelay`` — we must NOT sleep that long inside a running job, so
    we cap the honored wait low enough that a daily exhaustion exhausts retries
    quickly and the caller degrades (silent beat) instead of hanging for minutes.
    """

    def _wait(retry_state) -> float:
        outcome = getattr(retry_state, "outcome", None)
        exc = outcome.exception() if outcome is not None else None
        retry_after = getattr(exc, "retry_after", None)
        if retry_after is not None and retry_after >= 0:
            return min(float(retry_after), cap)
        return fallback(retry_state)

    return _wait


class LLMAdapter:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = None  # lazy

    @property
    def client(self):
        if self._client is None:
            from google import genai  # imported lazily so mock mode needs no key

            self._client = genai.Client(api_key=self.settings.gemini_api_key)
        return self._client

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        response_model: type[T],
        mock_factory: Callable[[str], T],
    ) -> T:
        if self.settings.mock_mode:
            log.info("LLMAdapter mock_mode -> %s", response_model.__name__)
            if self.settings.mock_token_usage:
                usage.record(total=self.settings.mock_token_usage)
            return mock_factory(user)

        provider = (self.settings.llm_provider or "gemini").strip().lower()
        if provider in {"deepseek", "deepseek-chat", "deepseek-reasoner"}:
            return self._complete_json_openai_compatible(
                base_url=self.settings.deepseek_base_url,
                api_key=self.settings.deepseek_api_key,
                model=self.settings.llm_model or self.settings.deepseek_model,
                system=system,
                user=user,
                response_model=response_model,
            )
        if provider in {"openai", "openai_compatible", "custom"}:
            return self._complete_json_openai_compatible(
                base_url=self.settings.openai_compat_base_url,
                api_key=self.settings.openai_compat_api_key,
                model=self.settings.llm_model or self.settings.openai_compat_model,
                system=system,
                user=user,
                response_model=response_model,
            )
        return self._complete_json_real(system, user, response_model)

    @retry(
        retry=retry_if_exception_type(TransientProviderError),
        stop=stop_after_attempt(3),
        wait=wait_retry_after(wait_exponential(multiplier=1, min=2, max=20), cap=20.0),
        reraise=True,
    )
    def _complete_json_openai_compatible(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        system: str,
        user: str,
        response_model: type[T],
    ) -> T:
        """Call DeepSeek/OpenAI-compatible chat completions and validate JSON."""
        if not base_url.strip() or not api_key.strip() or not model.strip():
            raise PermanentProviderError(
                f"LLM provider 配置不完整: base_url/model/key missing for {self.settings.llm_provider}"
            )
        schema = json.dumps(response_model.model_json_schema(), ensure_ascii=False)
        system_prompt = (
            f"{system}\n\n"
            "Return ONLY valid JSON matching this JSON Schema; no markdown fences:\n"
            f"{schema}"
        )
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }
        try:
            response = httpx.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=120.0,
            )
            if response.status_code == 429 or response.status_code >= 500:
                raise TransientProviderError(
                    f"{self.settings.llm_provider} HTTP {response.status_code}: {response.text[:400]}"
                )
            response.raise_for_status()
            body: dict[str, Any] = response.json()
            content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
            if isinstance(content, list):
                content = "".join(str(x.get("text", "")) if isinstance(x, dict) else str(x) for x in content)
            usage_body = body.get("usage") or {}
            usage.record(
                prompt=int(usage_body.get("prompt_tokens") or 0),
                output=int(usage_body.get("completion_tokens") or 0),
                total=int(usage_body.get("total_tokens") or 0) or None,
            )
            parsed = _extract_json_object(str(content))
            return response_model.model_validate(parsed)
        except TransientProviderError:
            raise
        except httpx.TimeoutException as exc:
            raise TransientProviderError(f"{self.settings.llm_provider} timeout: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            raise PermanentProviderError(
                f"{self.settings.llm_provider} HTTP {exc.response.status_code}: {exc.response.text[:500]}"
            ) from exc
        except (ValueError, KeyError, TypeError, ValidationError) as exc:
            raise PermanentProviderError(
                f"{self.settings.llm_provider} returned invalid {response_model.__name__}: {exc}"
            ) from exc
        except Exception as exc:  # noqa: BLE001 - normalize provider errors
            raise PermanentProviderError(f"{self.settings.llm_provider} request failed: {exc}") from exc

    @retry(
        retry=retry_if_exception_type(TransientProviderError),
        stop=stop_after_attempt(3),
        # Honor a 429's own retryDelay (capped) when present, else exponential.
        wait=wait_retry_after(wait_exponential(multiplier=1, min=2, max=20), cap=20.0),
        reraise=True,
    )
    def _complete_json_real(self, system: str, user: str, response_model: type[T]) -> T:
        try:
            resp = self.client.models.generate_content(
                model=self.settings.gemini_text_model,
                contents=f"{system}\n\n{user}",
                config={
                    "response_mime_type": "application/json",
                    "response_schema": response_model,
                },
            )
        except Exception as exc:  # noqa: BLE001 - normalize provider errors
            if _is_transient(exc):
                raise TransientProviderError(
                    str(exc), retry_after=parse_retry_after(exc)
                ) from exc
            raise PermanentProviderError(str(exc)) from exc

        usage.record_usage(getattr(resp, "usage_metadata", None))
        try:
            return response_model.model_validate_json(resp.text)
        except ValidationError as exc:
            # Bad JSON is permanent for this call; tenacity won't retry it.
            raise PermanentProviderError(
                f"LLM returned invalid {response_model.__name__}: {exc}"
            ) from exc


def _extract_json_object(text: str) -> dict:
    """Strip common markdown fences and recover the first JSON object."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


def available_llm_profiles(settings: Settings) -> list[dict[str, object]]:
    """Safe text-provider choices for the visual workbench; never returns keys."""
    return [
        {
            "id": "deepseek",
            "label": "DeepSeek（dsh 环境变量）",
            "models": [
                {"id": "deepseek-chat", "label": "deepseek-chat"},
                {"id": "deepseek-reasoner", "label": "deepseek-reasoner"},
            ],
            "configured": bool(settings.deepseek_api_key.strip()),
            "base_url": settings.deepseek_base_url,
        },
        {
            "id": "gemini",
            "label": "Gemini",
            "models": [{"id": settings.gemini_text_model, "label": settings.gemini_text_model}],
            "configured": bool(settings.gemini_api_key.strip()),
        },
        {
            "id": "openai_compatible",
            "label": "OpenAI-compatible（自定义）",
            "models": [{"id": settings.openai_compat_model or "custom-model", "label": "自定义模型"}],
            "configured": bool(settings.openai_compat_api_key.strip() and settings.openai_compat_base_url.strip()),
            "base_url": settings.openai_compat_base_url,
        },
        {"id": "mock", "label": "Mock（离线）", "models": [{"id": "mock", "label": "Mock"}], "configured": True},
    ]
