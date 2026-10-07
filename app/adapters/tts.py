"""Gemini TTS adapter — synthesize ONE narration clip per beat (caption).

The assembler voices each caption separately so every image can be held for
exactly its narration's length (audio-driven per-beat holds). That is 15-30 TTS
calls per job against a HARD 10 RPM / 100 RPD per-model quota, so:

    get_tts_limiter (capacity=1, ~8 RPM)  ──▶ paces calls UNDER the per-minute cap
    honor 429 retryDelay (capped ~20s)    ──▶ wait EXACTLY what Google asks, then retry

Only after ``tts_max_retries`` are exhausted does a beat degrade to silence (the
assembler's silence fallback). Returns WAV bytes (Gemini TTS emits raw PCM, which
we wrap). Mock mode (no key) returns a short silent WAV so the pipeline + tests
run offline.
"""

from __future__ import annotations

import io
import time
import wave

from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.adapters.errors import PermanentProviderError, TransientProviderError
from app.adapters.llm import _is_transient, parse_retry_after, wait_retry_after
from app.adapters.rate_limit import get_tts_limiter
from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.observability import usage

log = get_logger(__name__)

_TTS_RATE = 24000  # Gemini TTS PCM: 24 kHz, 16-bit, mono


class GeminiTTSAdapter:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self.settings.gemini_api_key)
        return self._client

    def synthesize(self, text: str) -> bytes:
        """Narration text -> WAV bytes. Silent WAV in mock mode / for empty text."""
        if self.settings.mock_mode or not text.strip():
            if self.settings.mock_mode and text.strip() and self.settings.mock_token_usage:
                usage.record(total=self.settings.mock_token_usage)
            return _silent_wav(1.0)
        retrying = Retrying(
            retry=retry_if_exception_type(TransientProviderError),
            stop=stop_after_attempt(max(1, self.settings.tts_max_retries)),
            wait=wait_retry_after(
                wait_exponential(multiplier=1, min=2, max=20),
                cap=self.settings.tts_retry_cap_seconds,
            ),
            reraise=True,
        )
        return retrying(self._synthesize_real, text)

    def _synthesize_real(self, text: str) -> bytes:
        from google.genai import types

        # RPM guard: pace TTS calls UNDER the 10 RPM per-model cap (capacity=1, no
        # burst) before every request — the proactive half of the fix.
        get_tts_limiter(self.settings.tts_rpm_limit).acquire()
        try:
            resp = self.client.models.generate_content(
                model=self.settings.gemini_tts_model,
                contents=text,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=self.settings.tts_voice
                            )
                        )
                    ),
                ),
            )
        except Exception as exc:  # noqa: BLE001 - normalize provider errors
            if _is_transient(exc):
                raise TransientProviderError(
                    str(exc), retry_after=parse_retry_after(exc)
                ) from exc
            raise PermanentProviderError(str(exc)) from exc

        usage.record_usage(getattr(resp, "usage_metadata", None))
        pcm = _extract_audio(resp)
        if pcm is None:
            raise PermanentProviderError("Gemini TTS returned no audio.")
        return _pcm_to_wav(pcm)


def _extract_audio(resp) -> bytes | None:
    for candidate in getattr(resp, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            inline = getattr(part, "inline_data", None)
            if inline and getattr(inline, "data", None):
                return inline.data
    return None


def _pcm_to_wav(pcm: bytes, *, rate: int = _TTS_RATE, channels: int = 1, width: int = 2) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def _silent_wav(seconds: float, *, rate: int = _TTS_RATE) -> bytes:
    return _pcm_to_wav(b"\x00\x00" * int(rate * seconds), rate=rate)


def silent_wav(seconds: float) -> bytes:
    """Public silence helper (a failed beat keeps the image on screen, just unvoiced)."""
    return _silent_wav(max(0.0, seconds))


def wav_duration_seconds(data: bytes) -> float:
    """Length of a WAV blob in seconds (frames / framerate). Drives per-beat holds so an
    image stays on screen exactly as long as its narration."""
    with wave.open(io.BytesIO(data), "rb") as w:
        frames, rate = w.getnframes(), w.getframerate()
    return frames / rate if rate else 0.0


def concat_wavs(parts: list[bytes]) -> bytes:
    """Concatenate WAV blobs sequentially into one WAV (same format assumed). The
    narration FALLBACK: when the precisely-aligned track can't be built we still get one
    continuous voiceover from the audio we already synthesized — no extra TTS calls."""
    if not parts:
        return _silent_wav(0.5)
    buf = io.BytesIO()
    out: wave.Wave_write | None = None
    try:
        for p in parts:
            with wave.open(io.BytesIO(p), "rb") as w:
                if out is None:
                    out = wave.open(buf, "wb")
                    out.setnchannels(w.getnchannels())
                    out.setsampwidth(w.getsampwidth())
                    out.setframerate(w.getframerate())
                out.writeframes(w.readframes(w.getnframes()))
    finally:
        if out is not None:
            out.close()
    return buf.getvalue()

class IndexTTSTTSAdapter:
    """Synchronous adapter for the configured IndexTTS workbench.

    The assembler already calls ``synthesize`` in a worker thread, so keeping this
    client synchronous avoids an event-loop bridge.  The workbench contract is
    deliberately small: create a job, explicitly start it, poll, then download
    ``wav``.  Voice IDs are read from ``/api/voices`` when not configured.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.base_url = self.settings.tts_base_url.rstrip("/")

    def synthesize(self, text: str) -> bytes:
        if not text.strip():
            return _silent_wav(1.0)
        voice_id = self.settings.tts_voice_id.strip() or self._first_voice_id()
        if not voice_id:
            raise PermanentProviderError(
                "IndexTTS 没有可用 voice_id；请先确认 /api/voices 返回音色。"
            )
        payload = {
            "text": text,
            "voice_id": voice_id,
            "lang": "zh",
            "max_chars": 140,
            "pause_ms": 260,
            "max_text_tokens_per_segment": 120,
            "duration_factor": 1.0,
            "emotion_random": False,
        }
        try:
            job = self._json_request("POST", "/api/jobs", payload)
            job_id = job.get("job_id") or job.get("id")
            if not job_id:
                raise PermanentProviderError(f"IndexTTS create response missing job_id: {job}")
            # Unlike H3, IndexTTS creation and execution are explicitly separate.
            self._json_request("POST", f"/api/jobs/{job_id}/start", {})
            deadline = time.monotonic() + self.settings.video_timeout_seconds
            state: dict = {}
            while time.monotonic() < deadline:
                time.sleep(2)
                state = self._json_request("GET", f"/api/jobs/{job_id}")
                if state.get("status") in {"completed", "failed", "cancelled", "paused"}:
                    break
            if state.get("status") != "completed":
                raise TransientProviderError(f"IndexTTS job {job_id} did not complete: {state}")
            return self._bytes_request("GET", f"/api/jobs/{job_id}/download/wav")
        except (PermanentProviderError, TransientProviderError):
            raise
        except Exception as exc:  # noqa: BLE001 - normalize provider failures
            raise PermanentProviderError(f"IndexTTS request failed: {exc}") from exc

    def _first_voice_id(self) -> str:
        voices = self._json_request("GET", "/api/voices")
        if isinstance(voices, dict):
            values = voices.get("voices") or voices.get("items")
            if values is None:
                values = []
                for key in ("presets", "saved"):
                    entries = voices.get(key)
                    if isinstance(entries, list):
                        values.extend(entries)
            voices = values
        if not isinstance(voices, list):
            return ""
        for voice in voices:
            if isinstance(voice, str) and voice.strip():
                return voice.strip()
            if isinstance(voice, dict):
                voice_id = str(
                    voice.get("voice_id") or voice.get("id") or voice.get("key") or ""
                ).strip()
                if voice_id:
                    return voice_id
        return ""

    def _json_request(self, method: str, path: str, payload: dict | None = None) -> dict:
        import json
        import urllib.request

        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.settings.tts_http_timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))

    def _bytes_request(self, method: str, path: str) -> bytes:
        import urllib.request

        request = urllib.request.Request(self.base_url + path, method=method)
        with urllib.request.urlopen(request, timeout=self.settings.tts_http_timeout_seconds) as response:
            return response.read()


def make_tts_adapter(settings: Settings):
    """Build the selected TTS provider without leaking provider details to runner."""
    provider = (settings.tts_provider or "none").strip().lower()
    if provider in {"none", "off", "disabled"}:
        return None
    if provider in {"indextts", "index_tts", "index-tts", "local"}:
        return IndexTTSTTSAdapter(settings)
    if provider in {"gemini", "gemini_tts"}:
        return GeminiTTSAdapter(settings)
    raise PermanentProviderError(f"未知 TTS_PROVIDER={settings.tts_provider!r}")
