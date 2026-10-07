"""Provider boundary for AI video generation.

The LangGraph node never talks to a vendor directly.  The default development
provider is deterministic mock video; the first real local implementation is the
configured MiniMax-H3 workbench.  A future Seedance/Kling/Veo adapter
can implement the same ``generate_video`` contract without changing the graph or
frontend.
"""

from __future__ import annotations

import asyncio
import hashlib
import mimetypes
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from app.adapters.errors import PermanentProviderError, TransientProviderError
from app.core.config import Settings
from app.core.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class VideoGenerationResult:
    """A generated clip returned by a provider."""

    data: bytes
    provider: str
    model: str | None
    duration_seconds: float


class VideoModelAdapter:
    """Stable interface used by the ``video_gen`` LangGraph node.

    ``image_path`` is optional on purpose.  H3 FL2VA supports pure text-to-video
    by default and can also accept an image-conditioned seed; future providers
    can ignore it.  The UI only selects a
    provider/model profile; all credentials and server URLs stay in environment
    settings rather than in browser requests.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def generate_video(
        self,
        prompt: str,
        *,
        duration_seconds: float,
        aspect_ratio: str = "9:16",
        seed: int | None = None,
        image_path: str | Path | None = None,
        gpu: int | None = None,
    ) -> VideoGenerationResult:
        provider = (self.settings.video_provider or "mock").strip().lower()
        if provider in {"mock", "color-lavfi"} or (
            self.settings.mock_mode and provider in {"custom", "http", "seedance"}
        ):
            return await self._mock_video(
                prompt,
                duration_seconds=duration_seconds,
                aspect_ratio=aspect_ratio,
                seed=seed,
            )
        if provider in {"h3", "h3_workbench", "minimax-h3", "minimax_h3"}:
            return await self._h3_workbench_video(
                prompt,
                duration_seconds=duration_seconds,
                aspect_ratio=aspect_ratio,
                seed=seed,
                image_path=image_path,
                gpu=gpu,
            )
        if provider in {"http", "custom_http", "seedance", "seedance_http"}:
            raise PermanentProviderError(
                "通用 HTTP 视频 provider 还没有绑定具体 Seedance 协议；"
                "请先把供应商的 submit/poll/download 实现接到此 adapter，"
                "或直接选择 h3_workbench。"
            )
        raise PermanentProviderError(
            f"未知 VIDEO_PROVIDER={self.settings.video_provider!r}。"
            "可选 mock、h3_workbench，或后续接入自定义 adapter。"
        )

    async def _h3_workbench_video(
        self,
        prompt: str,
        *,
        duration_seconds: float,
        aspect_ratio: str,
        seed: int | None,
        image_path: str | Path | None,
        gpu: int | None,
    ) -> VideoGenerationResult:
        """Call the configured H3 workbench HTTP API.

        The workbench queues a text-only H3 job by default.  If ``i2v`` is
        selected, it uploads a seed image first.  In both cases we return
        ``/outputs/*.webm`` after polling ``/api/jobs/{id}`` and inspect every
        item rather than trusting the top-level ``done`` status.
        """
        input_mode = (self.settings.video_input_mode or "t2v").strip().lower()
        if input_mode not in {"t2v", "i2v", "ref2va"}:
            raise PermanentProviderError(
                f"不支持 VIDEO_INPUT_MODE={self.settings.video_input_mode!r}；可选 t2v/i2v。"
            )
        if input_mode != "t2v" and (image_path is None or not Path(image_path).exists()):
            raise PermanentProviderError(
                "H3 i2v 模式需要 image_path；请先生成 seed frame 或提供用户图片。"
            )

        base_url = self.settings.video_base_url.rstrip("/")
        if not base_url:
            raise PermanentProviderError(
                "VIDEO_BASE_URL 为空；请把 H3 workbench endpoint 暴露到本地或可访问的私有地址。"
            )
        model = (self.settings.video_model or "fl2va").strip()
        if input_mode == "t2v" and model != "fl2va":
            raise PermanentProviderError(
                f"H3 模型 {model} 不支持当前纯文生配置；请改用 fl2va 或切换 VIDEO_INPUT_MODE=i2v。"
            )
        selected_gpu = self.settings.video_gpu if gpu is None else int(gpu)
        frames = self.settings.video_frames
        if frames <= 0:
            frames = max(9, min(345, round(max(1.0, duration_seconds) * 24)))
        payload = {
            "images": [],
            "text_only": input_mode == "t2v",
            "aspect_ratio": aspect_ratio,
            "prompt": prompt,
            "gpu": selected_gpu,
            "model": model,
            "quality": self.settings.video_quality,
            "frames": max(9, min(345, frames)),
            "steps": max(1, min(12, self.settings.video_steps)),
            "seed": seed if seed is not None else "random",
            "venhance": self.settings.video_enhance,
            "venh_model": self.settings.video_enhance_model,
            "venh_scale": self.settings.video_enhance_scale,
            "venh_tile": self.settings.video_enhance_tile,
            "venh_sharpen": self.settings.video_enhance_sharpen,
        }
        image_file = Path(image_path) if image_path is not None else None
        timeout = httpx.Timeout(self.settings.video_http_timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                if input_mode != "t2v":
                    assert image_file is not None
                    with image_file.open("rb") as fh:
                        upload = await client.post(
                            f"{base_url}/api/upload",
                            files={
                                "file": (
                                    image_file.name,
                                    fh,
                                    mimetypes.guess_type(image_file.name)[0] or "image/png",
                                )
                            },
                        )
                    upload.raise_for_status()
                    uploaded_name = upload.json().get("filename")
                    if not uploaded_name:
                        raise PermanentProviderError(f"H3 upload response missing filename: {upload.text[:400]}")
                    payload["images"] = [uploaded_name]
                    payload["text_only"] = False
                response = await client.post(f"{base_url}/api/job", json=payload)
                response.raise_for_status()
                job = response.json()
                job_id = job.get("id")
                if not job_id:
                    raise PermanentProviderError(f"H3 job response missing id: {response.text[:400]}")
                log.info("H3 job submitted: id=%s gpu=%s model=%s", job_id, selected_gpu, model)

                deadline = time.monotonic() + self.settings.video_timeout_seconds
                final: dict[str, Any] = job
                while time.monotonic() < deadline:
                    await asyncio.sleep(self.settings.video_poll_interval_seconds)
                    status_response = await client.get(f"{base_url}/api/jobs/{job_id}")
                    status_response.raise_for_status()
                    final = status_response.json()
                    status = str(final.get("status", "")).lower()
                    if status in {"done", "completed", "failed", "cancelled", "error"}:
                        break
                else:
                    raise TransientProviderError(
                        f"H3 job {job_id} 超过 {self.settings.video_timeout_seconds:.0f}s，"
                        "请先查询服务器任务状态后再重试。"
                    )

                status = str(final.get("status", "")).lower()
                items = final.get("items") or []
                failed = [item for item in items if item.get("status") in {"failed", "error"} or item.get("error")]
                if status not in {"done", "completed"} or failed:
                    detail = failed[0].get("error") if failed else final.get("error")
                    raise PermanentProviderError(
                        f"H3 job {job_id} failed: status={status}, detail={detail or final}"
                    )
                output = next((item.get("output") for item in items if item.get("output")), None)
                if not output:
                    raise PermanentProviderError(f"H3 job {job_id} done but no item output: {final}")
                output_url = output if str(output).startswith("http") else f"{base_url}/{str(output).lstrip('/')}"
                media = await client.get(output_url)
                media.raise_for_status()
                normalized = await _normalize_video_bytes(media.content, float(duration_seconds))
                return VideoGenerationResult(
                    data=normalized,
                    provider="h3_workbench",
                    model=model,
                    duration_seconds=float(duration_seconds),
                )
            except httpx.TimeoutException as exc:
                raise TransientProviderError(f"H3 HTTP timeout: {exc}") from exc
            except httpx.HTTPStatusError as exc:
                body = exc.response.text[:500]
                raise PermanentProviderError(
                    f"H3 HTTP {exc.response.status_code}: {body}"
                ) from exc
            except (PermanentProviderError, TransientProviderError):
                raise
            except Exception as exc:  # noqa: BLE001 - normalize provider failures
                raise PermanentProviderError(f"H3 request failed: {exc}") from exc

    async def _mock_video(
        self,
        prompt: str,
        *,
        duration_seconds: float,
        aspect_ratio: str,
        seed: int | None,
    ) -> VideoGenerationResult:
        """Create a deterministic local MP4 without image-generation APIs."""
        if aspect_ratio == "16:9":
            size = "1280x720"
        else:
            size = "720x1280"
        digest = hashlib.sha256(f"{prompt}:{seed}".encode("utf-8")).hexdigest()
        color = f"0x{digest[:6]}"
        duration = max(1.0, float(duration_seconds))

        with tempfile.TemporaryDirectory(prefix="storyboard_mock_video_") as tmp:
            output = Path(tmp) / "clip.mp4"
            cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"color=c={color}:s={size}:r=30:d={duration:.3f}",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(output),
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0 or not output.exists():
                raise PermanentProviderError(
                    f"mock video generation failed: {stderr.decode(errors='replace')[-800:]}"
                )
            log.info("mock video clip generated (%ss, %s)", duration, aspect_ratio)
            return VideoGenerationResult(
                data=output.read_bytes(),
                provider="mock",
                model="color-lavfi",
                duration_seconds=duration,
            )


def available_video_profiles(settings: Settings) -> list[dict[str, Any]]:
    """Return safe, non-secret provider choices for the UI."""
    return [
        {
            "id": "mock",
            "label": "Mock（本地流程验证）",
            "models": [{"id": "color-lavfi", "label": "Color / FFmpeg"}],
            "requires_image": False,
            "configured": True,
        },
        {
            "id": "h3_workbench",
            "label": "MiniMax-H3 工作台",
            "models": [
                {"id": "fl2va", "label": "FL2VA · 纯文生 / 首帧图生 · Q8", "input_modes": ["t2v", "i2v"]},
                {"id": "ref2va", "label": "REF2VA · 参考图生视频 · Q6", "input_modes": ["i2v"]},
            ],
            "requires_image": settings.video_input_mode != "t2v",
            "configured": bool(settings.video_base_url.strip()),
            "base_url": settings.video_base_url,
            "gpu": settings.video_gpu,
            "input_mode": settings.video_input_mode,
        },
        {
            "id": "seedance_http",
            "label": "Seedance（预留 HTTP 接口）",
            "models": [{"id": settings.video_model or "seedance-model", "label": "由后续 adapter 接入"}],
            "requires_image": False,
            "configured": False,
        },
    ]


async def _normalize_video_bytes(data: bytes, duration_seconds: float) -> bytes:
    """Retiming/looping makes provider clips honor the storyboard duration hint."""
    if duration_seconds <= 0:
        return data
    import tempfile
    with tempfile.TemporaryDirectory(prefix="storyboard_retime_") as tmp:
        source = Path(tmp) / "source.webm"
        output = Path(tmp) / "normalized.mp4"
        source.write_bytes(data)
        cmd = [
            "ffmpeg", "-y", "-stream_loop", "-1", "-i", str(source),
            "-t", f"{duration_seconds:.3f}", "-an", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0 or not output.exists():
            raise PermanentProviderError(
                f"video retime failed: {stderr.decode(errors='replace')[-600:]}"
            )
        return output.read_bytes()
