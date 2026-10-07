"""Generate one AI video clip per storyboard shot."""

from __future__ import annotations

import asyncio
from app.adapters.errors import ProviderError
from app.adapters.frame_seed import ensure_seed_frame
from app.adapters.video_model import VideoModelAdapter
from app.core.logging import get_logger
from app.graph.state import Deps, StoryboardState, VideoClipMeta
from app.schemas import Shotlist

log = get_logger(__name__)


def _video_gpu_pool(settings) -> tuple[int, ...]:
    """Parse the configured pool while keeping old single-GPU configs valid."""
    raw = str(getattr(settings, "video_gpu_pool", "") or "")
    values: list[int] = []
    for token in raw.replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            gpu = int(token)
        except ValueError:
            continue
        if gpu not in values and gpu >= 0:
            values.append(gpu)
    if not values:
        values = [int(settings.video_gpu)]
    return tuple(values)


async def video_gen(state: StoryboardState, deps: Deps) -> StoryboardState:
    """Generate or resume video clips for the shotlist.

    The node does not know which vendor is used.  It only consumes the stable
    ``VideoModelAdapter`` contract, which is the seam for the user's future
    video-model integration.
    """
    if deps.video is None:
        raise RuntimeError("video_gen: VideoModelAdapter is not configured")

    store = deps.store
    shotlist = Shotlist.model_validate(store.load_json(state["shotlist_path"]))
    clips: list[VideoClipMeta] = []
    warnings: list[str] = []
    shots = shotlist.shots
    if deps.settings.video_max_shots > 0:
        shots = shots[: deps.settings.video_max_shots]
        if len(shots) < len(shotlist.shots):
            warnings.append(
                f"video_gen: limited to {len(shots)} shots by VIDEO_MAX_SHOTS="
                f"{deps.settings.video_max_shots}"
            )
    # Keep the assembled film close to the requested duration. The provider
    # receives this as a per-shot duration hint; a future adapter can also trim
    # or pad its returned clip if the API only supports fixed durations.
    target_clip_seconds = state["duration"] / max(1, len(shots))
    clip_seconds = min(deps.settings.video_clip_seconds, target_clip_seconds)
    gpu_pool = _video_gpu_pool(deps.settings)
    log.info(
        "video generation dispatch: concurrency=%d gpu_pool=%s",
        max(1, deps.settings.video_max_concurrency),
        ",".join(str(gpu) for gpu in gpu_pool),
    )

    async def one(shot) -> VideoClipMeta:
        order = shot.order
        gpu = gpu_pool[(order - 1) % len(gpu_pool)]
        name = f"clip_{order:02d}.mp4"
        path = store.abspath(name)
        seed_path = None
        if (deps.settings.video_input_mode or "t2v").strip().lower() != "t2v":
            seed_name = f"seed_{order:02d}.png"
            seed_path = ensure_seed_frame(
                store.abspath(seed_name),
                prompt=shot.image_prompt,
                order=order,
                aspect_ratio=deps.settings.video_aspect_ratio,
            )
        if store.exists(name):
            log.info("reusing video clip %02d", order)
            return {
                "order": order,
                "path": path,
                "caption": shot.narration,
                "status": "ok",
                "provider": deps.settings.video_provider,
                "model": deps.settings.video_model or None,
                "duration_seconds": clip_seconds,
                "gpu": gpu,
            }

        try:
            motion_prompt = (
                f"{shot.image_prompt}. Animate this educational shot with one clear action, "
                "smooth natural camera movement and readable composition. No written words, "
                "captions, logos, watermarks, or random symbols; leave a clean lower third "
                "for subtitles. Preserve subject identity and the stated visual style."
            )
            result = await deps.video.generate_video(
                motion_prompt,
                duration_seconds=clip_seconds,
                aspect_ratio=deps.settings.video_aspect_ratio,
                seed=deps.settings.video_seed,
                image_path=seed_path,
                gpu=gpu,
            )
            store.save_bytes(name, result.data)
            return {
                "order": order,
                "path": path,
                "caption": shot.narration,
                "status": "ok",
                "provider": result.provider,
                "model": result.model,
                "duration_seconds": result.duration_seconds,
                "gpu": gpu,
            }
        except ProviderError as exc:
            log.warning("video clip %02d failed: %s", order, exc)
            warnings.append(f"video_gen: clip {order} failed ({exc})")
            return {
                "order": order,
                "path": path,
                "caption": shot.narration,
                "status": "degraded",
                "provider": deps.settings.video_provider,
                "model": deps.settings.video_model or None,
                "duration_seconds": clip_seconds,
                "gpu": gpu,
            }

    # Keep concurrency bounded; video APIs are expensive and often rate-limited.
    semaphore = asyncio.Semaphore(max(1, deps.settings.video_max_concurrency))

    async def guarded(shot):
        async with semaphore:
            return await one(shot)

    clips = list(await asyncio.gather(*(guarded(shot) for shot in shots)))
    clips.sort(key=lambda c: c["order"])
    degraded = sum(c["status"] == "degraded" for c in clips)
    if degraded:
        warnings.append(f"video_gen: {degraded} clip(s) degraded")
    log.info("video generation ready: %d clip(s), %d degraded", len(clips), degraded)
    return {"clips": clips, "warnings": warnings}
