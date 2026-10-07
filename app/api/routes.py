"""HTTP surface — the async job API (curl/Postman friendly).

    POST /storyboard            -> 202 {job_id, status}        (enqueue, returns fast)
    GET  /storyboard/{id}       -> 200 status + STAGES + artifacts | 404
    GET  /storyboard/{id}/logs  -> 200 paginated log lines (?stage=&after=&limit=) | 404
    GET  /storyboard/{id}/events-> 200 SSE live stream (status|stage|log|done) | 404
    GET  /storyboard/{id}/video -> 200 mp4 | 409 not-ready | 404

Generation runs in the background (minutes). A client can either POLL (GET status
+ GET logs?after=cursor) or SUBSCRIBE (GET events, SSE) for live progress.
"""

from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.adapters.turnstile import verify_turnstile
from app.adapters.video_model import available_video_profiles
from app.adapters.research import available_research_profiles
from app.adapters.llm import available_llm_profiles
from app.core.config import get_settings
from app.graph.build import pipeline_stage_names
from app.db import repository as repo
from app.db.session import get_session, get_sessionmaker
from app.schemas import (
    ArtifactRef,
    JobCreatedResponse,
    JobStatus,
    JobStatusResponse,
    LogLine,
    LogsResponse,
    StageRef,
    StoryboardContent,
    StoryboardRequest,
    StoryboardScene,
)
from app.workers.runner import run_job

router = APIRouter(prefix="/storyboard", tags=["storyboard"])


def _normalise_tts_voices(payload: object) -> list[dict[str, str]]:
    """Convert IndexTTS voice payloads into safe, UI-facing metadata.

    The workbench has returned both ``voices`` and ``items`` wrappers across
    versions, and voice IDs have appeared as either ``id`` or ``voice_id``.
    Keep this adapter deliberately permissive while never exposing paths or
    provider internals to the browser.
    """
    if isinstance(payload, dict):
        values = payload.get("voices") or payload.get("items")
        if values is None:
            # IndexTTS workbench currently separates built-in and saved voices.
            values = []
            for key in ("presets", "saved"):
                entries = payload.get(key)
                if isinstance(entries, list):
                    values.extend(entries)
    else:
        values = payload
    if not isinstance(values, list):
        return []

    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in values:
        if isinstance(item, str):
            voice_id = item.strip()
            name = voice_id
            description = ""
            language = ""
        elif isinstance(item, dict):
            voice_id = str(
                item.get("voice_id") or item.get("id") or item.get("key") or ""
            ).strip()
            name = str(item.get("name") or item.get("label") or voice_id).strip()
            description = str(
                item.get("description") or item.get("style") or item.get("remark") or ""
            ).strip()
            language = str(item.get("language") or item.get("lang") or "").strip()
        else:
            continue
        if not voice_id or voice_id in seen:
            continue
        seen.add(voice_id)
        entry = {"id": voice_id, "label": name or voice_id}
        if description:
            entry["description"] = description
        if language:
            entry["language"] = language
        result.append(entry)
    return result


def _fetch_tts_voices_sync(base_url: str, timeout: float) -> object:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/voices",
        headers={"Accept": "application/json"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


async def _configured_tts_voices(settings) -> list[dict[str, str]]:
    """Best-effort voice discovery; a stopped local TTS service must not break UI."""
    base_url = settings.tts_base_url.strip()
    if not base_url or settings.use_mock_providers:
        return []
    try:
        # Keep provider discovery snappy when the SSH tunnel/service is down.
        payload = await asyncio.wait_for(
            asyncio.to_thread(
                _fetch_tts_voices_sync,
                base_url,
                min(2.5, max(0.5, float(settings.tts_http_timeout_seconds))),
            ),
            timeout=3.0,
        )
    except Exception:  # noqa: BLE001 - provider discovery is optional
        return []
    return _normalise_tts_voices(payload)


@router.get("/providers")
async def get_provider_profiles() -> dict:
    """Safe provider/model metadata for the model switcher UI."""
    settings = get_settings()
    return {
        "defaults": {
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            "research_enabled": settings.research_enabled,
            "research_provider": settings.research_provider,
            "video_provider": settings.video_provider,
            "video_model": settings.video_model,
            "video_gpu": settings.video_gpu,
            "video_gpu_pool": settings.video_gpu_pool,
            "video_max_concurrency": settings.video_max_concurrency,
            "tts_provider": settings.tts_provider if settings.enable_tts else "none",
            "tts_voice_id": settings.tts_voice_id,
        },
        "llm": available_llm_profiles(settings),
        "research": available_research_profiles(settings),
        "video": available_video_profiles(settings),
        "tts": [
            {"id": "none", "label": "不生成旁白", "configured": True},
            {"id": "indextts", "label": "IndexTTS workbench", "configured": bool(settings.tts_base_url.strip())},
            {"id": "gemini", "label": "Gemini TTS", "configured": bool(settings.gemini_api_key.strip())},
        ],
        "tts_voices": await _configured_tts_voices(settings),
    }

# Per-IP rate limiter for the public create endpoint (registered on the app in
# app/main.py). Disabled in tests via settings.rate_limit_enabled so the shared
# in-memory counter can't bleed across test cases.
_settings = get_settings()
limiter = Limiter(
    key_func=get_remote_address,
    enabled=_settings.rate_limit_enabled and bool(_settings.rate_limit_create.strip()),
)
_CREATE_RATE_LIMIT = _settings.rate_limit_create.strip() or "100/minute"


# --------------------------------------------------------------------------- #
# ORM -> schema mappers (shared by the pull endpoints + the SSE stream).
# --------------------------------------------------------------------------- #
def _stage_ref(st) -> StageRef:
    return StageRef(
        name=st.name,
        seq=st.seq,
        status=st.status,
        message=st.message,
        progress_current=st.progress_current,
        progress_total=st.progress_total,
        tokens=st.tokens,
        started_at=st.started_at,
        ended_at=st.ended_at,
    )


def _log_line(lg) -> LogLine:
    return LogLine(
        id=lg.id, stage=lg.stage, level=lg.level, message=lg.message, created_at=lg.created_at
    )


def _current_stage(stages) -> str | None:
    """The name of the stage currently 'running', if any (else None)."""
    return next((st.name for st in stages if st.status == "running"), None)


def _expires_at(job) -> datetime:
    """When a job (and its artifacts) expire = created_at + the TTL."""
    created = job.created_at
    if created.tzinfo is None:  # SQLite can hand back a naive datetime
        created = created.replace(tzinfo=timezone.utc)
    return created + timedelta(seconds=get_settings().job_ttl_seconds)


async def _load_live_job(session: AsyncSession, job_id: str):
    """Fetch a job; 404 if unknown, 410 if past its 1-hour TTL.

    The background sweeper deletes expired rows on its own cadence; this makes
    expiry immediate + explicit to clients even before the next sweep runs."""
    job = await repo.get_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if datetime.now(timezone.utc) >= _expires_at(job):
        raise HTTPException(
            status_code=410, detail="job expired (jobs are kept for 1 hour)"
        )
    return job


def _artifact_url(job_id: str, artifact_id: str) -> str:
    """Relative URL the browser uses to stream an artifact's bytes. Relative so it
    works behind any proxy/host; the frontend prefixes its API base."""
    return f"/storyboard/{job_id}/artifacts/{artifact_id}"


def _file_response(path: str) -> FileResponse:
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="artifact file missing on disk")
    media_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type)


@router.post("", status_code=202, response_model=JobCreatedResponse)
@limiter.limit(_CREATE_RATE_LIMIT)
async def create_storyboard(
    request: Request,
    body: StoryboardRequest,
    background: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
) -> JobCreatedResponse:
    # Guard the public, un-authed endpoint against quota-draining abuse: a per-IP
    # rate limit (the decorator) + a Cloudflare Turnstile check (both no-ops when
    # disabled for local/mock dev).
    if not await verify_turnstile(
        get_settings(), body.turnstile_token, get_remote_address(request)
    ):
        raise HTTPException(status_code=403, detail="Turnstile verification failed")
    options = {
        key: value
        for key, value in {
            "llm_provider": body.llm_provider,
            "llm_model": body.llm_model,
            "research_enabled": body.research_enabled,
            "research_provider": body.research_provider,
            "video_provider": body.video_provider,
            "video_model": body.video_model,
            "video_input_mode": body.video_input_mode,
            "video_gpu": body.video_gpu,
            "video_gpu_pool": body.video_gpu_pool,
            "video_max_concurrency": body.video_max_concurrency,
            "tts_provider": body.tts_provider,
            "tts_voice_id": body.tts_voice,
            "enable_tts": (
                body.tts_provider is not None
                and body.tts_provider.strip().lower() not in {"", "none", "off", "disabled"}
            ) if body.tts_provider is not None else None,
        }.items()
        if value is not None and value != ""
    }
    job = await repo.create_job(session, body.prompt, body.duration, options)
    background.add_task(run_job, job.id, body.prompt, body.duration, False, options)
    return JobCreatedResponse(job_id=job.id, status=JobStatus(job.status))


@router.post("/{job_id}/resume", status_code=202, response_model=JobCreatedResponse)
async def resume_storyboard(
    job_id: str,
    background: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
) -> JobCreatedResponse:
    """Resume an interrupted/failed job. Idempotent nodes skip work already done.

    409 if it's already running or already finished cleanly (nothing to resume).
    NOTE: this status check is UX only — the real concurrency guard is the atomic
    ``try_acquire_job`` inside ``run_job``, so two racing resumes can't both run.
    """
    job = await _load_live_job(session, job_id)

    status = JobStatus(job.status)
    if status == JobStatus.RUNNING:
        raise HTTPException(status_code=409, detail="job is already running")
    if status in (JobStatus.COMPLETED, JobStatus.COMPLETED_WITH_WARNINGS):
        raise HTTPException(
            status_code=409,
            detail=f"job already finished (status={job.status}); nothing to resume",
        )

    background.add_task(run_job, job.id, job.prompt, job.duration, True, job.options or {})
    return JobCreatedResponse(job_id=job.id, status=JobStatus.RUNNING)


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_storyboard(
    job_id: str, session: AsyncSession = Depends(get_session)
) -> JobStatusResponse:
    job = await _load_live_job(session, job_id)
    stages = await repo.get_stages(session, job_id)
    return JobStatusResponse(
        job_id=job.id,
        status=JobStatus(job.status),
        prompt=job.prompt,
        duration=job.duration,
        created_at=job.created_at,
        expires_at=_expires_at(job),
        error=job.error,
        warnings=job.warnings or [],
        current_stage=_current_stage(stages),
        total_tokens=job.total_tokens,
        prompt_tokens=job.prompt_tokens,
        output_tokens=job.output_tokens,
        pipeline=pipeline_stage_names(get_settings()),
        stages=[_stage_ref(st) for st in stages],
        artifacts=[
            ArtifactRef(
                id=a.id,
                kind=a.kind,
                url=_artifact_url(job.id, a.id),
                status=a.status,
                order=a.order_index,
            )
            for a in job.artifacts
        ],
    )


@router.get("/{job_id}/logs", response_model=LogsResponse)
async def list_logs(
    job_id: str,
    after: int = 0,
    stage: str | None = None,
    limit: int = 500,
    session: AsyncSession = Depends(get_session),
) -> LogsResponse:
    """Paginated log pull — the expand-view + a curl-friendly replay of the stream.

    ``after`` is the cursor (the last id you've seen; 0 = from the start);
    ``stage`` filters to one stage; poll by passing back ``next_after``.
    """
    await _load_live_job(session, job_id)
    limit = max(1, min(limit, 2000))
    logs = await repo.get_logs(session, job_id, after_id=after, stage=stage, limit=limit)
    next_after = logs[-1].id if logs else after
    return LogsResponse(
        job_id=job_id, logs=[_log_line(lg) for lg in logs], next_after=next_after
    )


@router.get("/{job_id}/events")
async def stream_events(job_id: str, request: Request) -> EventSourceResponse:
    """Server-Sent Events: live status + stage + log stream for one job.

    On connect: emits a status+stage snapshot, then REPLAYS logs since
    ``Last-Event-ID`` (or from the start). Then polls the DB every
    ``log_poll_interval_seconds`` for new logs (id>cursor) + changed stages,
    emitting diffs. Closes with a ``done`` event once the job is terminal AND the
    log stream has drained. ``log`` events carry ``id:`` (the cursor); ``status``/
    ``stage`` events are idempotent snapshots, safe to re-send on reconnect.

    Uses its OWN short-lived session per poll (not a request-scoped one held open
    for minutes), so it sees the runner's committed writes and survives a future
    move to a separate worker process.
    """
    settings = get_settings()
    if not settings.enable_log_streaming:
        raise HTTPException(status_code=503, detail="log streaming is disabled")

    sessionmaker = get_sessionmaker()
    async with sessionmaker() as s:  # existence + expiry check before streaming
        await _load_live_job(s, job_id)

    poll = max(0.01, settings.log_poll_interval_seconds)
    batch = settings.log_drain_batch

    async def event_gen():
        # Resume support: a reconnecting EventSource sends the last id it saw.
        hdr = request.headers.get("last-event-id")
        cursor = int(hdr) if (hdr and hdr.lstrip("-").isdigit() and int(hdr) >= 0) else 0
        last_sig: tuple | None = None
        terminal_quiet = 0

        while True:
            if await request.is_disconnected():
                break

            async with sessionmaker() as s:
                job = await repo.get_job(s, job_id)
                stages = await repo.get_stages(s, job_id)
                logs = await repo.get_logs(s, job_id, after_id=cursor, limit=batch)

            if job is None:  # deleted mid-stream -> stop cleanly
                break

            # Status + stage snapshot, only when something changed (idempotent).
            sig = (
                job.status,
                tuple(
                    (st.name, st.status, st.progress_current, st.progress_total)
                    for st in stages
                ),
            )
            if sig != last_sig:
                yield {
                    "event": "status",
                    "data": json.dumps(
                        {"status": job.status, "current_stage": _current_stage(stages)}
                    ),
                }
                yield {
                    "event": "stage",
                    "data": json.dumps([_stage_ref(st).model_dump(mode="json") for st in stages]),
                }
                last_sig = sig

            for lg in logs:
                yield {"id": str(lg.id), "event": "log", "data": _log_line(lg).model_dump_json()}
                cursor = lg.id

            # Close only after the job is terminal AND two consecutive quiet polls
            # (gives the best-effort log drainer a cycle to flush any stragglers).
            if repo.is_terminal(job.status) and not logs:
                terminal_quiet += 1
                if terminal_quiet >= 2:
                    yield {"event": "done", "data": json.dumps({"status": job.status})}
                    break
            else:
                terminal_quiet = 0

            await asyncio.sleep(poll)

    return EventSourceResponse(event_gen())


@router.get("/{job_id}/video")
async def get_video(
    job_id: str, session: AsyncSession = Depends(get_session)
) -> FileResponse:
    job = await _load_live_job(session, job_id)

    video = next((a for a in job.artifacts if a.kind == "video"), None)
    if video is None:
        if repo.is_terminal(job.status):
            raise HTTPException(status_code=404, detail="no video produced for this job")
        raise HTTPException(
            status_code=409, detail=f"video not ready (status={job.status})"
        )
    if not os.path.exists(video.path):
        raise HTTPException(status_code=404, detail="video file missing on disk")

    return FileResponse(video.path, media_type="video/mp4", filename=f"{job_id}.mp4")


@router.get("/{job_id}/artifacts/{artifact_id}")
async def get_artifact(
    job_id: str, artifact_id: str, session: AsyncSession = Depends(get_session)
) -> FileResponse:
    """Stream one artifact's bytes (frame / style_ref / script / ...).

    The browser-safe way to fetch images: the raw filesystem path is never exposed
    to clients, only this opaque id-addressed URL."""
    job = await _load_live_job(session, job_id)
    art = next((a for a in job.artifacts if a.id == artifact_id), None)
    if art is None:
        raise HTTPException(status_code=404, detail="artifact not found")
    return _file_response(art.path)


@router.get("/{job_id}/frames/{order}")
async def get_frame(
    job_id: str, order: int, session: AsyncSession = Depends(get_session)
) -> FileResponse:
    """Convenience: stream the frame at a given order (1..N)."""
    job = await _load_live_job(session, job_id)
    art = next(
        (a for a in job.artifacts if a.kind == "frame" and a.order_index == order),
        None,
    )
    if art is None:
        raise HTTPException(status_code=404, detail="frame not found")
    return _file_response(art.path)


@router.get("/{job_id}/clips/{order}")
async def get_clip(
    job_id: str, order: int, session: AsyncSession = Depends(get_session)
) -> FileResponse:
    """Convenience endpoint for one provider-generated scene clip."""
    job = await _load_live_job(session, job_id)
    art = next(
        (a for a in job.artifacts if a.kind == "clip" and a.order_index == order),
        None,
    )
    if art is None:
        raise HTTPException(status_code=404, detail="clip not found")
    return _file_response(art.path)


@router.get("/{job_id}/storyboard", response_model=StoryboardContent)
async def get_storyboard_content(
    job_id: str, session: AsyncSession = Depends(get_session)
) -> StoryboardContent:
    """Return the final film plus one AI-generated clip per storyboard shot."""
    job = await _load_live_job(session, job_id)
    artifacts = list(job.artifacts)
    by_kind = {a.kind: a for a in artifacts}

    def _read_json(art) -> dict:
        if art is None or not os.path.exists(art.path):
            return {}
        try:
            return json.loads(Path(art.path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    script = _read_json(by_kind.get("script"))
    shotlist = _read_json(by_kind.get("shotlist"))
    narration_by_order: dict[int, str] = {
        s["order"]: s.get("narration", "")
        for s in shotlist.get("shots", [])
        if "order" in s
    }
    if not narration_by_order:
        narration_by_order = {
            s["order"]: s.get("narration", "")
            for s in script.get("scenes", [])
            if "order" in s
        }

    frame_by_order = {a.order_index: a for a in artifacts if a.kind == "frame"}
    clip_by_order = {a.order_index: a for a in artifacts if a.kind == "clip"}
    motion_by_order = {a.order_index: a for a in artifacts if a.kind == "motion"}
    orders = sorted(
        order for order in set(frame_by_order) | set(clip_by_order) | set(motion_by_order)
        if order is not None
    )

    scenes = []
    for order in orders:
        frame = frame_by_order.get(order)
        clip = clip_by_order.get(order)
        motion = motion_by_order.get(order)
        selected_video = clip or motion
        scenes.append(
            StoryboardScene(
                order=order,
                narration=narration_by_order.get(order, ""),
                frame_url=_artifact_url(job_id, frame.id) if frame else None,
                video_url=_artifact_url(job_id, selected_video.id) if selected_video else None,
                status=(
                    "degraded"
                    if (clip and clip.status == "degraded")
                    or (motion and motion.status == "degraded")
                    else (frame.status if frame else "ok")
                ),
            )
        )

    video = by_kind.get("video")
    return StoryboardContent(
        job_id=job_id,
        status=JobStatus(job.status),
        title=script.get("title", ""),
        logline=script.get("logline", ""),
        style=script.get("style", ""),
        duration=job.duration,
        video_url=f"/storyboard/{job_id}/video" if video is not None else None,
        style_ref_url=None,
        total_tokens=job.total_tokens,
        narrated="narration" in by_kind,
        motion_coverage={
            "total": len(orders),
            "animated": len(clip_by_order) + len(motion_by_order),
            "generative": len(clip_by_order),
            "local_generative": 0,
            "composited": 0,
            "kenburns": 0,
            "still": max(0, len(orders) - len(clip_by_order) - len(motion_by_order)),
            "by_renderer": {},
            "by_model": {},
        },
        scenes=scenes,
    )
