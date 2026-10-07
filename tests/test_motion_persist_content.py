"""Provider video clips -> artifact rows -> scenes[].video_url."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import get_settings
from app.db import repository as repo
from app.db.session import get_sessionmaker
from app.main import app
from app.workers.runner import _persist

_BASE_PIPELINE = ["script_writer", "reflection", "designer", "video_gen", "assembler"]


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _make_job() -> str:
    async with get_sessionmaker()() as s:
        job = await repo.create_job(s, "p", 30)
        return job.id


async def test_persist_maps_provider_clips_to_video_url(client, tmp_path):
    job_id = await _make_job()
    clip1 = tmp_path / "clip_01.mp4"
    clip2 = tmp_path / "clip_02.mp4"
    clip1.write_bytes(b"clip-1")
    clip2.write_bytes(b"clip-2")
    final = {
        "clips": [
            {"order": 1, "path": str(clip1), "caption": "one", "status": "ok"},
            {"order": 2, "path": str(clip2), "caption": "two", "status": "ok"},
        ],
        "warnings": [],
    }
    await _persist(job_id, final)

    body = (await client.get(f"/storyboard/{job_id}/storyboard")).json()
    scenes = {sc["order"]: sc for sc in body["scenes"]}
    assert scenes[1]["video_url"] is not None
    assert scenes[2]["video_url"] is not None
    assert body["motion_coverage"]["generative"] == 2

    status = (await client.get(f"/storyboard/{job_id}")).json()
    kinds = [a["kind"] for a in status["artifacts"]]
    assert kinds.count("clip") == 2


async def test_no_provider_clips_leaves_video_url_none(client):
    job_id = await _make_job()
    final = {"warnings": []}
    await _persist(job_id, final)
    body = (await client.get(f"/storyboard/{job_id}/storyboard")).json()
    assert body["scenes"] == []


async def test_pipeline_field_is_provider_video_graph(client):
    job_id = await _make_job()
    body = (await client.get(f"/storyboard/{job_id}")).json()
    assert body["pipeline"] == _BASE_PIPELINE


async def test_legacy_motion_flag_does_not_change_active_graph(client, monkeypatch):
    monkeypatch.setenv("MOTION_ENGINE_ENABLED", "true")
    get_settings.cache_clear()
    try:
        job_id = await _make_job()
        body = (await client.get(f"/storyboard/{job_id}")).json()
        assert body["pipeline"] == _BASE_PIPELINE
    finally:
        get_settings.cache_clear()
