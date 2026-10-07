"""Regression checks for AI-clip repetition and missing narration subtitles."""
from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path

import pytest

from app.adapters.video_model import _normalize_video_bytes
from app.adapters.tts import silent_wav
from app.agents.assembler import assembler
from app.artifacts.store import ArtifactStore
from app.core.config import Settings
from app.graph.state import Deps

needs_ffmpeg = pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg required"
)


def two_phase_clip(path: Path) -> bytes:
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=black:s=64x112:r=12:d=1",
        "-f", "lavfi", "-i", "color=white:s=64x112:r=12:d=1",
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
    ], check=True, capture_output=True)
    return path.read_bytes()


def mean_luma(path: Path, seconds: float) -> float:
    data = subprocess.check_output([
        "ffmpeg", "-v", "error", "-ss", str(seconds), "-i", str(path),
        "-frames:v", "1", "-vf", "scale=8:8,format=gray", "-f", "rawvideo", "-",
    ])
    return sum(data) / len(data)


@needs_ffmpeg
async def test_short_generated_clip_is_retimed_once_instead_of_replayed(tmp_path):
    source = two_phase_clip(tmp_path / "source.mp4")
    normalized = tmp_path / "normalized.mp4"
    normalized.write_bytes(await _normalize_video_bytes(source, 4.0))
    # One continuous black->white arc, not black->white->black->white.
    assert mean_luma(normalized, 0.25) < 20
    assert mean_luma(normalized, 2.25) > 200, "scene restarted: provider clip was looped"


@needs_ffmpeg
async def test_provider_video_exports_narration_subtitles(tmp_path):
    store = ArtifactStore(tmp_path, "subtitle-regression")
    clip = Path(store.abspath("clip_01.mp4"))
    clip.parent.mkdir(parents=True, exist_ok=True)
    two_phase_clip(clip)
    settings = Settings(use_mock_providers=True, enable_tts=False)
    state = {"job_id": "subtitle-regression", "duration": 2, "clips": [
        {"order": 1, "path": str(clip), "caption": "A drop returns to the ocean.",
         "status": "ok", "duration_seconds": 2.0},
    ]}
    await assembler(state, Deps(settings=settings, store=store, llm=None))
    srt = Path(store.abspath("subtitles.srt"))
    assert srt.exists(), "provider-video assembly skipped subtitles entirely"
    assert "A drop returns to the ocean." in srt.read_text(encoding="utf-8")
