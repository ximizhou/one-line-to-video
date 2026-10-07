from __future__ import annotations

from app.adapters.video_model import VideoGenerationResult
from app.agents.video_gen import video_gen
from app.artifacts.store import ArtifactStore
from app.core.config import Settings
from app.graph.state import Deps
from app.schemas import Shot, Shotlist


class _FakeVideo:
    def __init__(self) -> None:
        self.gpus: list[int | None] = []

    async def generate_video(self, prompt: str, **kwargs) -> VideoGenerationResult:
        self.gpus.append(kwargs.get("gpu"))
        return VideoGenerationResult(
            data=b"not-a-real-video",
            provider="fake",
            model="fake-model",
            duration_seconds=1.0,
        )


async def test_video_gen_round_robins_shots_across_gpu_pool(tmp_path):
    settings = Settings(
        video_provider="mock",
        video_model="fl2va",
        video_gpu=2,
        video_gpu_pool="2,3",
        video_max_concurrency=2,
        video_max_shots=0,
        video_clip_seconds=1.0,
        video_input_mode="t2v",
        seconds_per_frame=2.0,
    )
    store = ArtifactStore(tmp_path, "job")
    shotlist = Shotlist(
        style_bible="simple educational animation",
        shots=[
            Shot(order=1, image_prompt="ocean", narration="海洋"),
            Shot(order=2, image_prompt="cloud", narration="云朵"),
        ],
    )
    shotlist_path = store.save_json("shotlist.json", shotlist)
    video = _FakeVideo()

    result = await video_gen(
        {"shotlist_path": shotlist_path, "duration": 2},
        Deps(settings=settings, store=store, llm=None, video=video),
    )

    assert [clip["gpu"] for clip in result["clips"]] == [2, 3]
    assert video.gpus == [2, 3]
    assert all(clip["status"] == "ok" for clip in result["clips"])
