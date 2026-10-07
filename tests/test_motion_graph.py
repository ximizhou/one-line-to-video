"""The active graph uses a provider-backed video_gen stage only."""

from __future__ import annotations

from types import SimpleNamespace

from app.core.config import Settings
from app.graph.build import build_graph, pipeline_stage_names
from app.graph.state import Deps

_BASE = ["script_writer", "reflection", "designer", "video_gen", "assembler"]


def _deps() -> Deps:
    return Deps(
        settings=Settings(use_mock_providers=True),
        store=SimpleNamespace(), llm=SimpleNamespace(), image=None,
        video=SimpleNamespace(), tts=None, reporter=None, motion=None,
    )


def test_pipeline_names_are_provider_video_first():
    assert pipeline_stage_names(Settings()) == _BASE


def test_legacy_motion_toggle_does_not_reintroduce_removed_branch():
    assert pipeline_stage_names(Settings(motion_engine_enabled=True)) == _BASE


def test_graph_has_video_generation_node():
    nodes = set(build_graph(_deps()).get_graph().nodes)
    assert "video_gen" in nodes
    assert "image_gen" not in nodes
    assert "motion" not in nodes
