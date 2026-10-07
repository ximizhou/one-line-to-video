"""Research stage: collect lightweight web sources before creative generation."""
from __future__ import annotations

import asyncio

from app.adapters.research import ResearchAdapter
from app.core.logging import get_logger
from app.graph.state import Deps, StoryboardState

log = get_logger(__name__)


async def research(state: StoryboardState, deps: Deps) -> StoryboardState:
    if deps.store.exists("research.json"):
        log.info("reusing research artifact (resume)")
        return {"research_path": deps.store.abspath("research.json")}
    if deps.research is None:
        return {"warnings": ["research: provider unavailable; script continues without web sources"]}
    report = await asyncio.to_thread(deps.research.collect, state["prompt"])
    path = deps.store.save_json("research.json", report)
    log.info("research collected %d source(s) with %s", report.get("source_count", 0), report.get("provider"))
    warnings: list[str] = []
    if not report.get("sources"):
        warnings.append("research: no sources returned; fact claims will be conservative")
    return {"research_path": path, "warnings": warnings}
