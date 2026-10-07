from app.adapters.research import ResearchAdapter, ResearchSource, _matches_topic, _topic_terms
from app.core.config import Settings


def test_water_cycle_search_uses_chinese_and_english_queries(monkeypatch):
    adapter = ResearchAdapter(Settings(research_provider="wikipedia", research_max_results=6))
    calls = []

    def fake_search(query, *, limit=None):
        calls.append(query)
        if "water cycle" in query:
            return [ResearchSource("Water cycle", "https://en.wikipedia.org/wiki/Water_cycle", "Evaporation and runoff", "wikipedia")]
        if "hydrologic" in query:
            return [ResearchSource("Hydrologic cycle", "https://en.wikipedia.org/wiki/Hydrologic_cycle", "Condensation and precipitation", "wikipedia")]
        return [ResearchSource("水循环", f"https://example.test/{len(calls)}", "蒸发、凝结、降水和径流", "wikipedia")]

    monkeypatch.setattr(adapter, "search", fake_search)
    report = adapter.collect("水循环的一生")

    assert report["source_count"] == 4
    assert calls == report["queries"]
    assert any("水循环" in query for query in report["queries"])
    assert any("water cycle" in query for query in report["queries"])
    assert all(_matches_topic(ResearchSource(**source), _topic_terms("水循环的一生")) for source in report["sources"])


def test_topic_filter_rejects_search_engine_noise():
    terms = _topic_terms("水循环的一生")
    relevant = ResearchSource("Water cycle", "https://en.wikipedia.org/wiki/Water_cycle", "Evaporation and runoff", "wikipedia")
    irrelevant = ResearchSource("Air Jordan 23", "https://example.com", "Shoes and sneakers", "bing")

    assert _matches_topic(relevant, terms)
    assert not _matches_topic(irrelevant, terms)
