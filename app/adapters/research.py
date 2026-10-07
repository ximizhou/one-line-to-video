"""Small, provider-switchable web research client used before script writing."""
from __future__ import annotations

import base64
import html
import os
import re
from dataclasses import dataclass, asdict
from typing import Any
from urllib.parse import parse_qs, quote, quote_plus, unquote, urlparse

import httpx

from app.adapters.errors import PermanentProviderError, TransientProviderError
from app.core.config import Settings
from app.core.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class ResearchSource:
    title: str
    url: str
    snippet: str
    source: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


class ResearchAdapter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def search(self, query: str, *, limit: int | None = None) -> list[ResearchSource]:
        provider = (self.settings.research_provider or "duckduckgo").strip().lower()
        limit = limit or self.settings.research_max_results
        if provider in {"none", "off", "disabled"}:
            return []
        if provider in {"duckduckgo", "ddg", "duckduckgo_html"}:
            return self._duckduckgo(query, limit)
        if provider in {"bing", "bing_html"}:
            return self._bing(query, limit)
        if provider == "tavily":
            return self._tavily(query, limit)
        if provider == "serpapi":
            return self._serpapi(query, limit)
        if provider in {"wikipedia", "wiki"}:
            return self._wikipedia(query, limit)
        raise PermanentProviderError(f"未知 RESEARCH_PROVIDER={self.settings.research_provider!r}")

    def collect(self, topic: str) -> dict[str, Any]:
        core = _extract_core_topic(topic)
        if "水循环" in core:
            queries = [
                core,
                "水循环 蒸发 凝结 降水 径流",
                "water cycle evaporation condensation precipitation runoff",
                "hydrologic cycle groundwater infiltration",
            ]
        else:
            queries = [
                core,
                f"{core} science explanation process stages",
                f"{core} facts educational source NOAA USGS Britannica",
            ]
        sources: list[ResearchSource] = []
        seen: set[str] = set()
        topic_terms = _topic_terms(core)
        provider = (self.settings.research_provider or "duckduckgo").strip().lower()
        for query in queries:
            try:
                for item in self.search(query, limit=max(2, self.settings.research_max_results // 2)):
                    # Search endpoints can return redirect/login/geo results that have
                    # nothing to do with the brief. Keep only topic-relevant evidence.
                    if topic_terms and not _matches_topic(item, topic_terms):
                        continue
                    key = item.url.lower()
                    if key not in seen:
                        seen.add(key)
                        sources.append(item)
                    if len(sources) >= self.settings.research_max_results:
                        break
            except (PermanentProviderError, TransientProviderError) as exc:
                log.warning("research query failed (%s): %s", query, exc)
            if len(sources) >= self.settings.research_max_results:
                break

        # A search page may be blocked or locale-poisoned. Wikipedia's API is a
        # deterministic no-key fallback, and is especially useful for Chinese topics.
        if len(sources) < min(2, self.settings.research_max_results):
            try:
                fallback_query = topic_terms[0] if topic_terms else core
                for item in self._wikipedia(fallback_query, self.settings.research_max_results):
                    if topic_terms and not _matches_topic(item, topic_terms):
                        continue
                    key = item.url.lower()
                    if key not in seen:
                        seen.add(key)
                        sources.append(item)
                    if len(sources) >= self.settings.research_max_results:
                        break
            except (PermanentProviderError, TransientProviderError) as exc:
                log.warning("wikipedia fallback failed: %s", exc)

        return {
            "topic": topic,
            "provider": self.settings.research_provider,
            "queries": queries,
            "sources": [s.as_dict() for s in sources],
            "source_count": len(sources),
        }

    def _duckduckgo(self, query: str, limit: int) -> list[ResearchSource]:
        try:
            response = httpx.get(
                f"https://html.duckduckgo.com/html/?q={quote_plus(query)}",
                headers={"User-Agent": "StoryboardAgent/0.1 research bot"},
                timeout=self.settings.research_timeout_seconds,
                follow_redirects=True,
            )
            if response.status_code == 429 or response.status_code >= 500:
                raise TransientProviderError(f"DuckDuckGo HTTP {response.status_code}")
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise TransientProviderError(f"DuckDuckGo timeout: {exc}") from exc
        except httpx.HTTPError as exc:
            raise PermanentProviderError(f"DuckDuckGo request failed: {exc}") from exc
        text = response.text
        pattern = re.compile(
            r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>.*?'
            r'<(?:a|div)[^>]+class="result__snippet"[^>]*>(.*?)</(?:a|div)>',
            re.IGNORECASE | re.DOTALL,
        )
        out: list[ResearchSource] = []
        for raw_url, raw_title, raw_snippet in pattern.findall(text):
            url = html.unescape(raw_url)
            if "uddg=" in url:
                try:
                    url = unquote(url.split("uddg=", 1)[1].split("&", 1)[0])
                except Exception:
                    pass
            title = _clean_html(raw_title)
            snippet = _clean_html(raw_snippet)
            if urlparse(url).scheme in {"http", "https"} and title:
                out.append(ResearchSource(title, url, snippet, "duckduckgo"))
            if len(out) >= limit:
                break
        return out

    def _bing(self, query: str, limit: int) -> list[ResearchSource]:
        try:
            response = httpx.get(
                f"https://www.bing.com/search?q={quote_plus(query)}",
                headers={"User-Agent": "Mozilla/5.0 StoryboardAgent/0.1"},
                timeout=self.settings.research_timeout_seconds,
                follow_redirects=True,
            )
            if response.status_code == 429 or response.status_code >= 500:
                raise TransientProviderError(f"Bing HTTP {response.status_code}")
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise TransientProviderError(f"Bing timeout: {exc}") from exc
        except httpx.HTTPError as exc:
            raise PermanentProviderError(f"Bing request failed: {exc}") from exc
        blocks = re.findall(r'<li class="b_algo".*?</li>', response.text, re.IGNORECASE | re.DOTALL)
        out: list[ResearchSource] = []
        for block in blocks:
            link = re.search(r'<h2[^>]*>.*?<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', block, re.IGNORECASE | re.DOTALL)
            snippet_match = re.search(r'<div class="b_caption".*?<p[^>]*>(.*?)</p>', block, re.IGNORECASE | re.DOTALL)
            if not link:
                continue
            raw_url, raw_title = link.groups()
            url = html.unescape(raw_url)
            try:
                encoded = parse_qs(urlparse(url).query).get("u", [""])[0]
                if encoded.startswith("a1"):
                    padded = encoded[2:] + "=" * (-len(encoded[2:]) % 4)
                    url = base64.b64decode(padded).decode("utf-8", errors="ignore")
            except Exception:
                pass
            title = _clean_html(raw_title)
            snippet = _clean_html(snippet_match.group(1) if snippet_match else "")
            if urlparse(url).scheme in {"http", "https"} and title:
                out.append(ResearchSource(title, url, snippet, "bing"))
            if len(out) >= limit:
                break
        return out

    def _tavily(self, query: str, limit: int) -> list[ResearchSource]:
        key = self.settings.tavily_api_key or os.getenv("TAVILY_API_KEY", "")
        if not key:
            raise PermanentProviderError("TAVILY_API_KEY 未配置")
        response = httpx.post(
            "https://api.tavily.com/search",
            json={"api_key": key, "query": query, "max_results": limit, "search_depth": "advanced"},
            timeout=self.settings.research_timeout_seconds,
        )
        if response.status_code == 429 or response.status_code >= 500:
            raise TransientProviderError(f"Tavily HTTP {response.status_code}")
        response.raise_for_status()
        return [
            ResearchSource(x.get("title", ""), x.get("url", ""), x.get("content", ""), "tavily")
            for x in response.json().get("results", [])
            if x.get("url")
        ][:limit]

    def _wikipedia(self, query: str, limit: int) -> list[ResearchSource]:
        """Search Wikipedia and return short plain-text excerpts without an API key."""
        language = "zh" if re.search(r"[\u4e00-\u9fff]", query) else "en"
        endpoint = f"https://{language}.wikipedia.org/w/api.php"
        try:
            response = httpx.get(
                endpoint,
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "srlimit": limit,
                    "format": "json",
                    "utf8": 1,
                },
                headers={"User-Agent": "StoryboardAgent/0.1 (research)"},
                timeout=self.settings.research_timeout_seconds,
            )
            if response.status_code == 429 or response.status_code >= 500:
                raise TransientProviderError(f"Wikipedia HTTP {response.status_code}")
            response.raise_for_status()
            hits = response.json().get("query", {}).get("search", [])[:limit]
        except httpx.TimeoutException as exc:
            raise TransientProviderError(f"Wikipedia timeout: {exc}") from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise PermanentProviderError(f"Wikipedia request failed: {exc}") from exc

        out: list[ResearchSource] = []
        for hit in hits:
            title = str(hit.get("title", "")).strip()
            if not title:
                continue
            snippet = _clean_html(str(hit.get("snippet", "")))
            out.append(
                ResearchSource(
                    title=title,
                    url=f"https://{language}.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}",
                    snippet=snippet[:800],
                    source="wikipedia",
                )
            )
        return out


    def _serpapi(self, query: str, limit: int) -> list[ResearchSource]:
        key = self.settings.serpapi_api_key or os.getenv("SERPAPI_API_KEY", "")
        if not key:
            raise PermanentProviderError("SERPAPI_API_KEY 未配置")
        response = httpx.get(
            "https://serpapi.com/search.json",
            params={"engine": "google", "q": query, "api_key": key, "num": limit},
            timeout=self.settings.research_timeout_seconds,
        )
        if response.status_code == 429 or response.status_code >= 500:
            raise TransientProviderError(f"SerpAPI HTTP {response.status_code}")
        response.raise_for_status()
        return [
            ResearchSource(x.get("title", ""), x.get("link", ""), x.get("snippet", ""), "serpapi")
            for x in response.json().get("organic_results", [])
            if x.get("link")
        ][:limit]


def _clean_html(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", value))).strip()


def available_research_profiles(settings: Settings) -> list[dict[str, Any]]:
    return [
        {"id": "none", "label": "不联网检索", "configured": True},
        {"id": "bing_html", "label": "Bing（无需 API Key）", "configured": True},
        {"id": "duckduckgo", "label": "DuckDuckGo（无需 API Key）", "configured": True},
        {"id": "tavily", "label": "Tavily（高质量检索）", "configured": bool(settings.tavily_api_key or os.getenv("TAVILY_API_KEY"))},
        {"id": "serpapi", "label": "SerpAPI（Google 结果）", "configured": bool(settings.serpapi_api_key or os.getenv("SERPAPI_API_KEY"))},
        {"id": "wikipedia", "label": "Wikipedia（无 Key 兜底）", "configured": True},
    ]


def _topic_terms(core: str) -> list[str]:
    """Return compact multilingual terms for rejecting unrelated search noise."""
    value = re.sub(r"(?:的一生|的发展史|发展史|的历史|是什么|怎么回事)$", "", core).strip()
    if "水循环" in value:
        return [value, "water cycle", "hydrologic cycle"]
    return [value] if len(value) >= 2 else ([core] if core else [])


def _matches_topic(item: ResearchSource, terms: list[str]) -> bool:
    haystack = f"{item.title} {item.snippet} {item.url}".casefold()
    return any(term.casefold() in haystack for term in terms)


def _extract_core_topic(text: str) -> str:
    """Turn a long creative brief into a search-friendly topic phrase."""
    quoted = re.search(r"主题是[“\"]([^”\"]+)[”\"]", text)
    if quoted:
        return quoted.group(1).strip()
    lowered = text.split("：", 1)[0].strip()
    return lowered[:80] or text[:80]
