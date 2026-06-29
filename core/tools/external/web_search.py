import math
import os
from typing import Any, Dict, List
from urllib.parse import urlparse

from core.services.lead_discovery_policy import (
    DEFAULT_TAVILY_SEARCH_DEPTH,
    EXCLUDED_LEAD_SOURCE_DOMAINS,
    clamp_web_search_max_results,
)
from core.tools.base import BaseTool, ToolContext, ToolExecutionError
from core.tools.external.html_search_fallback import HTMLSearchFallbackClient
from core.tools.external.tavily_client import TavilyClient


class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Search the web with optional multi-query expansion and return normalized results with sources."
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer"},
            "max_results_per_query": {"type": "integer"},
            "search_depth": {"type": "string"},
            "include_domains": {"type": "array"},
            "exclude_domains": {"type": "array"},
            "include_raw_content": {"type": "boolean"},
            "industry": {"type": "string"},
            "location": {"type": "string"},
            "company_name": {"type": "string"},
            "expand_queries": {"type": "boolean"},
            "query_variants": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["query"],
    }
    timeout_seconds = 35
    _variant_limit = 6
    _default_query_patterns = (
        "{query}",
        "{query} company",
        "{query} official website",
        "{query} founder OR CEO",
    )
    _industry_location_patterns = (
        "{industry} companies in {location}",
        "{industry} firms in {location}",
        "{industry} agencies in {location}",
        "{industry} services in {location}",
        "\"{industry}\" \"{location}\" company",
        "\"{industry}\" \"{location}\" founder",
    )
    _default_excluded_domains = (
        *EXCLUDED_LEAD_SOURCE_DOMAINS,
        "linkedin.com",
        "www.linkedin.com",
        "facebook.com",
        "www.facebook.com",
        "instagram.com",
        "www.instagram.com",
        "crunchbase.com",
        "www.crunchbase.com",
        "zoominfo.com",
        "www.zoominfo.com",
    )
    _fallback_codes = {
        "missing_api_key",
        "tavily_error",
        "tavily_quota_exceeded",
        "invalid_api_key",
    }

    @staticmethod
    def _normalize_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized = []
        for item in results or []:
            normalized.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                    "score": item.get("score"),
                    "published_date": item.get("published_date"),
                }
            )
        return normalized

    @staticmethod
    def _normalize_query(text: str) -> str:
        return " ".join((text or "").strip().split())

    def _build_query_variants(self, arguments: Dict[str, Any], context: ToolContext) -> List[str]:
        query = self._normalize_query(arguments.get("query", ""))
        industry = self._normalize_query(arguments.get("industry") or context.state.get("industry") or "")
        location = self._normalize_query(arguments.get("location") or context.state.get("location") or "")
        company_name = self._normalize_query(arguments.get("company_name") or context.state.get("company_name") or "")
        explicit_variants = arguments.get("query_variants") or []

        variants: List[str] = []

        def add_variant(text: str) -> None:
            normalized = self._normalize_query(text)
            if normalized and normalized not in variants:
                variants.append(normalized)

        add_variant(query)
        for variant in explicit_variants:
            add_variant(str(variant))

        expand_queries = bool(arguments.get("expand_queries", False))
        if not expand_queries:
            return variants[: self._variant_limit]

        if industry and location:
            for pattern in self._industry_location_patterns:
                add_variant(pattern.format(industry=industry, location=location))
        else:
            for pattern in self._default_query_patterns:
                add_variant(pattern.format(query=query))

        if company_name:
            add_variant(f"{company_name} official website")
            add_variant(f"{company_name} founder")
            add_variant(f"{company_name} LinkedIn")

        return variants[: self._variant_limit]

    @staticmethod
    def _result_key(item: Dict[str, Any]) -> str:
        url = (item.get("url") or "").strip()
        if url:
            return url.lower()
        title = (item.get("title") or "").strip().lower()
        content = (item.get("content") or "").strip().lower()[:120]
        return f"{title}|{content}"

    @staticmethod
    def _source_domain(item: Dict[str, Any]) -> str:
        return (urlparse(item.get("url") or "").hostname or "").lower()

    @classmethod
    def _merged_excluded_domains(cls, supplied: List[str]) -> List[str]:
        merged: List[str] = []
        for domain in list(cls._default_excluded_domains) + list(supplied or []):
            cleaned = (domain or "").strip().lower()
            if cleaned and cleaned not in merged:
                merged.append(cleaned)
        return merged

    @classmethod
    def _quality_boost(cls, item: Dict[str, Any]) -> int:
        domain = cls._source_domain(item)
        title = str(item.get("title") or "").lower()
        content = str(item.get("content") or "").lower()
        boost = 0
        if domain and domain not in cls._default_excluded_domains:
            boost += 2
        if "official" in title or "official" in content:
            boost += 2
        if any(term in title for term in ("about", "team", "leadership", "contact")):
            boost += 1
        return boost

    async def _search_variant(
        self,
        variant: str,
        payload: Dict[str, Any],
        tavily_client: TavilyClient | None,
        fallback_client: HTMLSearchFallbackClient,
        providers_used: List[str],
        warnings: List[str],
    ) -> Dict[str, Any]:
        tavily_error: ToolExecutionError | None = None

        if tavily_client is not None:
            try:
                result = await tavily_client.search(payload)
                providers_used.append("tavily")
                return result
            except ToolExecutionError as exc:
                tavily_error = exc
                if exc.code not in self._fallback_codes:
                    raise
                warnings.append(
                    f"Tavily search failed for '{variant}': "
                    f"{exc.details.get('message') or str(exc)}"
                )

        try:
            result = await fallback_client.search(payload)
            providers_used.append(result.get("provider", HTMLSearchFallbackClient.PROVIDER_NAME))
            return result
        except ToolExecutionError as fallback_exc:
            if tavily_error is not None:
                raise ToolExecutionError(
                    "Primary and fallback web search both failed",
                    code="search_unavailable",
                    details={
                        "query": variant,
                        "primary_error": {
                            "code": tavily_error.code,
                            "message": str(tavily_error),
                            "details": tavily_error.details,
                        },
                        "fallback_error": {
                            "code": fallback_exc.code,
                            "message": str(fallback_exc),
                            "details": fallback_exc.details,
                        },
                    },
                )
            raise

    async def run(self, arguments: Dict[str, Any], context: ToolContext) -> Dict[str, Any]:
        query = arguments.get("query", "").strip()
        if not query:
            raise ToolExecutionError("query is required", code="invalid_arguments")

        query_variants = self._build_query_variants(arguments, context)
        max_results = clamp_web_search_max_results(arguments.get("max_results"))
        max_results_per_query = int(
            arguments.get("max_results_per_query")
            or max(2, math.ceil(max_results / max(1, len(query_variants))))
        )
        max_results_per_query = max(1, min(max_results_per_query, 4))

        payload_base = {
            "search_depth": DEFAULT_TAVILY_SEARCH_DEPTH,
            "include_domains": arguments.get("include_domains") or [],
            "exclude_domains": self._merged_excluded_domains(arguments.get("exclude_domains") or []),
            "include_raw_content": bool(arguments.get("include_raw_content", False)),
        }

        tavily_client = TavilyClient() if os.getenv("TAVILY_API_KEY") else None
        fallback_client = HTMLSearchFallbackClient()
        aggregated: List[Dict[str, Any]] = []
        answer_fragments: List[str] = []
        response_times: List[Any] = []
        providers_used: List[str] = []
        warnings: List[str] = []
        last_error: ToolExecutionError | None = None

        for variant in query_variants:
            try:
                raw = await self._search_variant(
                    variant=variant,
                    payload={
                        "query": variant,
                        "max_results": max_results_per_query,
                        **payload_base,
                    },
                    tavily_client=tavily_client,
                    fallback_client=fallback_client,
                    providers_used=providers_used,
                    warnings=warnings,
                )
            except ToolExecutionError as exc:
                last_error = exc
                warnings.append(f"Search variant failed for '{variant}': {str(exc)}")
                continue
            normalized_results = self._normalize_results(raw.get("results", []))
            aggregated.extend(normalized_results)
            if raw.get("answer"):
                answer_fragments.append(str(raw["answer"]))
            if raw.get("response_time") is not None:
                response_times.append(raw.get("response_time"))

        if not aggregated and last_error is not None:
            raise last_error

        deduped: List[Dict[str, Any]] = []
        seen = set()
        for item in sorted(
            aggregated,
            key=lambda result: (result.get("score") or 0) + self._quality_boost(result),
            reverse=True,
        ):
            key = self._result_key(item)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
            if len(deduped) >= max_results:
                break

        domains = []
        seen_domains = set()
        for item in deduped:
            domain = self._source_domain(item)
            if domain and domain not in seen_domains:
                seen_domains.add(domain)
                domains.append(domain)

        return {
            "query": query,
            "queries_used": query_variants,
            "search_depth": DEFAULT_TAVILY_SEARCH_DEPTH,
            "max_results": max_results,
            "answer": "\n".join(answer_fragments[:3]).strip() or None,
            "results": deduped,
            "count": len(deduped),
            "response_time": response_times,
            "domains": domains,
            "providers_used": providers_used,
            "fallback_used": any(provider != "tavily" for provider in providers_used),
            "warnings": warnings,
        }
