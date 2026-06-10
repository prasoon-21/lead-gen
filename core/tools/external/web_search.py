import math
from typing import Any, Dict, List
from urllib.parse import urlparse

from core.tools.base import BaseTool, ToolContext, ToolExecutionError
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
        "clutch.co",
        "www.clutch.co",
        "goodfirms.co",
        "www.goodfirms.co",
        "yelp.com",
        "www.yelp.com",
        "builtin.com",
        "www.builtin.com",
        "wellfound.com",
        "www.wellfound.com",
    )

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

        expand_queries = bool(arguments.get("expand_queries", True))
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

    async def run(self, arguments: Dict[str, Any], context: ToolContext) -> Dict[str, Any]:
        query = arguments.get("query", "").strip()
        if not query:
            raise ToolExecutionError("query is required", code="invalid_arguments")

        query_variants = self._build_query_variants(arguments, context)
        max_results = int(arguments.get("max_results", 12))
        max_results_per_query = int(
            arguments.get("max_results_per_query")
            or max(4, math.ceil(max_results / max(1, len(query_variants))))
        )

        payload_base = {
            "search_depth": arguments.get("search_depth", "advanced"),
            "include_domains": arguments.get("include_domains") or [],
            "exclude_domains": self._merged_excluded_domains(arguments.get("exclude_domains") or []),
            "include_raw_content": bool(arguments.get("include_raw_content", False)),
        }

        client = TavilyClient()
        aggregated: List[Dict[str, Any]] = []
        answer_fragments: List[str] = []
        response_times: List[Any] = []

        for variant in query_variants:
            raw = await client.search(
                {
                    "query": variant,
                    "max_results": max_results_per_query,
                    **payload_base,
                }
            )
            normalized_results = self._normalize_results(raw.get("results", []))
            aggregated.extend(normalized_results)
            if raw.get("answer"):
                answer_fragments.append(str(raw["answer"]))
            if raw.get("response_time") is not None:
                response_times.append(raw.get("response_time"))

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
            "answer": "\n".join(answer_fragments[:3]).strip() or None,
            "results": deduped,
            "count": len(deduped),
            "response_time": response_times,
            "domains": domains,
        }
