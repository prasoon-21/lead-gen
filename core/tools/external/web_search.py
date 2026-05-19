from typing import Any, Dict, List

from core.tools.base import BaseTool, ToolContext, ToolExecutionError
from core.tools.external.tavily_client import TavilyClient


class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Search the web and return normalized results with sources."
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer"},
            "search_depth": {"type": "string"},
            "include_domains": {"type": "array"},
            "exclude_domains": {"type": "array"},
            "include_raw_content": {"type": "boolean"},
        },
        "required": ["query"],
    }
    timeout_seconds = 35

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

    async def run(self, arguments: Dict[str, Any], context: ToolContext) -> Dict[str, Any]:
        query = arguments.get("query", "").strip()
        if not query:
            raise ToolExecutionError("query is required", code="invalid_arguments")

        payload = {
            "query": query,
            "max_results": int(arguments.get("max_results", 5)),
            "search_depth": arguments.get("search_depth", "advanced"),
            "include_domains": arguments.get("include_domains") or [],
            "exclude_domains": arguments.get("exclude_domains") or [],
            "include_raw_content": bool(arguments.get("include_raw_content", False)),
        }

        client = TavilyClient()
        raw = await client.search(payload)
        normalized_results = self._normalize_results(raw.get("results", []))

        return {
            "query": query,
            "answer": raw.get("answer"),
            "results": normalized_results,
            "count": len(normalized_results),
            "response_time": raw.get("response_time"),
        }
