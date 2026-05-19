import json
from typing import Any, Dict, List

from core.tools.base import BaseTool, ToolContext, ToolExecutionError
from core.tools.external.tavily_client import TavilyClient


DEFAULT_RESEARCH_SCHEMA = {
    "summary": "Concise synthesis of findings",
    "key_points": ["List of key facts"],
    "sources": [{"title": "Source title", "url": "https://...", "relevance": "why this source matters"}],
    "confidence": "low|medium|high",
    "gaps": ["What could not be verified"],
}


class WebResearchTool(BaseTool):
    name = "web_research"
    description = "Run deep web research and return structured, cited synthesis."
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer"},
            "search_depth": {"type": "string"},
            "include_domains": {"type": "array"},
            "exclude_domains": {"type": "array"},
            "output_schema": {"type": "object"},
        },
        "required": ["query"],
    }
    timeout_seconds = 50

    @staticmethod
    def _simplify_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        simplified = []
        for item in results or []:
            simplified.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", "")[:2000],
                    "score": item.get("score"),
                }
            )
        return simplified

    def _build_synthesis_prompt(
        self,
        query: str,
        results: List[Dict[str, Any]],
        output_schema: Dict[str, Any],
    ) -> str:
        packed = json.dumps(results[:8], ensure_ascii=False)
        schema_text = json.dumps(output_schema, ensure_ascii=False, indent=2)
        return (
            "You are a research synthesizer. Build a factual, source-cited report.\n"
            "Use only the provided web results. If evidence is weak, lower confidence.\n"
            "Return JSON only and match this schema shape:\n"
            f"{schema_text}\n\n"
            f"Research Query: {query}\n"
            f"Web Results JSON:\n{packed}\n\n"
            "Citation rule: Include source URLs in `sources` and reference only real URLs from input."
        )

    async def run(self, arguments: Dict[str, Any], context: ToolContext) -> Dict[str, Any]:
        query = arguments.get("query", "").strip()
        if not query:
            raise ToolExecutionError("query is required", code="invalid_arguments")

        payload = {
            "query": query,
            "max_results": int(arguments.get("max_results", 8)),
            "search_depth": arguments.get("search_depth", "advanced"),
            "include_domains": arguments.get("include_domains") or [],
            "exclude_domains": arguments.get("exclude_domains") or [],
        }

        client = TavilyClient()
        raw = None
        try:
            raw = await client.research(payload)
        except Exception:
            raw = await client.search(payload)

        results = self._simplify_results(raw.get("results", []))
        output_schema = arguments.get("output_schema") or DEFAULT_RESEARCH_SCHEMA
        adapter = context.resources.get("adapter")

        if not adapter:
            return {
                "report": {
                    "summary": raw.get("answer") or "Research completed without synthesis model.",
                    "key_points": [],
                    "sources": [{"title": r.get("title", ""), "url": r.get("url", ""), "relevance": "search_result"} for r in results[:5]],
                    "confidence": "medium" if results else "low",
                    "gaps": [],
                },
                "results": results,
                "count": len(results),
            }

        report = await adapter.generate_json(
            prompt=self._build_synthesis_prompt(query, results, output_schema),
            temperature=0.2,
        )
        return {
            "report": report,
            "results": results,
            "count": len(results),
            "query": query,
        }
