from typing import Any, Dict

from core.tools.base import BaseTool, ToolContext, ToolExecutionError


class RAGSearchTool(BaseTool):
    name = "rag_search"
    description = "Search the vector store and return relevant chunks."
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "context": {"type": "object"},
            "collection_name": {"type": "string"},
            "top_k": {"type": "integer"},
            "filters": {"type": "object"},
        },
        "required": ["query", "collection_name"],
    }
    timeout_seconds = 25

    async def run(self, arguments: Dict[str, Any], context: ToolContext) -> Dict[str, Any]:
        retriever = context.resources.get("retriever")
        config_loader = context.resources.get("config_loader")
        if not retriever:
            raise ToolExecutionError("Retriever is not configured", code="retriever_unavailable")

        query = arguments["query"]
        collection_name = arguments["collection_name"]
        call_context = arguments.get("context", {}) or {}
        top_k = arguments.get("top_k")
        filter_config = None

        filters = arguments.get("filters")
        if filters and config_loader:
            filter_config = config_loader.parse_filter_config(filters)

        results = await retriever.search(
            query=query,
            context=call_context,
            top_k=top_k,
            collection_name=collection_name,
            filter_config=filter_config,
        )
        return {"results": results, "count": len(results)}
