from typing import Any, Dict

from core.services.file_analyzer import FileAnalyzer
from core.tools.base import BaseTool, ToolContext, ToolExecutionError


class FileAnalyzeTool(BaseTool):
    name = "file_analyze"
    description = "Analyze uploaded document/image/pdf content with vision model."
    input_schema = {
        "type": "object",
        "properties": {
            "document": {"type": "object"},
            "instruction": {"type": "string"},
            "user_query": {"type": "string"},
            "mode": {"type": "string"},
        },
        "required": ["document"],
    }
    timeout_seconds = 40

    async def run(self, arguments: Dict[str, Any], context: ToolContext) -> Dict[str, Any]:
        adapter = context.resources.get("adapter")
        if not adapter:
            raise ToolExecutionError("LLM adapter is not configured", code="adapter_unavailable")

        analyzer = FileAnalyzer(adapter)
        document = arguments["document"]
        mode = (arguments.get("mode") or "analyze").strip().lower()

        if mode == "tutor":
            text = await analyzer.analyze_for_tutoring(document=document, user_query=arguments.get("user_query", ""))
            return {"text": text, "mode": mode}

        instruction = arguments.get("instruction") or "Extract key information from the document."
        response = await analyzer.analyze(
            document=document,
            instruction=instruction,
            user_query=arguments.get("user_query"),
        )
        return {
            "text": response.text,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "latency_ms": response.latency_ms,
            "mode": mode,
        }
