from typing import Any, Dict

from core.extraction.json_extractor import JSONExtractor
from core.tools.base import BaseTool, ToolContext, ToolExecutionError


class JSONExtractTool(BaseTool):
    name = "json_extract"
    description = "Extract structured JSON from free text."
    input_schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "schema": {"type": "string"},
            "system_prompt": {"type": "string"},
        },
        "required": ["text", "schema"],
    }
    timeout_seconds = 20

    async def run(self, arguments: Dict[str, Any], context: ToolContext) -> Dict[str, Any]:
        adapter = context.resources.get("adapter")
        if not adapter:
            raise ToolExecutionError("LLM adapter is not configured", code="adapter_unavailable")
        extractor = JSONExtractor(adapter)
        result = await extractor.extract(
            text=arguments["text"],
            schema_description=arguments["schema"],
            system_prompt=arguments.get("system_prompt"),
        )
        return {"json": result}
