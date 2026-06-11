from core.tools.registry import ToolRegistry
from core.tools.external.http_request import HTTPRequestTool
from core.tools.external.web_search import WebSearchTool
from core.tools.external.linkedin_research import LinkedInResearchTool
from core.tools.internal.file_analyze import FileAnalyzeTool
from core.tools.internal.image_generate import ImageGenerateTool
from core.tools.internal.json_extract import JSONExtractTool
from core.tools.internal.rag_search import RAGSearchTool
from core.tools.internal.todo_tools import (
    TodoEnsureStoreTool,
    TodoCreateTool,
    TodoListTool,
    TodoUpdateTool,
    TodoCompleteTool,
)
from core.tools.internal.lead_quality import LeadQualityTool
from core.tools.remote_loader import load_remote_tools


def build_default_tool_registry(
    remote_tools_dir: str = "config/remote_tools",
) -> ToolRegistry:
    registry = ToolRegistry()

    # ── Built-in local tools ──────────────────────────────────────────────────
    registry.register(RAGSearchTool())
    registry.register(FileAnalyzeTool())
    registry.register(ImageGenerateTool())
    registry.register(JSONExtractTool())
    registry.register(HTTPRequestTool())
    registry.register(WebSearchTool())
    registry.register(LinkedInResearchTool())
    registry.register(TodoEnsureStoreTool())
    registry.register(TodoCreateTool())
    registry.register(TodoListTool())
    registry.register(TodoUpdateTool())
    registry.register(TodoCompleteTool())
    registry.register(LeadQualityTool())

    # ── Remote tools from config/remote_tools/*.yaml ─────────────────────────
    for tool in load_remote_tools(remote_tools_dir):
        registry.register(tool)

    return registry
