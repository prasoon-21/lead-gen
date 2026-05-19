"""
Studio API — helpers for the Studio UI (tool catalogue, etc.)

Endpoints:
    GET  /api/studio/tools   list every tool registered in the runtime registry
"""

from typing import Any, Dict, List

from fastapi import APIRouter

from api.state import get_state

router = APIRouter()


@router.get("/tools", summary="List all registered tools")
async def list_tools() -> List[Dict[str, Any]]:
    state = get_state()
    if not state.tool_registry:
        return []

    tools = []
    for tool in state.tool_registry.all_tools():
        schema = tool.input_schema or {}
        props = schema.get("properties", {})
        required = schema.get("required", [])

        params = []
        for param_name, prop in props.items():
            params.append({
                "name": param_name,
                "type": prop.get("type", "string"),
                "description": prop.get("description", ""),
                "required": param_name in required,
            })

        tools.append({
            "name": tool.name,
            "description": tool.description,
            "category": _infer_category(tool.name),
            "params": params,
            "requires_auth": _requires_auth(tool.name),
        })

    tools.sort(key=lambda t: (t["category"], t["name"]))
    return tools


def _infer_category(name: str) -> str:
    if name.startswith("todo"):
        return "productivity"
    if name in ("rag_search",):
        return "knowledge"
    if name in ("http_request",):
        return "api"
    if name in ("web_search", "web_research"):
        return "web"
    if name in ("image_generate",):
        return "media"
    if name in ("file_analyze", "json_extract"):
        return "data"
    return "other"


def _requires_auth(name: str) -> bool:
    return name in ("http_request", "web_search", "web_research", "rag_search", "image_generate")
