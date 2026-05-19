from typing import Dict, Iterable, List, Optional

from core.tools.base import BaseTool


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        if not tool.name:
            raise ValueError("Tool name is required")
        self._tools[tool.name] = tool

    def get(self, tool_name: str) -> Optional[BaseTool]:
        return self._tools.get(tool_name)

    def all_tools(self) -> Iterable[BaseTool]:
        return self._tools.values()

    def list_specs(self, allowed_tools: Optional[Iterable[str]] = None) -> List[Dict]:
        allowed = set(allowed_tools or [])
        specs = []
        for tool in self._tools.values():
            if allowed and tool.name not in allowed:
                continue
            specs.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema or {"type": "object", "properties": {}},
                }
            )
        return specs
