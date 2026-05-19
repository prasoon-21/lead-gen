from dataclasses import dataclass, field
from typing import Dict, Optional, Set


@dataclass
class ToolPolicy:
    allowed_tools: Set[str] = field(default_factory=set)
    max_tool_calls: int = 6
    default_timeout_seconds: int = 20
    per_tool_timeout_seconds: Dict[str, int] = field(default_factory=dict)

    def is_allowed(self, tool_name: str) -> bool:
        if not self.allowed_tools:
            return True
        return tool_name in self.allowed_tools

    def timeout_for(self, tool_name: str, tool_default: Optional[int] = None) -> int:
        if tool_name in self.per_tool_timeout_seconds:
            return self.per_tool_timeout_seconds[tool_name]
        if tool_default:
            return tool_default
        return self.default_timeout_seconds
