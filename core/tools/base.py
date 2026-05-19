import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


class ToolExecutionError(Exception):
    def __init__(self, message: str, code: str = "tool_error", details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


@dataclass
class ToolContext:
    session_id: str
    trace_id: str
    resources: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    ok: bool
    tool_name: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "tool_name": self.tool_name,
            "data": self.data or {},
            "error": self.error,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
        }


class BaseTool(ABC):
    name: str = ""
    description: str = ""
    input_schema: Dict[str, Any] = {}
    timeout_seconds: int = 20

    def validate_input(self, arguments: Dict[str, Any]) -> None:
        schema = self.input_schema or {}
        if not schema:
            return

        if schema.get("type") == "object":
            required = schema.get("required", []) or []
            for field_name in required:
                if field_name not in arguments:
                    raise ToolExecutionError(
                        f"Missing required field: {field_name}",
                        code="invalid_arguments",
                    )

    async def execute(self, arguments: Dict[str, Any], context: ToolContext) -> ToolResult:
        self.validate_input(arguments)
        start = time.perf_counter()
        try:
            data = await self.run(arguments, context)
            return ToolResult(
                ok=True,
                tool_name=self.name,
                data=data if isinstance(data, dict) else {"result": data},
                latency_ms=(time.perf_counter() - start) * 1000,
            )
        except ToolExecutionError as exc:
            return ToolResult(
                ok=False,
                tool_name=self.name,
                error={"code": exc.code, "message": str(exc), "details": exc.details},
                latency_ms=(time.perf_counter() - start) * 1000,
            )
        except Exception as exc:  # pragma: no cover - safety fallback
            return ToolResult(
                ok=False,
                tool_name=self.name,
                error={"code": "tool_runtime_error", "message": str(exc), "details": {}},
                latency_ms=(time.perf_counter() - start) * 1000,
            )

    @abstractmethod
    async def run(self, arguments: Dict[str, Any], context: ToolContext) -> Dict[str, Any]:
        raise NotImplementedError
