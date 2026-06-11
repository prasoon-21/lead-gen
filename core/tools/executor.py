import asyncio
import logging
import json
from typing import Any, Dict, Optional

from core.tools.base import ToolContext, ToolResult
from core.tools.policy import ToolPolicy
from core.tools.registry import ToolRegistry


class ToolExecutor:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self._logger = logging.getLogger("agent.runtime")

    @staticmethod
    def _preview_payload(value: Any, limit: int = 1000) -> str:
        try:
            text = json.dumps(value, ensure_ascii=False)
        except Exception:
            text = str(value)
        if len(text) > limit:
            return text[:limit] + "...[truncated]"
        return text

    async def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        context: ToolContext,
        policy: Optional[ToolPolicy] = None,
    ) -> ToolResult:
        policy = policy or ToolPolicy()
        trace_id = context.trace_id
        session_id = context.session_id
        if not policy.is_allowed(tool_name):
            self._logger.info(
                "tool_blocked",
                extra={
                    "event": "tool_blocked",
                    "trace_id": trace_id,
                    "session_id": session_id,
                    "agent_id": context.state.get("agent_id", "-"),
                    "workflow_id": "-",
                    "step": 0,
                    "tool_name": tool_name,
                    "status": "not_allowed",
                    "payload": self._preview_payload({"arguments": arguments}),
                },
            )
            return ToolResult(
                ok=False,
                tool_name=tool_name,
                error={"code": "tool_not_allowed", "message": f"Tool '{tool_name}' is not allowed", "details": {}},
            )

        tool = self.registry.get(tool_name)
        if not tool:
            self._logger.info(
                "tool_missing",
                extra={
                    "event": "tool_missing",
                    "trace_id": trace_id,
                    "session_id": session_id,
                    "agent_id": context.state.get("agent_id", "-"),
                    "workflow_id": "-",
                    "step": 0,
                    "tool_name": tool_name,
                    "status": "not_registered",
                    "payload": self._preview_payload({"arguments": arguments}),
                },
            )
            return ToolResult(
                ok=False,
                tool_name=tool_name,
                error={"code": "tool_not_found", "message": f"Tool '{tool_name}' is not registered", "details": {}},
            )

        timeout_seconds = policy.timeout_for(tool_name, tool.timeout_seconds)
        self._logger.info(
            "tool_execute_start",
            extra={
                "event": "tool_execute_start",
                "trace_id": trace_id,
                "session_id": session_id,
                "agent_id": context.state.get("agent_id", "-"),
                "workflow_id": "-",
                "step": 0,
                "tool_name": tool_name,
                "status": f"timeout={timeout_seconds}",
                "payload": self._preview_payload({"arguments": arguments}),
            },
        )
        try:
            result = await asyncio.wait_for(tool.execute(arguments or {}, context), timeout=timeout_seconds)
            self._logger.info(
                "tool_execute_end",
                extra={
                    "event": "tool_execute_end",
                    "trace_id": trace_id,
                    "session_id": session_id,
                    "agent_id": context.state.get("agent_id", "-"),
                    "workflow_id": "-",
                    "step": 0,
                    "tool_name": tool_name,
                    "status": f"ok={result.ok} latency_ms={int(result.latency_ms)}",
                    "payload": self._preview_payload(result.to_dict()),
                },
            )
            return result
        except asyncio.TimeoutError:
            self._logger.info(
                "tool_execute_timeout",
                extra={
                    "event": "tool_execute_timeout",
                    "trace_id": trace_id,
                    "session_id": session_id,
                    "agent_id": context.state.get("agent_id", "-"),
                    "workflow_id": "-",
                    "step": 0,
                    "tool_name": tool_name,
                    "status": f"timeout={timeout_seconds}",
                    "payload": self._preview_payload({"arguments": arguments}),
                },
            )
            return ToolResult(
                ok=False,
                tool_name=tool_name,
                error={
                    "code": "tool_timeout",
                    "message": f"Tool '{tool_name}' timed out after {timeout_seconds}s",
                    "details": {"timeout_seconds": timeout_seconds},
                },
            )
        except Exception as exc:
            self._logger.exception(
                "tool_execute_error",
                extra={
                    "event": "tool_execute_error",
                    "trace_id": trace_id,
                    "session_id": session_id,
                    "agent_id": context.state.get("agent_id", "-"),
                    "workflow_id": "-",
                    "step": 0,
                    "tool_name": tool_name,
                    "status": type(exc).__name__,
                    "payload": self._preview_payload({"arguments": arguments, "error": str(exc)}),
                },
            )
            return ToolResult(
                ok=False,
                tool_name=tool_name,
                error={"code": "tool_runtime_error", "message": str(exc), "details": {}},
            )
