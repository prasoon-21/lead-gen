import os
import json
import logging
import httpx
from typing import Any, Dict, List, Optional

from core.tools.base import BaseTool, ToolContext, ToolExecutionError


class RemoteToolWrapper(BaseTool):
    """Wraps an external HTTP endpoint as a standard BaseTool.

    By default, the external project exposes a POST endpoint that receives:
        {
            "arguments": { ...tool args... },
            "session_id": "abc123",
            "trace_id": "WTRACE_xyz"
        }

    And returns:
        { "result": { ...any dict... } }

    This allows tools that live in a different project (e.g. Aurika OS)
    to be called by any agent in Agentic Core just like a local tool.
    """

    def __init__(
        self,
        name: str,
        description: str,
        endpoint_url: str,
        input_schema: Optional[Dict[str, Any]] = None,
        auth_header_name: Optional[str] = None,
        auth_header_env: Optional[str] = None,
        auth_header_prefix: str = "",
        timeout_seconds: int = 30,
        method: str = "POST",
        request_body_mode: str = "wrapped",
        static_headers: Optional[Dict[str, str]] = None,
        argument_headers: Optional[Dict[str, str]] = None,
        omit_arguments_from_body: Optional[List[str]] = None,
    ):
        self.name = name
        self.description = description
        self.endpoint_url = endpoint_url
        self.input_schema = input_schema or {"type": "object", "properties": {}}
        self._auth_header_name = auth_header_name
        self._auth_header_env = auth_header_env
        self._auth_header_prefix = auth_header_prefix or ""
        self.timeout_seconds = timeout_seconds
        self.method = (method or "POST").upper()
        self.request_body_mode = request_body_mode or "wrapped"
        self.static_headers = static_headers or {}
        self.argument_headers = argument_headers or {}
        self.omit_arguments_from_body = set(omit_arguments_from_body or [])
        self._logger = logging.getLogger("agent.runtime")

    def _redact_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        redacted = {}
        for key, value in headers.items():
            lowered = key.lower()
            if "authorization" in lowered or "api-key" in lowered or "token" in lowered:
                redacted[key] = "[REDACTED]"
            else:
                redacted[key] = value
        return redacted

    def _payload_preview(self, payload: Dict[str, Any]) -> str:
        try:
            preview = json.dumps(payload, ensure_ascii=False)
        except Exception:
            preview = str(payload)
        if len(preview) > 600:
            return preview[:600] + "...[truncated]"
        return preview

    async def run(self, arguments: Dict[str, Any], context: ToolContext) -> Dict[str, Any]:
        headers = {"Content-Type": "application/json", **self.static_headers}

        if self._auth_header_name and self._auth_header_env:
            token = os.getenv(self._auth_header_env)
            if token:
                headers[self._auth_header_name] = f"{self._auth_header_prefix}{token}"

        for header_name, argument_name in self.argument_headers.items():
            value = arguments.get(argument_name)
            if value is not None:
                headers[header_name] = str(value)

        body_arguments = {
            key: value
            for key, value in arguments.items()
            if key not in self.omit_arguments_from_body
        }

        if self.request_body_mode == "raw_arguments":
            payload = body_arguments
        else:
            payload = {
                "arguments": body_arguments,
                "session_id": context.session_id,
                "trace_id": context.trace_id,
            }

        self._logger.info(
            (
                "remote_tool_request "
                f"url={self.endpoint_url} method={self.method} body_mode={self.request_body_mode} "
                f"headers={self._redact_headers(headers)} payload={self._payload_preview(payload)}"
            ),
            extra={
                "event": "remote_tool_request",
                "trace_id": context.trace_id,
                "session_id": context.session_id,
                "agent_id": context.state.get("agent_id", "-"),
                "workflow_id": "-",
                "step": 0,
                "tool_name": self.name,
                "status": "request_sent",
            },
        )

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                if self.method == "GET":
                    response = await client.request(
                        self.method,
                        self.endpoint_url,
                        params=body_arguments if body_arguments else None,
                        headers=headers,
                    )
                else:
                    response = await client.request(
                        self.method,
                        self.endpoint_url,
                        json=payload,
                        headers=headers,
                    )
                response.raise_for_status()
                data = response.json()
                result_payload = data.get("result", data) if isinstance(data, dict) else {"result": data}
                self._logger.info(
                    (
                        "remote_tool_response "
                        f"url={self.endpoint_url} status_code={response.status_code} "
                        f"response={self._payload_preview(result_payload if isinstance(result_payload, dict) else {'result': result_payload})}"
                    ),
                    extra={
                        "event": "remote_tool_response",
                        "trace_id": context.trace_id,
                        "session_id": context.session_id,
                        "agent_id": context.state.get("agent_id", "-"),
                        "workflow_id": "-",
                        "step": 0,
                        "tool_name": self.name,
                        "status": f"http={response.status_code}",
                    },
                )
                # Accept either { "result": {...} } or a raw dict
                return result_payload

        except httpx.TimeoutException:
            self._logger.info(
                f"remote_tool_timeout url={self.endpoint_url} timeout_seconds={self.timeout_seconds}",
                extra={
                    "event": "remote_tool_timeout",
                    "trace_id": context.trace_id,
                    "session_id": context.session_id,
                    "agent_id": context.state.get("agent_id", "-"),
                    "workflow_id": "-",
                    "step": 0,
                    "tool_name": self.name,
                    "status": f"timeout={self.timeout_seconds}",
                },
            )
            raise ToolExecutionError(
                f"Remote tool '{self.name}' timed out after {self.timeout_seconds}s",
                code="remote_tool_timeout",
                details={"endpoint_url": self.endpoint_url},
            )
        except httpx.HTTPStatusError as exc:
            response_text = exc.response.text[:600] if exc.response is not None and exc.response.text else ""
            self._logger.info(
                (
                    "remote_tool_http_error "
                    f"url={self.endpoint_url} status_code={exc.response.status_code} "
                    f"response_body={response_text}"
                ),
                extra={
                    "event": "remote_tool_http_error",
                    "trace_id": context.trace_id,
                    "session_id": context.session_id,
                    "agent_id": context.state.get("agent_id", "-"),
                    "workflow_id": "-",
                    "step": 0,
                    "tool_name": self.name,
                    "status": f"http={exc.response.status_code}",
                },
            )
            raise ToolExecutionError(
                f"Remote tool '{self.name}' returned HTTP {exc.response.status_code}",
                code="remote_tool_http_error",
                details={
                    "status_code": exc.response.status_code,
                    "endpoint_url": self.endpoint_url,
                },
            )
        except Exception as exc:
            self._logger.info(
                f"remote_tool_error url={self.endpoint_url} error={exc}",
                extra={
                    "event": "remote_tool_error",
                    "trace_id": context.trace_id,
                    "session_id": context.session_id,
                    "agent_id": context.state.get("agent_id", "-"),
                    "workflow_id": "-",
                    "step": 0,
                    "tool_name": self.name,
                    "status": "exception",
                },
            )
            raise ToolExecutionError(
                f"Remote tool '{self.name}' failed: {exc}",
                code="remote_tool_error",
                details={"endpoint_url": self.endpoint_url},
            )
