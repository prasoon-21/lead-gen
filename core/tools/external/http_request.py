import os
import logging
from typing import Any, Dict
from urllib.parse import urlparse

import httpx

from core.tools.base import BaseTool, ToolContext, ToolExecutionError


class HTTPRequestTool(BaseTool):
    name = "http_request"
    description = "Call an HTTP endpoint and return structured response."
    input_schema = {
        "type": "object",
        "properties": {
            "method": {"type": "string"},
            "url": {"type": "string"},
            "headers": {"type": "object"},
            "params": {"type": "object"},
            "json": {"type": "object"},
            "data": {"type": "object"},
            "timeout_seconds": {"type": "integer"},
        },
        "required": ["method", "url"],
    }
    timeout_seconds = 25
    _allowed_methods = {"GET", "POST", "PUT", "PATCH", "DELETE"}
    _logger = logging.getLogger("agent.runtime")

    def _validate_domain(self, url: str) -> None:
        allowed_domains_raw = os.getenv("ALLOWED_HTTP_TOOL_DOMAINS", "").strip()
        if not allowed_domains_raw:
            return

        allowed_domains = {item.strip().lower() for item in allowed_domains_raw.split(",") if item.strip()}
        host = (urlparse(url).hostname or "").lower()
        if not host:
            raise ToolExecutionError("Invalid URL host", code="invalid_url")

        if host not in allowed_domains:
            raise ToolExecutionError(
                f"Domain '{host}' is not allowed by ALLOWED_HTTP_TOOL_DOMAINS",
                code="domain_not_allowed",
            )

    async def run(self, arguments: Dict[str, Any], context: ToolContext) -> Dict[str, Any]:
        method = str(arguments["method"]).upper()
        if method not in self._allowed_methods:
            raise ToolExecutionError(f"Unsupported HTTP method: {method}", code="invalid_method")

        url = arguments["url"]
        self._validate_domain(url)
        timeout_seconds = int(arguments.get("timeout_seconds") or self.timeout_seconds)
        self._logger.info(
            "http_request_outbound_start",
            extra={
                "event": "http_request_outbound_start",
                "trace_id": context.trace_id,
                "session_id": context.session_id,
                "agent_id": context.state.get("agent_id", "-"),
                "workflow_id": "-",
                "step": 0,
                "tool_name": self.name,
                "status": f"{method} {url}",
            },
        )

        async with httpx.AsyncClient(timeout=timeout_seconds, follow_redirects=True) as client:
            response = await client.request(
                method=method,
                url=url,
                headers=arguments.get("headers"),
                params=arguments.get("params"),
                json=arguments.get("json"),
                data=arguments.get("data"),
            )
        self._logger.info(
            "http_request_outbound_end",
            extra={
                "event": "http_request_outbound_end",
                "trace_id": context.trace_id,
                "session_id": context.session_id,
                "agent_id": context.state.get("agent_id", "-"),
                "workflow_id": "-",
                "step": 0,
                "tool_name": self.name,
                "status": f"status={response.status_code}",
            },
        )

        body_json = None
        body_text = response.text
        try:
            body_json = response.json()
        except Exception:
            body_json = None

        return {
            "status_code": response.status_code,
            "ok": response.is_success,
            "headers": dict(response.headers),
            "json": body_json,
            "text": body_text[:10000],
            "url": str(response.url),
        }
