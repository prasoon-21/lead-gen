import os
import logging
import json
from typing import Any, Dict, Optional

import httpx

from core.services.lead_discovery_policy import (
    DEFAULT_TAVILY_EXTRACT_DEPTH,
    DEFAULT_TAVILY_SEARCH_DEPTH,
    clamp_web_search_max_results,
)
from core.tools.base import ToolExecutionError


class TavilyClient:
    BASE_URL = "https://api.tavily.com"
    _research_unavailable = False

    def __init__(self, api_key: Optional[str] = None):
        self._logger = logging.getLogger("agent.runtime")
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ToolExecutionError("TAVILY_API_KEY is not configured", code="missing_api_key")

    @staticmethod
    def _preview_payload(value: Dict[str, Any], limit: int = 1000) -> str:
        try:
            text = json.dumps(value, ensure_ascii=False)
        except Exception:
            text = str(value)
        if len(text) > limit:
            return text[:limit] + "...[truncated]"
        return text

    @staticmethod
    def _response_error_detail(response: httpx.Response) -> Dict[str, Any]:
        detail: Dict[str, Any] = {"status_code": response.status_code}
        try:
            parsed = response.json()
            detail["response_json"] = parsed
            if isinstance(parsed, dict):
                nested_detail = parsed.get("detail")
                if isinstance(nested_detail, dict):
                    detail["message"] = nested_detail.get("error") or nested_detail.get("message")
                elif nested_detail:
                    detail["message"] = str(nested_detail)
        except Exception:
            pass

        response_text = (response.text or "").strip()
        if response_text:
            detail["response_text"] = response_text[:600]
            detail.setdefault("message", response_text[:240])
        return detail

    @staticmethod
    def _sanitize_search_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        sanitized = dict(payload or {})
        sanitized["search_depth"] = DEFAULT_TAVILY_SEARCH_DEPTH
        sanitized["max_results"] = clamp_web_search_max_results(sanitized.get("max_results"))
        return sanitized

    @staticmethod
    def _sanitize_extract_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        urls = payload.get("urls") or []
        normalized_urls = []
        for url in urls:
            cleaned = str(url or "").strip()
            if cleaned and cleaned not in normalized_urls:
                normalized_urls.append(cleaned)
        return {
            "urls": normalized_urls,
            "extract_depth": DEFAULT_TAVILY_EXTRACT_DEPTH,
            "include_images": bool(payload.get("include_images", False)),
        }

    async def search(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        sanitized_payload = self._sanitize_search_payload(payload)
        request_body = {"api_key": self.api_key, **sanitized_payload}
        self._logger.info(
            "tavily_search_start",
            extra={
                "event": "tavily_search_start",
                "trace_id": "-",
                "session_id": "-",
                "agent_id": "-",
                "workflow_id": "-",
                "step": 0,
                "tool_name": "web_search",
                "status": f"query={sanitized_payload.get('query', '')[:120]}",
                "payload": self._preview_payload(sanitized_payload),
            },
        )
        try:
            async with httpx.AsyncClient(timeout=35) as client:
                response = await client.post(f"{self.BASE_URL}/search", json=request_body)
                if response.status_code == 401:
                    raise ToolExecutionError("TAVILY_API_KEY is invalid or expired. Get a new key at https://tavily.com", code="invalid_api_key")
                response.raise_for_status()
                data = response.json()
                self._logger.info(
                    "tavily_search_end",
                    extra={
                        "event": "tavily_search_end",
                        "trace_id": "-",
                        "session_id": "-",
                        "agent_id": "-",
                        "workflow_id": "-",
                        "step": 0,
                        "tool_name": "web_search",
                        "status": f"status={response.status_code} results={len(data.get('results', []) or [])}",
                        "payload": self._preview_payload(
                            {
                                "status_code": response.status_code,
                                "result_count": len(data.get("results", []) or []),
                                "provider": "tavily",
                            }
                        ),
                    },
                )
                return data
        except ToolExecutionError:
            raise
        except httpx.HTTPStatusError as e:
            detail = self._response_error_detail(e.response)
            self._logger.warning(
                "tavily_search_http_error",
                extra={
                    "event": "tavily_search_http_error",
                    "trace_id": "-",
                    "session_id": "-",
                    "agent_id": "-",
                    "workflow_id": "-",
                    "step": 0,
                    "tool_name": "web_search",
                    "status": f"http={e.response.status_code}",
                    "payload": self._preview_payload(detail),
                },
            )
            message = detail.get("message") or f"Tavily search failed (HTTP {e.response.status_code})"
            error_code = "tavily_quota_exceeded" if e.response.status_code == 432 else "tavily_error"
            raise ToolExecutionError(message, code=error_code, details=detail)
        except Exception as e:
            self._logger.exception(
                "tavily_search_error",
                extra={
                    "event": "tavily_search_error",
                    "trace_id": "-",
                    "session_id": "-",
                    "agent_id": "-",
                    "workflow_id": "-",
                    "step": 0,
                    "tool_name": "web_search",
                    "status": type(e).__name__,
                    "payload": self._preview_payload({"error": str(e), "query": sanitized_payload.get("query", "")}),
                },
            )
            raise ToolExecutionError(f"Tavily search failed: {e}", code="tavily_error")

    async def research(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        raise ToolExecutionError(
            "Tavily research is disabled for the cost-efficient lead funnel. Use basic web_search instead.",
            code="research_disabled",
        )

    async def extract(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        sanitized_payload = self._sanitize_extract_payload(payload)
        if not sanitized_payload["urls"]:
            raise ToolExecutionError("urls is required for Tavily extract", code="invalid_arguments")

        request_body = {"api_key": self.api_key, **sanitized_payload}
        self._logger.info(
            "tavily_extract_start",
            extra={
                "event": "tavily_extract_start",
                "trace_id": "-",
                "session_id": "-",
                "agent_id": "-",
                "workflow_id": "-",
                "step": 0,
                "tool_name": "web_extract",
                "status": f"urls={len(sanitized_payload['urls'])}",
                "payload": self._preview_payload(sanitized_payload),
            },
        )
        try:
            async with httpx.AsyncClient(timeout=40) as client:
                response = await client.post(f"{self.BASE_URL}/extract", json=request_body)
                if response.status_code == 401:
                    raise ToolExecutionError("TAVILY_API_KEY is invalid or expired", code="invalid_api_key")
                response.raise_for_status()
                data = response.json()
                self._logger.info(
                    "tavily_extract_end",
                    extra={
                        "event": "tavily_extract_end",
                        "trace_id": "-",
                        "session_id": "-",
                        "agent_id": "-",
                        "workflow_id": "-",
                        "step": 0,
                        "tool_name": "web_extract",
                        "status": f"status={response.status_code}",
                        "payload": self._preview_payload(
                            {
                                "status_code": response.status_code,
                                "url_count": len(sanitized_payload["urls"]),
                                "provider": "tavily",
                            }
                        ),
                    },
                )
                return data
        except ToolExecutionError:
            raise
        except httpx.HTTPStatusError as exc:
            detail = self._response_error_detail(exc.response)
            self._logger.warning(
                "tavily_extract_http_error",
                extra={
                    "event": "tavily_extract_http_error",
                    "trace_id": "-",
                    "session_id": "-",
                    "agent_id": "-",
                    "workflow_id": "-",
                    "step": 0,
                    "tool_name": "web_extract",
                    "status": f"http={exc.response.status_code}",
                    "payload": self._preview_payload(detail),
                },
            )
            message = detail.get("message") or f"Tavily extract failed (HTTP {exc.response.status_code})"
            error_code = "tavily_quota_exceeded" if exc.response.status_code == 432 else "tavily_error"
            raise ToolExecutionError(message, code=error_code, details=detail)
        except Exception as exc:
            self._logger.exception(
                "tavily_extract_error",
                extra={
                    "event": "tavily_extract_error",
                    "trace_id": "-",
                    "session_id": "-",
                    "agent_id": "-",
                    "workflow_id": "-",
                    "step": 0,
                    "tool_name": "web_extract",
                    "status": type(exc).__name__,
                    "payload": self._preview_payload({"error": str(exc), "urls": sanitized_payload["urls"]}),
                },
            )
            raise ToolExecutionError(f"Tavily extract failed: {exc}", code="tavily_error")

