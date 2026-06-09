import os
import logging
from typing import Any, Dict, Optional

import httpx

from core.tools.base import ToolExecutionError


class TavilyClient:
    BASE_URL = "https://api.tavily.com"
    _research_unavailable = False

    def __init__(self, api_key: Optional[str] = None):
        self._logger = logging.getLogger("agent.runtime")
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ToolExecutionError("TAVILY_API_KEY is not configured", code="missing_api_key")

    async def search(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        request_body = {"api_key": self.api_key, **payload}
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
                "status": f"query={payload.get('query', '')[:120]}",
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
                    },
                )
                return data
        except ToolExecutionError:
            raise
        except httpx.HTTPStatusError as e:
            raise ToolExecutionError(f"Tavily search failed (HTTP {e.response.status_code})", code="tavily_error")
        except Exception as e:
            raise ToolExecutionError(f"Tavily search failed: {e}", code="tavily_error")

    async def research(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.__class__._research_unavailable:
            raise ToolExecutionError(
                "Tavily research endpoint is unavailable for this API key/session",
                code="research_unavailable",
            )
        request_body = {"api_key": self.api_key, **payload}
        self._logger.info(
            "tavily_research_start",
            extra={
                "event": "tavily_research_start",
                "trace_id": "-",
                "session_id": "-",
                "agent_id": "-",
                "workflow_id": "-",
                "step": 0,
                "tool_name": "web_research",
                "status": f"query={payload.get('query', '')[:120]}",
            },
        )
        try:
            async with httpx.AsyncClient(timeout=45) as client:
                response = await client.post(f"{self.BASE_URL}/research", json=request_body)
                if response.status_code == 401:
                    raise ToolExecutionError("TAVILY_API_KEY is invalid or expired", code="invalid_api_key")
                if response.status_code == 422:
                    self.__class__._research_unavailable = True
                    raise ToolExecutionError(
                        "Tavily research endpoint is unavailable for this API key/session",
                        code="research_unavailable",
                    )
                response.raise_for_status()
                data = response.json()
                self._logger.info(
                    "tavily_research_end",
                    extra={
                        "event": "tavily_research_end",
                        "trace_id": "-",
                        "session_id": "-",
                        "agent_id": "-",
                        "workflow_id": "-",
                        "step": 0,
                        "tool_name": "web_research",
                        "status": f"status={response.status_code} results={len(data.get('results', []) or [])}",
                    },
                )
                return data
        except ToolExecutionError:
            raise
        except Exception as e:
            # Fallback: /research may not be available on free plans, return empty
            self._logger.warning(f"Tavily research endpoint failed, caller should fall back to search: {e}")
            raise

