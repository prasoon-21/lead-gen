import os
import logging
from typing import Any, Dict, Optional

import httpx

from core.tools.base import ToolExecutionError


class TavilyClient:
    BASE_URL = "https://api.tavily.com"

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
        async with httpx.AsyncClient(timeout=35) as client:
            response = await client.post(f"{self.BASE_URL}/search", json=request_body)
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

    async def research(self, payload: Dict[str, Any]) -> Dict[str, Any]:
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
        async with httpx.AsyncClient(timeout=45) as client:
            response = await client.post(f"{self.BASE_URL}/research", json=request_body)
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
