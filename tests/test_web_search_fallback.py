import unittest
from unittest.mock import AsyncMock, patch

from core.tools.base import ToolContext, ToolExecutionError
from core.tools.external.web_search import WebSearchTool


class TestWebSearchFallback(unittest.IsolatedAsyncioTestCase):
    async def test_enforces_basic_depth_and_caps_result_budget(self):
        tool = WebSearchTool()
        context = ToolContext(session_id="test-session", trace_id="test-trace", state={})
        tavily_payload = {
            "results": [
                {
                    "title": "Example Co",
                    "url": "https://example.com",
                    "content": "Official company site",
                    "score": 0.9,
                    "published_date": None,
                }
            ],
            "answer": None,
            "response_time": 0.1,
        }

        with patch("core.tools.external.web_search.TavilyClient") as mock_tavily_cls, patch(
            "core.tools.external.web_search.os.getenv",
            return_value="fake-key",
        ):
            mock_tavily = mock_tavily_cls.return_value
            mock_tavily.search = AsyncMock(return_value=tavily_payload)

            result = await tool.run(
                {
                    "query": "site:clutch.co fintech delhi",
                    "max_results": 50,
                    "max_results_per_query": 50,
                    "search_depth": "advanced",
                },
                context,
            )

        self.assertEqual(result["search_depth"], "basic")
        self.assertEqual(result["max_results"], 12)
        payload = mock_tavily.search.await_args.args[0]
        self.assertEqual(payload["search_depth"], "basic")
        self.assertEqual(payload["max_results"], 4)

    async def test_uses_html_fallback_when_tavily_quota_is_exceeded(self):
        tool = WebSearchTool()
        context = ToolContext(session_id="test-session", trace_id="test-trace", state={})

        tavily_error = ToolExecutionError(
            "This request exceeds your plan's set usage limit.",
            code="tavily_quota_exceeded",
            details={"status_code": 432, "message": "quota exceeded"},
        )
        fallback_payload = {
            "provider": "duckduckgo_html",
            "results": [
                {
                    "title": "Example Co",
                    "url": "https://example.com",
                    "content": "Official company site",
                    "score": None,
                    "published_date": None,
                }
            ],
            "answer": None,
            "response_time": None,
        }

        with patch("core.tools.external.web_search.TavilyClient") as mock_tavily_cls, patch(
            "core.tools.external.web_search.HTMLSearchFallbackClient.search",
            new=AsyncMock(return_value=fallback_payload),
        ), patch("core.tools.external.web_search.os.getenv", return_value="fake-key"):
            mock_tavily = mock_tavily_cls.return_value
            mock_tavily.search = AsyncMock(side_effect=tavily_error)

            result = await tool.run({"query": "fintech companies in texas", "max_results": 5}, context)

        self.assertEqual(result["count"], 1)
        self.assertTrue(result["fallback_used"])
        self.assertIn("duckduckgo_html", result["providers_used"])
        self.assertTrue(result["warnings"])
        self.assertEqual(result["results"][0]["url"], "https://example.com")

    async def test_uses_html_fallback_when_tavily_is_not_configured(self):
        tool = WebSearchTool()
        context = ToolContext(session_id="test-session", trace_id="test-trace", state={})
        fallback_payload = {
            "provider": "duckduckgo_html",
            "results": [
                {
                    "title": "Fallback Result",
                    "url": "https://fallback.example",
                    "content": "Fallback search result",
                    "score": None,
                    "published_date": None,
                }
            ],
            "answer": None,
            "response_time": None,
        }

        with patch(
            "core.tools.external.web_search.HTMLSearchFallbackClient.search",
            new=AsyncMock(return_value=fallback_payload),
        ), patch("core.tools.external.web_search.os.getenv", return_value=""):
            result = await tool.run({"query": "solar installers austin", "max_results": 5}, context)

        self.assertEqual(result["count"], 1)
        self.assertTrue(result["fallback_used"])
        self.assertTrue(result["providers_used"])
        self.assertTrue(all(provider == "duckduckgo_html" for provider in result["providers_used"]))
