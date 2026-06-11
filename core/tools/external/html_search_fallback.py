import asyncio
import html
import re
from typing import Any, Dict, List
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from core.tools.base import ToolExecutionError


class HTMLSearchFallbackClient:
    SEARCH_URL = "https://html.duckduckgo.com/html/"
    PROVIDER_NAME = "duckduckgo_html"
    _result_link_pattern = re.compile(
        r'<a[^>]+class="result__a"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
        re.IGNORECASE | re.DOTALL,
    )
    _snippet_pattern = re.compile(
        r'<a[^>]+class="result__snippet"[^>]*>(?P<snippet>.*?)</a>|'
        r'<div[^>]+class="result__snippet"[^>]*>(?P<div_snippet>.*?)</div>',
        re.IGNORECASE | re.DOTALL,
    )

    @staticmethod
    def _strip_tags(value: str) -> str:
        text = re.sub(r"(?is)<[^>]+>", " ", value or "")
        return " ".join(html.unescape(text).split()).strip()

    @classmethod
    def _unwrap_duckduckgo_url(cls, href: str) -> str:
        parsed = urlparse(href or "")
        if parsed.netloc.endswith("duckduckgo.com") or not parsed.netloc:
            query = parse_qs(parsed.query)
            uddg = query.get("uddg")
            if uddg:
                return unquote(uddg[0])
        return href

    @classmethod
    def _parse_results(cls, html_text: str, max_results: int) -> List[Dict[str, Any]]:
        snippets = list(cls._snippet_pattern.finditer(html_text or ""))
        results: List[Dict[str, Any]] = []

        for index, match in enumerate(cls._result_link_pattern.finditer(html_text or "")):
            href = cls._unwrap_duckduckgo_url(match.group("href"))
            title = cls._strip_tags(match.group("title"))
            snippet = ""
            if index < len(snippets):
                snippet_match = snippets[index]
                snippet = cls._strip_tags(
                    snippet_match.group("snippet") or snippet_match.group("div_snippet") or ""
                )

            if not href or not title:
                continue

            results.append(
                {
                    "title": title,
                    "url": href,
                    "content": snippet,
                    "score": None,
                    "published_date": None,
                }
            )
            if len(results) >= max_results:
                break

        return results

    async def search(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        query = str(payload.get("query") or "").strip()
        if not query:
            raise ToolExecutionError("query is required for fallback search", code="invalid_arguments")

        max_results = int(payload.get("max_results") or 10)
        headers = {"User-Agent": "Mozilla/5.0"}

        try:
            response = await asyncio.to_thread(
                httpx.post,
                self.SEARCH_URL,
                data={"q": query},
                timeout=20,
                headers=headers,
                follow_redirects=True,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ToolExecutionError(
                f"Fallback search failed (HTTP {exc.response.status_code})",
                code="fallback_search_error",
                details={
                    "provider": self.PROVIDER_NAME,
                    "status_code": exc.response.status_code,
                    "response_text": (exc.response.text or "")[:600],
                },
            )
        except Exception as exc:
            raise ToolExecutionError(
                f"Fallback search failed: {exc}",
                code="fallback_search_error",
                details={"provider": self.PROVIDER_NAME},
            )

        results = self._parse_results(response.text, max_results=max_results)
        if not results:
            raise ToolExecutionError(
                "Fallback search returned no parsable results",
                code="fallback_search_empty",
                details={"provider": self.PROVIDER_NAME},
            )

        return {
            "provider": self.PROVIDER_NAME,
            "query": query,
            "results": results,
            "answer": None,
            "response_time": None,
        }
