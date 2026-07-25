from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx


@dataclass
class ProviderResult:
    records: List[Dict[str, Any]]
    provider: str = "none"
    requests: int = 0
    status: str = "not_configured"
    error: str = ""


class VentoSocialProvider:
    """Environment-configured provider adapter; no key is required at startup."""

    def __init__(self, *, client: Optional[httpx.AsyncClient] = None) -> None:
        self.provider = os.getenv("VENTO_SOCIAL_PROVIDER", "generic").strip().lower()
        self.api_url = os.getenv("SOCIAL_DATA_API_URL", "").strip()
        self.api_key = os.getenv("SOCIAL_DATA_API_KEY", "").strip()
        self.rapidapi_key = os.getenv("RAPIDAPI_KEY", "").strip()
        self.rapidapi_host = os.getenv("RAPIDAPI_HOST", "").strip()
        self.apify_token = os.getenv("APIFY_API_TOKEN", "").strip()
        self.timeout = max(5.0, min(float(os.getenv("VENTO_PROVIDER_TIMEOUT_SECONDS", "30")), 90.0))
        self.max_pages = max(1, min(int(os.getenv("VENTO_PROVIDER_MAX_PAGES", "5")), 25))
        self._client = client

    @property
    def configured(self) -> bool:
        if not self.api_url:
            return False
        if self.provider == "rapidapi":
            return bool(self.rapidapi_key and self.rapidapi_host)
        if self.provider == "apify":
            return bool(self.apify_token)
        return bool(self.api_key)

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.provider == "rapidapi":
            headers.update({"X-RapidAPI-Key": self.rapidapi_key, "X-RapidAPI-Host": self.rapidapi_host})
        elif self.provider != "apify":
            header = os.getenv("SOCIAL_DATA_AUTH_HEADER", "Authorization").strip() or "Authorization"
            scheme = os.getenv("SOCIAL_DATA_AUTH_SCHEME", "Bearer").strip()
            headers[header] = f"{scheme} {self.api_key}".strip()
        return headers

    def _url(self) -> str:
        if self.provider == "apify" and self.apify_token:
            separator = "&" if "?" in self.api_url else "?"
            return f"{self.api_url}{separator}token={self.apify_token}"
        return self.api_url

    @staticmethod
    def _records(data: Any) -> Tuple[List[Dict[str, Any]], str]:
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)], ""
        if not isinstance(data, dict):
            return [], ""
        for key in ("results", "data", "items", "records"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)], str(data.get("next_cursor") or data.get("cursor") or "")
            if isinstance(value, dict):
                for nested in ("items", "results", "records"):
                    nested_value = value.get(nested)
                    if isinstance(nested_value, list):
                        return [item for item in nested_value if isinstance(item, dict)], str(value.get("next_cursor") or data.get("next_cursor") or "")
        return [], ""

    async def fetch(
        self,
        *,
        location: str,
        influencer_type: str,
        follower_min: int,
        follower_max: int,
        limit: int,
    ) -> ProviderResult:
        if not self.configured:
            return ProviderResult(records=[])
        records: List[Dict[str, Any]] = []
        cursor = ""
        requests = 0
        owned_client = self._client is None
        client = self._client or httpx.AsyncClient(timeout=self.timeout)
        try:
            for page in range(1, self.max_pages + 1):
                payload = {
                    "location": location,
                    "niche": influencer_type,
                    "follower_min": follower_min,
                    "follower_max": follower_max,
                    "limit": min(max(1, limit - len(records)), 100),
                    "page": page,
                }
                if cursor:
                    payload["cursor"] = cursor
                response = await client.post(self._url(), headers=self._headers(), json=payload)
                requests += 1
                if response.status_code == 429:
                    if page < self.max_pages:
                        await asyncio.sleep(min(2 ** (page - 1), 8))
                        continue
                    return ProviderResult(records=records, provider=self.provider, requests=requests, status="rate_limited", error="HTTP 429")
                response.raise_for_status()
                page_records, next_cursor = self._records(response.json())
                records.extend(page_records)
                if len(records) >= limit or not page_records or (not next_cursor and len(page_records) < payload["limit"]):
                    break
                cursor = next_cursor
            return ProviderResult(records=records[:limit], provider=self.provider, requests=requests, status="completed")
        except Exception as exc:
            return ProviderResult(records=records, provider=self.provider, requests=requests, status="failed", error=f"{type(exc).__name__}: {exc}")
        finally:
            if owned_client:
                await client.aclose()
