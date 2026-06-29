import os
import secrets
from typing import Optional

from fastapi import Request
from fastapi.responses import JSONResponse


DEFAULT_API_KEY_HEADER = "x-api-key"


def _clean(value: Optional[str]) -> str:
    return (value or "").strip()


def _bearer_token(authorization: Optional[str]) -> str:
    value = _clean(authorization)
    if not value.lower().startswith("bearer "):
        return ""
    return value[7:].strip()


def _host_name(host: str) -> str:
    value = _clean(host).lower()
    if value.startswith("[") and "]" in value:
        return value[1 : value.index("]")]
    return value.split(":", 1)[0]


def _is_local_host(host: str) -> bool:
    value = _host_name(host)
    return value in {"localhost", "127.0.0.1", "::1"} or value.endswith(".localhost")


def is_production_mode(value: Optional[str] = None) -> bool:
    return _clean(value or os.getenv("APP_ENV") or os.getenv("ENVIRONMENT")).lower() in {"prod", "production"}


def is_valid_api_key(
    configured_key: Optional[str],
    header_value: Optional[str],
    authorization: Optional[str],
    *,
    require_key: bool = False,
) -> bool:
    """Validate an inbound API key.

    If no AGENTIC_CORE_API_KEY is configured, auth stays disabled for local/dev
    compatibility. When configured, callers may use X-API-Key or Bearer auth.
    """
    expected = _clean(configured_key)
    if not expected:
        return not require_key

    expected_bytes = expected.encode("utf-8")
    candidates = [_clean(header_value), _bearer_token(authorization)]
    return any(
        candidate and secrets.compare_digest(candidate.encode("utf-8"), expected_bytes)
        for candidate in candidates
    )


async def api_key_middleware(request: Request, call_next):
    if request.method == "OPTIONS":
        return await call_next(request)

    path = request.url.path
    if not path.startswith("/api/"):
        return await call_next(request)

    # Exempt internal UI calls (same-origin check)
    referer = request.headers.get("referer", "")
    origin = request.headers.get("origin", "")
    host = request.headers.get("host", "")
    sec_fetch_site = request.headers.get("sec-fetch-site", "").lower()
    if host and (
        (referer and host in referer)
        or (origin and host in origin)
        or sec_fetch_site in {"same-origin", "same-site"}
    ):
        return await call_next(request)

    # The Velit scheduler is a local browser control panel. Allow it on
    # localhost so opening the HTML file directly can still reach the API.
    if path.startswith("/api/velit-batch") and _is_local_host(host):
        return await call_next(request)

    header_name = os.getenv("AGENTIC_CORE_API_KEY_HEADER", DEFAULT_API_KEY_HEADER)
    configured_key = os.getenv("AGENTIC_CORE_API_KEY")
    require_key = is_production_mode()
    header_value = request.headers.get(header_name)
    authorization = request.headers.get("authorization")

    if not is_valid_api_key(configured_key, header_value, authorization, require_key=require_key):
        return JSONResponse(status_code=401, content={"detail": "Invalid or missing API key"})

    return await call_next(request)
