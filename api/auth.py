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


def is_valid_api_key(
    configured_key: Optional[str],
    header_value: Optional[str],
    authorization: Optional[str],
) -> bool:
    """Validate an inbound API key.

    If no AGENTIC_CORE_API_KEY is configured, auth stays disabled for local/dev
    compatibility. When configured, callers may use X-API-Key or Bearer auth.
    """
    expected = _clean(configured_key)
    if not expected:
        return True

    expected_bytes = expected.encode("utf-8")
    candidates = [_clean(header_value), _bearer_token(authorization)]
    return any(
        candidate and secrets.compare_digest(candidate.encode("utf-8"), expected_bytes)
        for candidate in candidates
    )


async def api_key_middleware(request: Request, call_next):
    if request.method == "OPTIONS":
        return await call_next(request)

    if not request.url.path.startswith("/api/"):
        return await call_next(request)

    # Exempt internal UI calls (same-origin check)
    referer = request.headers.get("referer", "")
    host = request.headers.get("host", "")
    if referer and host and host in referer:
        return await call_next(request)

    header_name = os.getenv("AGENTIC_CORE_API_KEY_HEADER", DEFAULT_API_KEY_HEADER)
    configured_key = os.getenv("AGENTIC_CORE_API_KEY")
    header_value = request.headers.get(header_name)
    authorization = request.headers.get("authorization")

    if not is_valid_api_key(configured_key, header_value, authorization):
        return JSONResponse(status_code=401, content={"detail": "Invalid or missing API key"})

    return await call_next(request)
