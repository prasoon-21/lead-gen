import logging
import os
import time
import uuid
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from fastapi import Request

_LOG_FIELDS = ("request_id", "method", "path", "status_code", "duration_ms")


class RequestLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        for field in _LOG_FIELDS:
            if not hasattr(record, field):
                setattr(record, field, "-")
        return True


def setup_request_logging(log_path: str | None = None) -> logging.Logger:
    logger = logging.getLogger("api.request")
    if logger.handlers:
        return logger

    resolved_path = log_path or os.getenv("REQUEST_LOG_PATH", "logs/requests.log")
    if os.getenv("VERCEL") == "1":
        resolved_path = os.path.join("/tmp", os.path.basename(resolved_path))

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s "
        "request_id=%(request_id)s method=%(method)s path=%(path)s "
        "status=%(status_code)s duration_ms=%(duration_ms)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.addFilter(RequestLogFilter())

    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    try:
        Path(resolved_path).parent.mkdir(parents=True, exist_ok=True)
        file_handler = TimedRotatingFileHandler(resolved_path, when="midnight", backupCount=7)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(RequestLogFilter())
        logger.addHandler(file_handler)
    except Exception as exc:
        print(f"Warning: Could not set up request file logger at {resolved_path} (possibly read-only filesystem): {exc}")

    logger.propagate = False
    return logger


async def request_logger(request: Request, call_next):
    logger = logging.getLogger("api.request")
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    start = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        duration_ms = int((time.perf_counter() - start) * 1000)
        logger.exception(
            "request_failed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": 500,
                "duration_ms": duration_ms,
            },
        )
        raise

    duration_ms = int((time.perf_counter() - start) * 1000)
    logger.info(
        "request",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        },
    )
    response.headers["x-request-id"] = request_id
    return response
