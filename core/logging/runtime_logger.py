import logging
import os
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

_RUNTIME_FIELDS = (
    "event",
    "trace_id",
    "session_id",
    "agent_id",
    "workflow_id",
    "step",
    "tool_name",
    "status",
    "payload",
)


class RuntimeLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        for field in _RUNTIME_FIELDS:
            if not hasattr(record, field):
                setattr(record, field, "-")
        return True


def _build_file_handler(resolved_path: str) -> logging.Handler:
    # Timed rotating handlers frequently fail on Windows during local reloads
    # because the file cannot be renamed while another process still holds it.
    if os.name == "nt":
        return logging.FileHandler(resolved_path, encoding="utf-8")
    return TimedRotatingFileHandler(resolved_path, when="midnight", backupCount=7)


def setup_runtime_logging(log_path: str | None = None) -> logging.Logger:
    logger = logging.getLogger("agent.runtime")
    if logger.handlers:
        return logger

    resolved_path = log_path or os.getenv("AGENT_RUNTIME_LOG_PATH", "logs/agent_runtime.log")
    if os.getenv("VERCEL") == "1":
        resolved_path = os.path.join("/tmp", os.path.basename(resolved_path))

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s "
        "event=%(event)s trace_id=%(trace_id)s session_id=%(session_id)s "
        "agent_id=%(agent_id)s workflow_id=%(workflow_id)s step=%(step)s "
        "tool=%(tool_name)s status=%(status)s payload=%(payload)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.addFilter(RuntimeLogFilter())

    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    try:
        Path(resolved_path).parent.mkdir(parents=True, exist_ok=True)
        file_handler = _build_file_handler(resolved_path)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(RuntimeLogFilter())
        logger.addHandler(file_handler)
    except Exception as exc:
        print(f"Warning: Could not set up runtime file logger at {resolved_path} (possibly read-only filesystem): {exc}")

    logger.propagate = False
    return logger
