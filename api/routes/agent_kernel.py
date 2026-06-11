"""
Agent Kernel — conversational agent endpoint with memory, tool calling,
and optional file/audio attachments.

Endpoints
---------
GET  /api/agent/list                 — list all available agent IDs
POST /api/agent/run                  — run an agent (JSON or multipart)

The /run endpoint auto-detects content type:

  JSON body (existing / simple):
    Content-Type: application/json
    { "agent_id", "message", "session_id"?, "context"?, "documents"? }

  Multipart form (with file attachments):
    Content-Type: multipart/form-data
    Fields : agent_id, message, session_id?, context? (JSON string)
    Files  : files  (one or more — any name is fine)

Supported attachment types
--------------------------
  Images  jpg/png/gif/webp  → described via Gemini Vision → injected into message
  Audio   mp3/wav/ogg/m4a/webm/flac → transcribed via Gemini → injected
  PDF     .pdf              → text extracted via pypdf → injected
  Word    .docx             → text extracted via python-docx → injected
  Excel   .xlsx/.xlsm       → text extracted from workbook XML → injected
  Slides  .pptx             → text extracted from slide XML → injected
  Text    .txt/.csv/.md     → decoded and injected directly
"""

import base64
import binascii
import json
import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from starlette.datastructures import UploadFile

from api.state import get_state
from core.media.processor import MediaProcessor

router = APIRouter()
runtime_logger = logging.getLogger("agent.runtime")

DEFAULT_MAX_FILE_BYTES = 30 * 1024 * 1024
DEFAULT_MAX_TOTAL_BYTES = 60 * 1024 * 1024


# ── Response schema ───────────────────────────────────────────────────────────

class AgentRunResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    success: bool = True
    session_id: str
    trace_id: str
    response_mode: str
    text: Optional[str] = None
    json_: Optional[Dict[str, Any]] = Field(default=None, alias="json", serialization_alias="json")
    steps: list = []
    attachments_processed: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}


# ── List endpoint ─────────────────────────────────────────────────────────────

@router.get("/list")
async def list_agents():
    state = get_state()
    if not state.config_loader:
        raise HTTPException(status_code=500, detail="Config loader not initialized")
    return {"agents": state.config_loader.list_agent_specs()}


# ── Run endpoint (JSON + multipart unified) ───────────────────────────────────

@router.post("/run", response_model=AgentRunResponse)
async def run_agent(request: Request):
    """
    Run a conversational agent. Accepts either:
      - application/json          → simple JSON body
      - multipart/form-data       → form fields + optional file attachments
    """
    state = get_state()
    if not state.agent_factory:
        raise HTTPException(status_code=500, detail="Agent factory not initialized")

    content_type = request.headers.get("content-type", "")

    # ── Parse request ─────────────────────────────────────────────────────────
    if "multipart/form-data" in content_type:
        agent_id, message, session_id, context, raw_files = await _parse_multipart(request)
    else:
        agent_id, message, session_id, context, raw_files = await _parse_json(request)

    if not agent_id:
        raise HTTPException(status_code=422, detail="agent_id is required")
    if not message and not raw_files:
        raise HTTPException(status_code=422, detail="message or at least one file is required")

    message = message or ""
    _validate_upload_sizes(raw_files, *_upload_limits())

    # ── Process attachments ───────────────────────────────────────────────────
    attachments_meta: List[Dict[str, Any]] = []
    if raw_files:
        processor = MediaProcessor(adapter=state.adapter)
        processed = await processor.process(raw_files)
        message = processed.build_message(message)
        attachments_meta = processed.to_metadata()

    # ── Log ───────────────────────────────────────────────────────────────────
    caller = (context or {}).get("user_id") or (context or {}).get("user") or "unknown"
    _safe_context = {k: v for k, v in (context or {}).items() if k not in {"password", "token", "secret"}}
    runtime_logger.info(
        "agent_api_call",
        extra={
            "event": "agent_api_call",
            "trace_id": "-",
            "session_id": session_id or "-",
            "agent_id": agent_id,
            "workflow_id": "-",
            "step": 0,
            "tool_name": "-",
            "status": f"caller={caller} attachments={len(attachments_meta)}",
            "payload": json.dumps({
                "agent_id": agent_id,
                "session_id": session_id or "-",
                "message_preview": (message or "")[:120],
                "context": _safe_context,
                "attachments": len(attachments_meta),
            }, ensure_ascii=False),
        },
    )

    # ── Execute ───────────────────────────────────────────────────────────────
    try:
        result = await state.agent_factory.run(
            agent_id=agent_id,
            message=message,
            session_id=session_id,
            context=context or {},
        )
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail=f"Unknown agent_id: {agent_id}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return AgentRunResponse(
        success=True,
        session_id=result.get("session_id", ""),
        trace_id=result.get("trace_id", ""),
        response_mode=result.get("response_mode", "text"),
        text=result.get("text"),
        json_=result.get("json"),
        steps=result.get("steps", []),
        attachments_processed=attachments_meta,
        metadata=result.get("metadata", {}),
    )


# ── Parsers ───────────────────────────────────────────────────────────────────

async def _parse_json(request: Request):
    """Parse a plain JSON body."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid JSON body")

    agent_id   = body.get("agent_id", "")
    message    = body.get("message", "")
    session_id = body.get("session_id") or None
    context    = body.get("context") or {}
    raw_files = _files_from_document_payloads(body.get("documents") or [])

    legacy_base64 = body.get("fileBase64") or body.get("file_base64")
    if legacy_base64:
        raw_files.extend(
            _files_from_document_payloads(
                [
                    {
                        "base64": legacy_base64,
                        "filename": body.get("fileName") or body.get("file_name") or "upload",
                        "mimeType": body.get("fileMimeType") or body.get("file_mime_type"),
                    }
                ]
            )
        )

    return agent_id, message, session_id, context, raw_files


async def _parse_multipart(request: Request):
    """
    Parse multipart/form-data.

    Text fields: agent_id, message, session_id, context (JSON string)
    File fields: any field named 'files' (multiple allowed) OR any uploaded file field
    """
    try:
        form = await request.form()
    except Exception:
        raise HTTPException(status_code=422, detail="Could not parse multipart form")

    agent_id   = form.get("agent_id", "")
    message    = form.get("message", "")
    session_id = form.get("session_id") or None

    context_raw = form.get("context", "")
    context: Dict[str, Any] = {}
    if context_raw:
        try:
            context = json.loads(context_raw)
        except json.JSONDecodeError:
            raise HTTPException(status_code=422, detail="'context' must be a valid JSON string")

    # Collect all uploaded files from all field names
    raw_files = []
    for key, value in form.multi_items():
        if isinstance(value, UploadFile):
            file_bytes = await value.read()
            if file_bytes:
                raw_files.append((
                    value.filename or key,
                    file_bytes,
                    value.content_type or None,
                ))

    return agent_id, message, session_id, context, raw_files


def _files_from_document_payloads(documents: Any) -> List[tuple[str, bytes, Optional[str]]]:
    if not documents:
        return []
    if not isinstance(documents, list):
        raise HTTPException(status_code=422, detail="'documents' must be a list")

    raw_files: List[tuple[str, bytes, Optional[str]]] = []
    for index, document in enumerate(documents, start=1):
        if not isinstance(document, dict):
            raise HTTPException(status_code=422, detail=f"documents[{index}] must be an object")

        encoded = document.get("base64") or document.get("fileBase64")
        if not encoded or not isinstance(encoded, str):
            raise HTTPException(status_code=422, detail=f"documents[{index}].base64 is required")

        if "," in encoded and encoded.split(",", 1)[0].lower().startswith("data:"):
            encoded = encoded.split(",", 1)[1]

        try:
            file_bytes = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError):
            raise HTTPException(status_code=422, detail=f"documents[{index}].base64 must be valid base64")

        extension = (document.get("extension") or "").strip().lstrip(".")
        filename = (
            document.get("filename")
            or document.get("fileName")
            or document.get("name")
            or (f"upload.{extension}" if extension else f"upload-{index}")
        )
        mime_type = document.get("mimeType") or document.get("mime_type") or document.get("contentType")
        raw_files.append((str(filename), file_bytes, str(mime_type) if mime_type else None))

    return raw_files


def _upload_limits() -> tuple[int, int]:
    return (
        _env_megabytes("AGENT_UPLOAD_MAX_FILE_MB", DEFAULT_MAX_FILE_BYTES),
        _env_megabytes("AGENT_UPLOAD_MAX_TOTAL_MB", DEFAULT_MAX_TOTAL_BYTES),
    )


def _env_megabytes(name: str, default_bytes: int) -> int:
    raw_value = os.getenv(name)
    if not raw_value:
        return default_bytes
    try:
        value = int(raw_value)
    except ValueError:
        return default_bytes
    return value * 1024 * 1024 if value > 0 else default_bytes


def _validate_upload_sizes(
    raw_files: List[tuple[str, bytes, Optional[str]]],
    max_file_bytes: int,
    max_total_bytes: int,
) -> None:
    total_bytes = 0
    for filename, file_bytes, _mime_type in raw_files:
        size = len(file_bytes)
        total_bytes += size
        if size > max_file_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File '{filename}' exceeds the {max_file_bytes // (1024 * 1024)} MB upload limit",
            )
    if total_bytes > max_total_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"combined upload size exceeds the {max_total_bytes // (1024 * 1024)} MB upload limit",
        )


