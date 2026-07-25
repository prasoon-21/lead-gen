"""Minimal API contract for the Vento Plan 3 lead finder."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field, field_validator

from api.state import get_state
from core.services.lead_service import LeadService
from core.services.vento.vento_lead_pipeline import VentoLeadPipeline

router = APIRouter()
VentoCategory = Literal["dog_parent_influencers", "online_pet_businesses", "pet_shops", "pet_grooming"]
_ALLOWED_UPLOAD_SUFFIXES = {".csv", ".xlsx"}


class VentoSearchRequest(BaseModel):
    category: VentoCategory
    location: str = Field(min_length=1, max_length=160)
    target_count: int = Field(default=50, ge=1, le=200)
    old_leads_csv: Optional[str] = Field(default=None, max_length=15_000_000)

    @field_validator("location")
    @classmethod
    def clean_location(cls, value: str) -> str:
        cleaned = " ".join(value.split()).strip()
        if not cleaned:
            raise ValueError("location is required")
        return cleaned


def _sheets_credentials_configured() -> bool:
    path = os.getenv("GOOGLE_CREDENTIALS_PATH", "").strip()
    return bool(
        (path and Path(path).expanduser().is_file())
        or os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
        or os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON_BASE64", "").strip()
    )


async def _save_to_sheets(result: Dict[str, Any]) -> None:
    store = get_state().todo_store
    if store is None or not _sheets_credentials_configured():
        result["storage_status"] = "not_configured"
        return
    try:
        report = await LeadService(store).export_leads(result.get("leads") or [], mode="vento")
        result["storage_status"] = "saved"
        result["storage_report"] = report
    except Exception as exc:  # Search results remain useful if Sheets is temporarily unavailable.
        result["storage_status"] = "failed"
        result["storage_error"] = str(exc)[:300]


@router.post("/search")
async def search(request: VentoSearchRequest) -> Dict[str, Any]:
    """Run Plan 3 discovery and enrichment, optionally deduping CSV history."""
    try:
        result = await VentoLeadPipeline().run(
            category=request.category,
            location=request.location,
            target_count=request.target_count,
            old_leads_csv=request.old_leads_csv,
        )
        await _save_to_sheets(result)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/ingest")
async def ingest_csv(
    file: UploadFile = File(...),
    category: VentoCategory = Form(...),
    location: str = Form(..., min_length=1, max_length=160),
    target_count: int = Form(default=50, ge=1, le=200),
) -> Dict[str, Any]:
    """Use an old CSV/XLSX as dedup history, then run the same Plan 3 search."""
    filename = Path(file.filename or "old-leads.csv").name
    suffix = Path(filename).suffix.lower()
    if suffix not in _ALLOWED_UPLOAD_SUFFIXES:
        raise HTTPException(status_code=400, detail="Only .csv and .xlsx files are supported")
    maximum = max(1, min(50, int(os.getenv("VENTO_MAX_UPLOAD_MB", "15")))) * 1024 * 1024
    temporary_path = ""
    size = 0
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary:
            temporary_path = temporary.name
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > maximum:
                    raise HTTPException(status_code=413, detail=f"Upload exceeds {maximum // (1024 * 1024)} MB")
                temporary.write(chunk)
        result = await VentoLeadPipeline().run(
            category=category,
            location=" ".join(location.split()).strip(),
            target_count=target_count,
            old_leads_file=temporary_path,
        )
        result["imported_filename"] = filename
        await _save_to_sheets(result)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        await file.close()
        if temporary_path:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass
