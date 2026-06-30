from __future__ import annotations

import io
import re
import zipfile
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse, StreamingResponse

from api.state import get_state
from core.services.velit.batch_scheduler import VelitBatchScheduler


router = APIRouter()


class AutomationRequest(BaseModel):
    name: Optional[str] = None
    interval_hours: Optional[float] = Field(default=None, ge=0.25, le=168)
    output_folder: Optional[str] = None
    repeat_mode: Optional[bool] = None
    run_immediately: Optional[bool] = None


class AutomationCreateRequest(BaseModel):
    name: str = "Velit multi-location run"
    interval_hours: float = Field(default=4.0, ge=0.25, le=168)
    output_folder: str = ""
    repeat_mode: bool = False
    run_immediately: bool = True


class QueueItemRequest(BaseModel):
    automation_id: Optional[str] = None
    label: str = ""
    location: str
    target_count: int = Field(default=10, ge=1, le=100)
    target_role: str = "Founder"
    industry: str = "Van Upfitter"
    seed_query: str = ""
    output_folder: str = ""


class QueueItemUpdateRequest(BaseModel):
    label: Optional[str] = None
    location: Optional[str] = None
    target_count: Optional[int] = Field(default=None, ge=1, le=100)
    target_role: Optional[str] = None
    industry: Optional[str] = None
    seed_query: Optional[str] = None
    output_folder: Optional[str] = None


class ReorderRequest(BaseModel):
    item_ids: List[str]


def _scheduler() -> VelitBatchScheduler:
    state = get_state()
    scheduler = getattr(state, "velit_batch_scheduler", None)
    if scheduler is None:
        scheduler = VelitBatchScheduler()
        state.velit_batch_scheduler = scheduler
    return scheduler


def _raise_api_error(exc: Exception) -> None:
    if isinstance(exc, KeyError):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


def _download_slug(value: str, fallback: str = "velit_exports") -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")
    return slug or fallback


@router.get("/automation")
async def get_automation(automation_id: Optional[str] = None) -> Dict[str, Any]:
    try:
        return await _scheduler().get_snapshot(automation_id)
    except Exception as exc:
        _raise_api_error(exc)


@router.post("/automations")
async def create_automation(request: AutomationCreateRequest) -> Dict[str, Any]:
    try:
        return await _scheduler().create_automation(
            name=request.name,
            interval_hours=request.interval_hours,
            output_folder=request.output_folder,
            repeat_mode=request.repeat_mode,
            run_immediately=request.run_immediately,
        )
    except Exception as exc:
        _raise_api_error(exc)


@router.delete("/automations/{automation_id}")
async def delete_automation(automation_id: str) -> Dict[str, Any]:
    try:
        return await _scheduler().delete_automation(automation_id)
    except Exception as exc:
        _raise_api_error(exc)


@router.post("/automation")
async def update_automation(request: AutomationRequest, automation_id: Optional[str] = None) -> Dict[str, Any]:
    try:
        return await _scheduler().update_automation(
            automation_id=automation_id,
            name=request.name,
            interval_hours=request.interval_hours,
            output_folder=request.output_folder,
            repeat_mode=request.repeat_mode,
            run_immediately=request.run_immediately,
        )
    except Exception as exc:
        _raise_api_error(exc)


@router.post("/automation/start")
async def start_automation(automation_id: Optional[str] = None) -> Dict[str, Any]:
    try:
        return await _scheduler().start_automation(automation_id)
    except Exception as exc:
        _raise_api_error(exc)


@router.post("/automation/pause")
async def pause_automation(automation_id: Optional[str] = None) -> Dict[str, Any]:
    try:
        return await _scheduler().pause_automation(automation_id)
    except Exception as exc:
        _raise_api_error(exc)


@router.post("/automation/resume")
async def resume_automation(automation_id: Optional[str] = None) -> Dict[str, Any]:
    try:
        return await _scheduler().resume_automation(automation_id)
    except Exception as exc:
        _raise_api_error(exc)


@router.post("/automation/stop")
async def stop_automation(automation_id: Optional[str] = None) -> Dict[str, Any]:
    try:
        return await _scheduler().stop_automation(automation_id)
    except Exception as exc:
        _raise_api_error(exc)


@router.post("/automation/reset-timer")
async def reset_automation_timer(automation_id: Optional[str] = None) -> Dict[str, Any]:
    try:
        return await _scheduler().reset_timer(automation_id)
    except Exception as exc:
        _raise_api_error(exc)


@router.get("/items")
async def list_items(automation_id: Optional[str] = None) -> Dict[str, Any]:
    try:
        return await _scheduler().list_items(automation_id)
    except Exception as exc:
        _raise_api_error(exc)


@router.post("/items")
async def add_item(request: QueueItemRequest) -> Dict[str, Any]:
    try:
        return await _scheduler().add_item(
            automation_id=request.automation_id,
            label=request.label,
            location=request.location,
            target_count=request.target_count,
            target_role=request.target_role,
            industry=request.industry,
            seed_query=request.seed_query,
            output_folder=request.output_folder,
        )
    except Exception as exc:
        _raise_api_error(exc)


@router.patch("/items/{item_id}")
async def update_item(item_id: str, request: QueueItemUpdateRequest) -> Dict[str, Any]:
    try:
        return await _scheduler().update_item(item_id, **request.dict(exclude_unset=True))
    except Exception as exc:
        _raise_api_error(exc)


@router.delete("/items/{item_id}")
async def delete_item(item_id: str) -> Dict[str, Any]:
    try:
        return await _scheduler().delete_item(item_id)
    except Exception as exc:
        _raise_api_error(exc)


@router.post("/items/{item_id}/run")
async def run_item_now(item_id: str) -> Dict[str, Any]:
    try:
        return await _scheduler().run_item_now(item_id)
    except Exception as exc:
        _raise_api_error(exc)


@router.post("/items/{item_id}/reset")
async def reset_item(item_id: str) -> Dict[str, Any]:
    try:
        return await _scheduler().reset_item(item_id)
    except Exception as exc:
        _raise_api_error(exc)


@router.post("/items/reorder")
async def reorder_items(request: ReorderRequest, automation_id: Optional[str] = None) -> Dict[str, Any]:
    try:
        return await _scheduler().reorder_items(request.item_ids, automation_id=automation_id)
    except Exception as exc:
        _raise_api_error(exc)


@router.get("/runs")
async def get_runs(limit: int = 50, automation_id: Optional[str] = None) -> Dict[str, Any]:
    try:
        return await _scheduler().recent_runs(limit=limit, automation_id=automation_id)
    except Exception as exc:
        _raise_api_error(exc)


@router.get("/runs/download-all")
async def download_all_runs(automation_id: Optional[str] = None) -> StreamingResponse:
    try:
        payload = await _scheduler().get_downloadable_run_files(automation_id)
        files = payload.get("files") or []
        if not files:
            raise ValueError("No Excel output files are available for this automation.")

        archive = io.BytesIO()
        used_names: set[str] = set()
        with zipfile.ZipFile(archive, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_file:
            for index, file_info in enumerate(files, start=1):
                path = file_info["path"]
                run = file_info.get("run") or {}
                base_name = path.name
                if base_name in used_names:
                    base_name = f"{index:03d}_{base_name}"
                used_names.add(base_name)
                zip_file.write(path, arcname=base_name)
        archive.seek(0)

        filename = f"{_download_slug(payload.get('automation_name') or payload.get('automation_id'))}_excel_exports.zip"
        return StreamingResponse(
            archive,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as exc:
        _raise_api_error(exc)


@router.get("/runs/{run_id}/download")
async def download_run(run_id: str, automation_id: Optional[str] = None) -> FileResponse:
    try:
        path = await _scheduler().get_run_output_file(run_id, automation_id=automation_id)
        return FileResponse(
            path,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename=path.name,
        )
    except Exception as exc:
        _raise_api_error(exc)
