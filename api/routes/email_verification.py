import asyncio
import io
import json
import re
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from core.utils.email_verify import verify_email_legitimacy


router = APIRouter()

EMAIL_PATTERN = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
DEFAULT_STATUS_HEADER = "email_verification_status"
EMPTY_EMAIL_VALUES = {"", "-", "--", "—", "n/a", "na", "none", "null", "not found"}
EMAIL_HEADER_ALIASES = {
    "email",
    "emailaddress",
    "emailid",
    "contactemail",
    "contactemailaddress",
    "personemail",
    "leademail",
    "companyemail",
    "workemail",
    "businessemail",
    "e-mail",
    "mail",
}


def _normalize_header(value: Any) -> str:
    text = str(value or "").strip().lower()
    return re.sub(r"[^a-z0-9-]+", "", text)


def _extract_email(value: Any) -> str:
    match = EMAIL_PATTERN.search(str(value or ""))
    return match.group(0).strip() if match else ""


def _find_email_column(ws, requested_header: str = "") -> int | None:
    headers = [ws.cell(row=1, column=column).value for column in range(1, ws.max_column + 1)]
    requested = _normalize_header(requested_header)

    if requested:
        for index, header in enumerate(headers, start=1):
            if _normalize_header(header) == requested:
                return index
        return None

    for index, header in enumerate(headers, start=1):
        normalized = _normalize_header(header)
        if normalized in EMAIL_HEADER_ALIASES:
            return index
        if "email" in normalized and "verification" not in normalized and "status" not in normalized:
            return index

    best_column = None
    best_hits = 0
    max_scan_row = min(ws.max_row, 50)
    for column in range(1, ws.max_column + 1):
        hits = 0
        for row in range(1, max_scan_row + 1):
            if _extract_email(ws.cell(row=row, column=column).value):
                hits += 1
        if hits > best_hits:
            best_hits = hits
            best_column = column

    return best_column if best_hits else None


def _ensure_status_column(ws, header: str = DEFAULT_STATUS_HEADER) -> int:
    normalized_target = _normalize_header(header)
    for column in range(1, ws.max_column + 1):
        if _normalize_header(ws.cell(row=1, column=column).value) == normalized_target:
            return column

    column = ws.max_column + 1
    ws.cell(row=1, column=column).value = header
    return column


def _verification_label(verification: dict) -> str:
    status = str(verification.get("status") or "").strip().lower()
    confidence = verification.get("confidence", 0)
    message = str(verification.get("message") or "").strip()

    if status == "valid":
        label = "verified"
    elif status == "risky":
        label = "needs manual review"
    elif status in {"invalid", "error"}:
        label = status
    else:
        label = status or "unknown"

    if confidence not in ("", None):
        label = f"{label} ({confidence}%)"
    if message:
        label = f"{label} - {message}"
    return label


def _load_workbook_from_bytes(data: bytes):
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail="openpyxl is not installed. Run `pip install -r requirements.txt` and restart the server.",
        ) from exc

    try:
        return load_workbook(io.BytesIO(data))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read Excel file: {type(exc).__name__}") from exc


@router.post("/bulk-excel")
async def verify_excel_emails(
    file: UploadFile = File(...),
    email_column: str = Form(""),
    max_concurrency: int = Form(4),
):
    filename = file.filename or "emails.xlsx"
    if not filename.lower().endswith((".xlsx", ".xlsm")):
        raise HTTPException(status_code=400, detail="Upload a .xlsx or .xlsm Excel file.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Excel file is too large. Keep it under 8 MB.")

    workbook = await run_in_threadpool(_load_workbook_from_bytes, data)
    jobs: list[tuple[Any, int, int, str]] = []
    no_email_rows: list[tuple[Any, int, int]] = []
    sheet_summaries: list[dict[str, Any]] = []

    for ws in workbook.worksheets:
        email_col = _find_email_column(ws, email_column)
        if not email_col:
            sheet_summaries.append({"sheet": ws.title, "email_column_found": False, "emails": 0})
            continue

        status_col = _ensure_status_column(ws)
        count = 0
        for row in range(2, ws.max_row + 1):
            raw_value = ws.cell(row=row, column=email_col).value
            raw_text = str(raw_value or "").strip()
            email = _extract_email(raw_text)
            if not email and raw_text.lower() in EMPTY_EMAIL_VALUES:
                no_email_rows.append((ws, row, status_col))
                continue
            if not email:
                email = raw_text
            if not email:
                continue
            jobs.append((ws, row, status_col, email))
            count += 1
        sheet_summaries.append({"sheet": ws.title, "email_column_found": True, "emails": count})

    if not jobs:
        if no_email_rows:
            for ws, row, status_col in no_email_rows:
                ws.cell(row=row, column=status_col).value = "no_email_found"
            output = io.BytesIO()
            await run_in_threadpool(workbook.save, output)
            output.seek(0)
            download_name = f"verified_{filename.rsplit('.', 1)[0]}.xlsx"
            headers = {
                "Content-Disposition": f'attachment; filename="{download_name}"',
                "X-Email-Verification-Summary": json.dumps(
                    {
                        "total_emails": 0,
                        "status_counts": {"no_email_found": len(no_email_rows)},
                        "sheets": sheet_summaries,
                    }
                ),
            }
            return StreamingResponse(
                output,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers=headers,
            )
        raise HTTPException(status_code=400, detail="No email addresses were found in the workbook.")

    concurrency = max(1, min(int(max_concurrency or 4), 8))
    semaphore = asyncio.Semaphore(concurrency)

    async def verify_job(job: tuple[Any, int, int, str]):
        async with semaphore:
            ws, row, status_col, email = job
            verification = await run_in_threadpool(verify_email_legitimacy, email)
            return ws, row, status_col, verification

    results = await asyncio.gather(*(verify_job(job) for job in jobs))
    status_counts: dict[str, int] = {}

    for ws, row, status_col in no_email_rows:
        ws.cell(row=row, column=status_col).value = "no_email_found"
    if no_email_rows:
        status_counts["no_email_found"] = len(no_email_rows)

    for ws, row, status_col, verification in results:
        status = str(verification.get("status") or "unknown").lower()
        summary_status = "needs_manual_review" if status == "risky" else status
        status_counts[summary_status] = status_counts.get(summary_status, 0) + 1
        ws.cell(row=row, column=status_col).value = _verification_label(verification)

    output = io.BytesIO()
    await run_in_threadpool(workbook.save, output)
    output.seek(0)

    summary = {
        "total_emails": len(jobs),
        "status_counts": status_counts,
        "sheets": sheet_summaries,
    }
    download_name = f"verified_{filename.rsplit('.', 1)[0]}.xlsx"
    headers = {
        "Content-Disposition": f'attachment; filename="{download_name}"',
        "X-Email-Verification-Summary": json.dumps(summary),
    }
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers=headers,
    )
