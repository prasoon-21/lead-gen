import json
import os
import uuid
import base64
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    import gspread
    from gspread import Worksheet
    from gspread.exceptions import WorksheetNotFound
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    gspread = None
    Worksheet = Any
    WorksheetNotFound = Exception
    Credentials = None
    GSPREAD_AVAILABLE = False

try:
    from googleapiclient.discovery import build
    GOOGLE_API_CLIENT_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    build = None
    GOOGLE_API_CLIENT_AVAILABLE = False

from core.utils.lead_summary import build_plain_lead_summary


class TodoSheetStore:
    LEAD_META_PREFIX = "[AGENTIC_META]"
    SCOPES = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.file",
        "https://www.googleapis.com/auth/drive.metadata.readonly",
    ]

    TASK_HEADERS = [
        "task_id",
        "title",
        "description",
        "category",
        "project",
        "status",
        "priority",
        "due_date",
        "tags",
        "notes",
        "created_at",
        "updated_at",
        "completed_at",
        "source_agent",
        "trace_id",
    ]

    # Dedicated leads worksheet columns (order matters)
    LEAD_HEADERS = [
        "Company Name",
        "Specialty",
        "Lead Summary",
        "Why Accepted",
        "Contact Name",
        "Email",
        "Phone",
        "Alternate Phones",
        "Phone Confidence",
        "Phone Source",
        "Phone Status",
        "Website",
        "City",
        "State",
        "Zip Code",
        "Status",
        "Notes",
        "Sources", # New column for citation summary
    ]

    AUDIT_HEADERS = [
        "event_id",
        "timestamp",
        "action",
        "task_id",
        "trace_id",
        "session_id",
        "payload_json",
        "before_json",
        "after_json",
        "status",
        "message",
    ]

    def __init__(
        self,
        credentials_path: Optional[str] = None,
        sheet_id: Optional[str] = None,
        sheet_name: Optional[str] = None,
        folder_id: Optional[str] = None,
        tasks_worksheet_name: Optional[str] = None,
        audit_worksheet_name: Optional[str] = None,
    ):
        self.credentials_path = credentials_path or os.getenv("GOOGLE_CREDENTIALS_PATH")
        self.credentials_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
        self.credentials_json_b64 = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON_BASE64")
        self.sheet_id = sheet_id or os.getenv("TODO_SHEET_ID")
        self.sheet_name = sheet_name or os.getenv("TODO_SHEET_FILE_NAME", "Aurika Agent Todo")
        self.folder_id = folder_id or os.getenv("GOOGLE_DRIVE_FOLDER_ID")
        self.tasks_worksheet_name = tasks_worksheet_name or os.getenv("TODO_WORKSHEET_NAME", "tasks")
        self.audit_worksheet_name = audit_worksheet_name or os.getenv("TODO_AUDIT_WORKSHEET", "audit_log")
        self.leads_worksheet_name = os.getenv("LEADS_WORKSHEET_NAME", "leads")

        self._creds = None
        self._gspread_client = None
        self._drive_service = None
        self._spreadsheet = None
        self._tasks_ws = None
        self._audit_ws = None
        self._leads_ws = None
        self._task_headers_current: List[str] = list(self.TASK_HEADERS)
        self._audit_headers_current: List[str] = list(self.AUDIT_HEADERS)
        self._lead_headers_current: List[str] = list(self.LEAD_HEADERS)

    @staticmethod
    def _now() -> str:
        return datetime.utcnow().isoformat()

    @staticmethod
    def _to_json(value: Any) -> str:
        if value is None:
            return ""
        return json.dumps(value, ensure_ascii=False)

    def _load_credentials(self):
        if self._creds is not None:
            return self._creds
        if not GSPREAD_AVAILABLE:
            raise RuntimeError("gspread is not installed")

        def _from_file(path: str):
            if path and os.path.exists(path):
                return Credentials.from_service_account_file(path, scopes=self.SCOPES)
            return None

        def _decode_raw_json(raw: str) -> Optional[Dict[str, Any]]:
            text = (raw or "").strip()
            if not text:
                return None
            if (text.startswith("'") and text.endswith("'")) or (text.startswith('"') and text.endswith('"')):
                text = text[1:-1].strip()
            if text.startswith("{"):
                return json.loads(text)
            if os.path.exists(text):
                with open(text, "r", encoding="utf-8") as f:
                    return json.load(f)
            # Backward-compatible behavior: allow base64 JSON in GOOGLE_SERVICE_ACCOUNT_JSON
            try:
                normalized = text.strip()
                padding = len(normalized) % 4
                if padding:
                    normalized += "=" * (4 - padding)
                decoded = base64.b64decode(normalized).decode("utf-8")
                if decoded.strip().startswith("{"):
                    return json.loads(decoded)
            except Exception:
                pass
            return None

        if self.credentials_json:
            try:
                creds_dict = _decode_raw_json(self.credentials_json)
                if creds_dict:
                    self._creds = Credentials.from_service_account_info(creds_dict, scopes=self.SCOPES)
                    return self._creds
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid GOOGLE_SERVICE_ACCOUNT_JSON format: {exc}. "
                    "Provide raw JSON, quoted JSON, or a file path."
                ) from exc

        if self.credentials_json_b64:
            try:
                decoded = base64.b64decode(self.credentials_json_b64).decode("utf-8")
                creds_dict = json.loads(decoded)
                self._creds = Credentials.from_service_account_info(creds_dict, scopes=self.SCOPES)
                return self._creds
            except Exception as exc:
                raise RuntimeError(
                    f"Invalid GOOGLE_SERVICE_ACCOUNT_JSON_BASE64: {exc}. "
                    "Expected base64-encoded service account JSON."
                ) from exc

        creds_from_path = _from_file(self.credentials_path or "")
        if creds_from_path:
            self._creds = creds_from_path
            return self._creds

        raise RuntimeError(
            "Google credentials not configured. Set GOOGLE_SERVICE_ACCOUNT_JSON (raw JSON), "
            "GOOGLE_SERVICE_ACCOUNT_JSON_BASE64, or GOOGLE_CREDENTIALS_PATH."
        )

    def _init_clients(self):
        if self._gspread_client is not None:
            return
        creds = self._load_credentials()
        self._gspread_client = gspread.authorize(creds)
        if GOOGLE_API_CLIENT_AVAILABLE:
            self._drive_service = build("drive", "v3", credentials=creds, cache_discovery=False)

    def _find_sheet_id(self) -> Optional[str]:
        if self.sheet_id:
            return self.sheet_id

        if self._drive_service:
            query = (
                "mimeType='application/vnd.google-apps.spreadsheet' "
                f"and trashed=false and name='{self.sheet_name}'"
            )
            if self.folder_id:
                query += f" and '{self.folder_id}' in parents"
            response = self._drive_service.files().list(
                q=query,
                pageSize=1,
                fields="files(id,name)",
            ).execute()
            files = response.get("files", [])
            if files:
                return files[0]["id"]

        for sheet in self._gspread_client.openall(self.sheet_name):
            if sheet.title == self.sheet_name:
                return sheet.id

        return None

    def _create_sheet_id(self) -> str:
        if self._drive_service:
            body = {
                "name": self.sheet_name,
                "mimeType": "application/vnd.google-apps.spreadsheet",
            }
            if self.folder_id:
                body["parents"] = [self.folder_id]
            created = self._drive_service.files().create(body=body, fields="id").execute()
            return created["id"]

        sheet = self._gspread_client.create(self.sheet_name)
        return sheet.id

    @staticmethod
    def _ensure_worksheet(spreadsheet, title: str, rows: int = 2000, cols: int = 30) -> Worksheet:
        try:
            return spreadsheet.worksheet(title)
        except WorksheetNotFound:
            return spreadsheet.add_worksheet(title=title, rows=rows, cols=cols)

    @staticmethod
    def _ensure_headers(worksheet: Worksheet, required_headers: List[str]) -> List[str]:
        existing = worksheet.row_values(1)
        merged = list(existing)
        for header in required_headers:
            if header not in merged:
                merged.append(header)

        if not existing:
            worksheet.update("1:1", [required_headers])
            return list(required_headers)

        if merged != existing:
            worksheet.update("1:1", [merged])
        return merged

    def ensure_store(self) -> Dict[str, Any]:
        self._init_clients()
        sheet_id = self._find_sheet_id() or self._create_sheet_id()
        try:
            self._spreadsheet = self._gspread_client.open_by_key(sheet_id)
        except Exception as exc:
            raise RuntimeError(
                f"Unable to open todo spreadsheet '{sheet_id}'. Ensure it exists and service account has Editor access."
            ) from exc

        self._tasks_ws = self._ensure_worksheet(self._spreadsheet, self.tasks_worksheet_name)
        self._audit_ws = self._ensure_worksheet(self._spreadsheet, self.audit_worksheet_name)
        self._leads_ws = self._ensure_worksheet(self._spreadsheet, self.leads_worksheet_name)

        self._task_headers_current = self._ensure_headers(self._tasks_ws, self.TASK_HEADERS)
        self._audit_headers_current = self._ensure_headers(self._audit_ws, self.AUDIT_HEADERS)
        self._lead_headers_current = self._ensure_headers(self._leads_ws, self.LEAD_HEADERS)

        return {
            "spreadsheet_id": sheet_id,
            "spreadsheet_name": self._spreadsheet.title,
            "tasks_worksheet": self.tasks_worksheet_name,
            "audit_worksheet": self.audit_worksheet_name,
            "leads_worksheet": self.leads_worksheet_name,
            "task_headers": self._task_headers_current,
        }

    def _require_ready(self):
        if self._tasks_ws is None or self._audit_ws is None or self._leads_ws is None:
            self.ensure_store()

    def _find_task_row(self, task_id: str) -> Tuple[int, Dict[str, Any]]:
        self._require_ready()
        rows = self._tasks_ws.get_all_values()
        if not rows:
            raise ValueError(f"Task not found: {task_id}")

        headers = rows[0]
        for idx, values in enumerate(rows[1:], start=2):
            row_data = {headers[i]: values[i] if i < len(values) else "" for i in range(len(headers))}
            if str(row_data.get("task_id", "")).strip() == str(task_id).strip():
                return idx, row_data
        raise ValueError(f"Task not found: {task_id}")

    @staticmethod
    def _normalize_tags(tags: Any) -> str:
        if tags is None:
            return ""
        if isinstance(tags, str):
            return tags
        if isinstance(tags, list):
            return ", ".join([str(item).strip() for item in tags if str(item).strip()])
        return str(tags)

    def _append_audit(
        self,
        action: str,
        task_id: str,
        trace_id: str = "",
        session_id: str = "",
        payload: Optional[Dict[str, Any]] = None,
        before: Optional[Dict[str, Any]] = None,
        after: Optional[Dict[str, Any]] = None,
        status: str = "ok",
        message: str = "",
    ) -> None:
        self._require_ready()
        record = {
            "event_id": f"EVT_{uuid.uuid4().hex[:10]}",
            "timestamp": self._now(),
            "action": action,
            "task_id": task_id,
            "trace_id": trace_id,
            "session_id": session_id,
            "payload_json": self._to_json(payload),
            "before_json": self._to_json(before),
            "after_json": self._to_json(after),
            "status": status,
            "message": message,
        }
        row = [record.get(h, "") for h in self._audit_headers_current]
        self._audit_ws.append_row(row, value_input_option="USER_ENTERED")

    def create_task(
        self,
        payload: Dict[str, Any],
        trace_id: str = "",
        session_id: str = "",
        source_agent: str = "",
    ) -> Dict[str, Any]:
        self._require_ready()
        now = self._now()
        task = {
            "task_id": payload.get("task_id") or f"TASK_{uuid.uuid4().hex[:10]}",
            "title": str(payload.get("title", "")).strip(),
            "description": str(payload.get("description", "")).strip(),
            "category": str(payload.get("category", "")).strip(),
            "project": str(payload.get("project", "")).strip(),
            "status": str(payload.get("status", "pending")).strip() or "pending",
            "priority": str(payload.get("priority", "medium")).strip() or "medium",
            "due_date": str(payload.get("due_date", "")).strip(),
            "tags": self._normalize_tags(payload.get("tags")),
            "notes": str(payload.get("notes", "")).strip(),
            "created_at": now,
            "updated_at": now,
            "completed_at": "",
            "source_agent": source_agent,
            "trace_id": trace_id,
        }
        if not task["title"]:
            raise ValueError("title is required")

        row = [task.get(h, "") for h in self._task_headers_current]
        self._tasks_ws.append_row(row, value_input_option="USER_ENTERED")
        self._append_audit(
            action="create_task",
            task_id=task["task_id"],
            trace_id=trace_id,
            session_id=session_id,
            payload=payload,
            after=task,
            status="ok",
            message="Task created",
        )
        return task

    def list_tasks(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        self._require_ready()
        filters = filters or {}
        records = self._tasks_ws.get_all_records(default_blank="")
        status_filter = filters.get("status")
        if not status_filter:
            status_values = ["pending", "in_progress", "blocked"]
        elif isinstance(status_filter, list):
            status_values = [str(item).strip() for item in status_filter if str(item).strip()]
        else:
            status_values = [part.strip() for part in str(status_filter).split(",") if part.strip()]

        category = str(filters.get("category", "")).strip().lower()
        project = str(filters.get("project", "")).strip().lower()
        search = str(filters.get("search", "")).strip().lower()
        limit = int(filters.get("limit", 50))
        include_done = bool(filters.get("include_done", False))

        result: List[Dict[str, Any]] = []
        for item in records:
            row = {k: item.get(k, "") for k in self._task_headers_current}
            status = str(row.get("status", "")).strip().lower()
            if not include_done and status == "done":
                continue
            if status_values and status and status not in [x.lower() for x in status_values]:
                continue
            if category and category not in str(row.get("category", "")).lower():
                continue
            if project and project not in str(row.get("project", "")).lower():
                continue
            if search:
                joined = " ".join(
                    [
                        str(row.get("title", "")),
                        str(row.get("description", "")),
                        str(row.get("tags", "")),
                        str(row.get("notes", "")),
                    ]
                ).lower()
                if search not in joined:
                    continue
            result.append(row)
            if len(result) >= limit:
                break
        return result

    def get_task(self, task_id: str) -> Dict[str, Any]:
        _, task = self._find_task_row(task_id)
        return task

    def update_task(
        self,
        task_id: str,
        updates: Dict[str, Any],
        trace_id: str = "",
        session_id: str = "",
    ) -> Dict[str, Any]:
        self._require_ready()
        row_idx, before = self._find_task_row(task_id)
        after = dict(before)
        allowed_fields = set(self._task_headers_current) - {"task_id", "created_at"}
        for key, value in (updates or {}).items():
            if key not in allowed_fields:
                continue
            if key == "tags":
                after[key] = self._normalize_tags(value)
            else:
                after[key] = str(value)
        after["updated_at"] = self._now()
        if str(after.get("status", "")).strip().lower() == "done" and not after.get("completed_at"):
            after["completed_at"] = self._now()

        row_values = [after.get(h, "") for h in self._task_headers_current]
        end_col = gspread.utils.rowcol_to_a1(1, len(self._task_headers_current)).replace("1", "")
        self._tasks_ws.update(f"A{row_idx}:{end_col}{row_idx}", [row_values], value_input_option="USER_ENTERED")
        self._append_audit(
            action="update_task",
            task_id=task_id,
            trace_id=trace_id,
            session_id=session_id,
            payload=updates,
            before=before,
            after=after,
            status="ok",
            message="Task updated",
        )
        return after

    def complete_task(
        self,
        task_id: str,
        completion_note: str = "",
        trace_id: str = "",
        session_id: str = "",
    ) -> Dict[str, Any]:
        updates = {
            "status": "done",
            "completed_at": self._now(),
        }
        if completion_note:
            updates["notes"] = completion_note
        return self.update_task(
            task_id=task_id,
            updates=updates,
            trace_id=trace_id,
            session_id=session_id,
        )

    # ------------------------------------------------------------------
    # Lead-specific helpers
    # ------------------------------------------------------------------

    def save_lead(self, lead: Dict[str, Any]) -> Dict[str, Any]:
        """
        Append a single lead row to the dedicated 'leads' worksheet.
        Column order: Company Name, Specialty, Contact Name, Email, Phone,
        Website, City, State, Zip Code, Status, Notes.
        Returns the dict that was written.
        """
        self._require_ready()
        notes = str(lead.get("notes", "") or lead.get("value_proposition", "")).strip()
        alternate_phones = lead.get("alternate_phones") or []
        alternate_phones_text = (
            alternate_phones.strip()
            if isinstance(alternate_phones, str)
            else " | ".join(str(phone).strip() for phone in alternate_phones if str(phone).strip())
        )
        
        metadata = self._build_lead_metadata(lead)
        serialized_notes = self._serialize_lead_notes(notes, metadata)
        row_data = {
            "Company Name": str(lead.get("company_name", "")).strip(),
            "Specialty":    str(lead.get("specialty", "")).strip(),
            "Lead Summary": build_plain_lead_summary(lead),
            "Why Accepted": str(lead.get("acceptance_reason", "") or lead.get("confidence_explanation", "")).strip(),
            "Contact Name": str(lead.get("contact_name", "") or lead.get("contact_person_name", "") or lead.get("founder_name", "")).strip(),
            "Email":        str(lead.get("email", "") or lead.get("contact_email", "")).strip(),
            "Phone":        str(lead.get("phone", "") or lead.get("contact_phone", "")).strip(),
            "Alternate Phones": alternate_phones_text,
            "Phone Confidence": str(lead.get("phone_confidence", "")).strip(),
            "Phone Source": str(lead.get("phone_source", "")).strip(),
            "Phone Status": str(lead.get("phone_validation_status", "")).strip(),
            "Website":      str(lead.get("website", "") or lead.get("company_website", "")).strip(),
            "City":         str(lead.get("city", "")).strip(),
            "State":        str(lead.get("state", "")).strip(),
            "Zip Code":     str(lead.get("zip_code", "")).strip(),
            "Status":       str(lead.get("status", "New")).strip() or "New",
            "Notes":        serialized_notes,
            "Sources":      metadata.pop("citation_summary", ""), # Populate new Sources column
        }
        row = [row_data.get(h, "") for h in self._lead_headers_current]
        self._leads_ws.append_row(row, value_input_option="USER_ENTERED")
        return row_data

    def list_leads(self) -> List[Dict[str, Any]]:
        """Return all rows from the leads worksheet as a list of dicts."""
        self._require_ready()
        records = self._leads_ws.get_all_records(default_blank="")
        enriched: List[Dict[str, Any]] = []
        for row in records:
            item = dict(row)
            raw_notes = str(item.get("Notes", "") or "")
            note_text, metadata = self._parse_lead_notes(raw_notes)
            item["Notes"] = note_text
            item["metadata"] = metadata
            enriched.append(item)
        return enriched

    def get_lead_company_names(self) -> List[str]:
        """Return existing company names from the leads sheet (for deduplication)."""
        self._require_ready()
        rows = self._leads_ws.get_all_values()
        if not rows or len(rows) < 2:
            return []
        headers = rows[0]
        try:
            col_idx = headers.index("Company Name")
        except ValueError:
            return []
        return [row[col_idx].strip() for row in rows[1:] if col_idx < len(row) and row[col_idx].strip()]

    @classmethod
    def _build_lead_metadata(cls, lead: Dict[str, Any]) -> Dict[str, Any]:
        source_value = lead.get("source", "")
        metadata = {
            "linkedin_url": str(lead.get("linkedin_url", "")).strip(),
            "contact_page": str(lead.get("contact_page", "")).strip(),
            "source": str(source_value).strip(),
            "source_details": lead.get("source_details", []) or [],
            "confidence": str(lead.get("confidence", "")).strip(),
            "quality_score": lead.get("quality_score", 0),
            "quality_status": str(lead.get("quality_status", "")).strip(),
            "lead_summary": build_plain_lead_summary(lead),
            "acceptance_reason": str(lead.get("acceptance_reason", "") or lead.get("confidence_explanation", "")).strip(),
            "contact_person_title": str(lead.get("contact_person_title", "")).strip(),
            "contact_person_name": str(lead.get("contact_person_name", "") or lead.get("founder_name", "")).strip(),
            "field_sources": lead.get("field_sources", {}) or {},
            "contact_paths": lead.get("contact_paths", []) or [],
            "alternate_phones": lead.get("alternate_phones", []) or [],
            "phone_confidence": lead.get("phone_confidence", ""),
            "phone_source": str(lead.get("phone_source", "")).strip(),
            "phone_validation_status": str(lead.get("phone_validation_status", "")).strip(),
            "phone_type": str(lead.get("phone_type", "")).strip(),
            "target_domain": str(lead.get("target_domain", "")).strip(),
            "pipeline_disposition": str(lead.get("pipeline_disposition", "")).strip(),
            "rejection_reason_code": str(lead.get("rejection_reason_code", "")).strip(),
            "contacts": lead.get("contacts", []) or [],
            "alternative_contacts": lead.get("alternative_contacts", {}) or {},
        }

        # Build citation summary
        citation_parts = []
        seen_urls_for_summary = set()

        # Prioritize company website
        company_website = str(lead.get("company_website", "")).strip()
        if company_website and company_website not in seen_urls_for_summary:
            citation_parts.append(f"Company Website: {company_website}")
            seen_urls_for_summary.add(company_website)

        # Add LinkedIn URL
        linkedin_url = str(lead.get("linkedin_url", "")).strip()
        if linkedin_url and linkedin_url not in seen_urls_for_summary:
            citation_parts.append(f"LinkedIn: {linkedin_url}")
            seen_urls_for_summary.add(linkedin_url)

        # Add Contact Page
        contact_page = str(lead.get("contact_page", "")).strip()
        if contact_page and contact_page not in seen_urls_for_summary and contact_page != company_website:
            citation_parts.append(f"Contact Page: {contact_page}")
            seen_urls_for_summary.add(contact_page)

        # Add URLs from source_details
        source_details = metadata.get("source_details", [])
        for detail in source_details:
            if isinstance(detail, dict) and "url" in detail:
                detail_url = str(detail["url"]).strip()
                if detail_url and detail_url not in seen_urls_for_summary:
                    label = "Source"
                    if detail.get("type") == "trusted_directory_seed":
                        label = "Directory Source"
                    elif detail.get("provider") == "tavily_extract":
                        label = "Content Source"
                    citation_parts.append(f"{label}: {detail_url}")
                    seen_urls_for_summary.add(detail_url)

        citation_summary = "Sources: " + "; ".join(citation_parts) if citation_parts else ""
        if citation_summary:
            metadata["citation_summary"] = citation_summary
        return {
            key: value
            for key, value in metadata.items()
            if value not in ("", [], {}, None)
        }

    @classmethod
    def _serialize_lead_notes(cls, notes: str, metadata: Dict[str, Any]) -> str:
        # citation_summary is now handled in its own column in save_lead, so we don't include it here.
        if not metadata:
            return notes or ""
        payload = json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
        full_notes_content = []
        if notes: full_notes_content.append(notes)
        if payload != "{}": full_notes_content.append(f"{cls.LEAD_META_PREFIX}{payload}")
        return "\n\n".join(full_notes_content)

    @classmethod
    def _parse_lead_notes(cls, value: str) -> Tuple[str, Dict[str, Any]]:
        text = str(value or "")
        if cls.LEAD_META_PREFIX not in text:
            return text, {}
        base, meta = text.split(cls.LEAD_META_PREFIX, 1)
        base = base.strip()
        try:
            return base, json.loads(meta.strip())
        except Exception:
            return base, {}
