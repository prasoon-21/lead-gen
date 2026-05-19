"""
DirectoryStore — CRUD for the agent directory in Firestore.

Firestore path: businesses/{business_id}/agent_directory/{agent_id}

This replaces the SQLite agent_directory table entirely.
Each document is a deployable agent entry with scope, status, versioning,
and full execution config — everything the old SQLAlchemy AgentRecord had.

Document shape:
    agent_id        str   — UUID, primary key
    name            str   — human-readable name
    slug            str   — URL-friendly identifier (unique per business)
    scope           str   — "generic" | "business" | "business_tenant"
    business_id     str?
    tenant_id       str?
    status          str   — "draft" | "active" | "inactive"
    version         str   — e.g. "1.0"
    description     str?
    welcome_message str?
    placeholder_text str?
    tags            list[str]
    model_profile   str?
    system_prompt   str?
    developer_prompt str?
    response_mode   str   — "text" | "json"
    allowed_tools   list[str]
    default_context dict
    output_schema   dict?
    memory_window   int
    max_steps       int
    max_tool_calls  int
    max_runtime_seconds int
    created_by      str?
    created_at      str   — ISO timestamp
    updated_at      SERVER_TIMESTAMP (set by FirestoreClient.set)
    metadata        dict
"""

import uuid
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.db.firestore_client import FirestoreClient

COLLECTION = "agent_directory"

VALID_SCOPES = {"generic", "business", "business_tenant"}
VALID_STATUSES = {"draft", "active", "inactive"}


def _slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"[\s_-]+", "-", text)


class DirectoryStore:
    def __init__(self, client: FirestoreClient):
        self._client = client

    # ── Read ──────────────────────────────────────────────────────────────────

    async def get(self, agent_id: str) -> Optional[Dict[str, Any]]:
        return await self._client.get(COLLECTION, agent_id)

    async def get_by_slug(self, slug: str) -> Optional[Dict[str, Any]]:
        results = await self._client.query(
            COLLECTION, filters=[("slug", "==", slug)], limit=1
        )
        return results[0] if results else None

    async def list_all(
        self,
        scope: Optional[str] = None,
        business_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        filters = []
        if scope:
            filters.append(("scope", "==", scope))
        if business_id:
            filters.append(("business_id", "==", business_id))
        if tenant_id:
            filters.append(("tenant_id", "==", tenant_id))
        if status:
            filters.append(("status", "==", status))
        return await self._client.query(
            COLLECTION, filters=filters, limit=limit, offset=offset
        )

    async def exists(self, agent_id: str) -> bool:
        return await self._client.exists(COLLECTION, agent_id)

    async def slug_taken(self, slug: str) -> bool:
        existing = await self.get_by_slug(slug)
        return existing is not None

    # ── Write ─────────────────────────────────────────────────────────────────

    async def create(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new agent directory entry. Returns the saved record."""
        agent_id = data.get("agent_id") or str(uuid.uuid4())

        # Auto-derive slug from name if not provided
        slug = data.get("slug") or _slugify(data.get("name", agent_id))

        # Ensure slug uniqueness — append short suffix on collision
        if await self.slug_taken(slug):
            slug = f"{slug}-{str(uuid.uuid4())[:8]}"

        record = {
            "agent_id": agent_id,
            "name": data.get("name", ""),
            "slug": slug,
            "scope": data.get("scope", "generic"),
            "business_id": data.get("business_id"),
            "tenant_id": data.get("tenant_id"),
            "status": data.get("status", "draft"),
            "version": data.get("version", "1.0"),
            "description": data.get("description"),
            "welcome_message": data.get("welcome_message"),
            "placeholder_text": data.get("placeholder_text"),
            "tags": data.get("tags", []),
            "model_profile": data.get("model_profile"),
            "system_prompt": data.get("system_prompt"),
            "developer_prompt": data.get("developer_prompt"),
            "response_mode": data.get("response_mode", "text"),
            "allowed_tools": data.get("allowed_tools", []),
            "default_context": data.get("default_context", {}),
            "output_schema": data.get("output_schema"),
            "memory_window": int(data.get("memory_window", 6)),
            "max_steps": int(data.get("max_steps", 8)),
            "max_tool_calls": int(data.get("max_tool_calls", 6)),
            "max_runtime_seconds": int(data.get("max_runtime_seconds", 45)),
            "created_by": data.get("created_by"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": data.get("metadata", {}),
        }

        await self._client.set(COLLECTION, agent_id, record)
        return record

    async def update(self, agent_id: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Partial update — only keys present in data are changed."""
        existing = await self.get(agent_id)
        if not existing:
            return None
        merged = {**existing, **data, "agent_id": agent_id}
        await self._client.set(COLLECTION, agent_id, merged)
        return await self.get(agent_id)

    async def set_status(self, agent_id: str, status: str) -> Optional[Dict[str, Any]]:
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status '{status}'. Must be one of {VALID_STATUSES}")
        return await self.update(agent_id, {"status": status})

    async def delete(self, agent_id: str) -> None:
        await self._client.delete(COLLECTION, agent_id)
