"""
ToolStore — CRUD for remote tool registrations in Firestore.

Firestore path: businesses/{business_id}/remote_tools/{tool_name}

Each document mirrors the YAML format used by RemoteToolWrapper so the
same loader can hydrate tools from DB or files transparently.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.db.firestore_client import FirestoreClient

COLLECTION = "remote_tools"


class ToolStore:
    def __init__(self, client: FirestoreClient):
        self._client = client

    # ── Read ──────────────────────────────────────────────────────────────────

    async def get(self, tool_name: str) -> Optional[Dict[str, Any]]:
        return await self._client.get(COLLECTION, tool_name)

    async def list_all(self) -> List[Dict[str, Any]]:
        return await self._client.list_all(COLLECTION)

    async def exists(self, tool_name: str) -> bool:
        return await self._client.exists(COLLECTION, tool_name)

    # ── Write ─────────────────────────────────────────────────────────────────

    async def save(self, tool_name: str, data: Dict[str, Any]) -> None:
        """Register or update a remote tool."""
        payload = {**data, "tool_name": tool_name}
        payload.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        await self._client.set(COLLECTION, tool_name, payload)

    async def delete(self, tool_name: str) -> None:
        await self._client.delete(COLLECTION, tool_name)
