"""
AgentStore — CRUD for agent definitions in Firestore.

Firestore path: businesses/{business_id}/agents/{agent_id}

Document shape mirrors AgentSpec from config/schemas.py so the ConfigLoader
can deserialise it the same way as a YAML file.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.db.firestore_client import FirestoreClient

COLLECTION = "agents"


class AgentStore:
    def __init__(self, client: FirestoreClient):
        self._client = client

    # ── Read ──────────────────────────────────────────────────────────────────

    async def get(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Return agent definition dict or None if not found."""
        return await self._client.get(COLLECTION, agent_id)

    async def list_all(self) -> List[Dict[str, Any]]:
        """Return all agent definitions for this business."""
        return await self._client.list_all(COLLECTION)

    async def exists(self, agent_id: str) -> bool:
        return await self._client.exists(COLLECTION, agent_id)

    # ── Write ─────────────────────────────────────────────────────────────────

    async def save(self, agent_id: str, data: Dict[str, Any]) -> None:
        """Create or fully replace an agent definition."""
        payload = {**data, "agent_id": agent_id}
        payload.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        await self._client.set(COLLECTION, agent_id, payload)

    async def delete(self, agent_id: str) -> None:
        await self._client.delete(COLLECTION, agent_id)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def from_yaml_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
        """Convert an AgentSpec dict (from YAML) into a Firestore-ready document.

        Strips the system_prompt_path and inlines the actual prompt text
        so the DB record is self-contained.
        """
        return {
            "agent_id": spec.get("agent_id", ""),
            "version": spec.get("version", "1.0"),
            "description": spec.get("description", ""),
            "model_profile": spec.get("model_profile", "large"),
            "system_prompt": spec.get("system_prompt", ""),
            "developer_prompt": spec.get("developer_prompt", ""),
            "response_mode": spec.get("response_mode", "text"),
            "output_schema": spec.get("output_schema"),
            "allowed_tools": spec.get("allowed_tools", []),
            "allowed_sub_agents": spec.get("allowed_sub_agents", []),
            "memory_window": int(spec.get("memory_window", 6)),
            "max_steps": int(spec.get("max_steps", 8)),
            "max_tool_calls": int(spec.get("max_tool_calls", 6)),
            "max_runtime_seconds": int(spec.get("max_runtime_seconds", 45)),
            "default_context": spec.get("default_context", {}),
            "metadata": spec.get("metadata", {}),
            "is_standard": spec.get("is_standard", False),
        }
