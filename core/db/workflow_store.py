"""
WorkflowStore — CRUD for workflow definitions in Firestore.

Firestore path: businesses/{business_id}/workflows/{workflow_id}

A workflow document stores the full DAG: nodes list with input_map,
output_map, and transitions — identical structure to the YAML spec so
the WorkflowEngine can deserialise it without changes.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.db.firestore_client import FirestoreClient

COLLECTION = "workflows"


class WorkflowStore:
    def __init__(self, client: FirestoreClient):
        self._client = client

    # ── Read ──────────────────────────────────────────────────────────────────

    async def get(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Return workflow definition dict or None if not found."""
        return await self._client.get(COLLECTION, workflow_id)

    async def list_all(self) -> List[Dict[str, Any]]:
        return await self._client.list_all(COLLECTION)

    async def exists(self, workflow_id: str) -> bool:
        return await self._client.exists(COLLECTION, workflow_id)

    # ── Write ─────────────────────────────────────────────────────────────────

    async def save(self, workflow_id: str, data: Dict[str, Any]) -> None:
        """Create or fully replace a workflow definition."""
        payload = {**data, "workflow_id": workflow_id}
        payload.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        await self._client.set(COLLECTION, workflow_id, payload)

    async def delete(self, workflow_id: str) -> None:
        await self._client.delete(COLLECTION, workflow_id)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def from_yaml_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a WorkflowSpec dict (loaded from YAML) into a Firestore document."""
        nodes = []
        for node in spec.get("nodes", []):
            transitions = []
            for t in node.get("transitions", []):
                if isinstance(t, dict):
                    transitions.append({"to": t.get("to", ""), "when": t.get("when", "always")})
                else:
                    transitions.append({"to": getattr(t, "to", ""), "when": getattr(t, "when", "always")})

            nodes.append({
                "node_id": node.get("node_id", ""),
                "node_type": node.get("node_type", ""),
                "config": node.get("config", {}),
                "input_map": node.get("input_map", {}),
                "output_map": node.get("output_map", {}),
                "transitions": transitions,
            })

        return {
            "workflow_id": spec.get("workflow_id", ""),
            "version": spec.get("version", "1.0"),
            "description": spec.get("description", ""),
            "start_at": spec.get("start_at", ""),
            "nodes": nodes,
            "metadata": spec.get("metadata", {}),
            "is_standard": spec.get("is_standard", False),
        }
