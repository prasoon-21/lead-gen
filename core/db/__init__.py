"""
core.db — Firestore-backed persistence layer for Agentic Core.

All records are scoped under:
    businesses/{FIRESTORE_BUSINESS_ID}/
        agents/          → AgentStore
        workflows/       → WorkflowStore
        remote_tools/    → ToolStore

Usage:
    from core.db import FirestoreClient, AgentStore, WorkflowStore, ToolStore
"""

from core.db.firestore_client import FirestoreClient
from core.db.agent_store import AgentStore
from core.db.workflow_store import WorkflowStore
from core.db.tool_store import ToolStore
from core.db.directory_store import DirectoryStore

__all__ = ["FirestoreClient", "AgentStore", "WorkflowStore", "ToolStore", "DirectoryStore"]
