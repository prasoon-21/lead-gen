"""
Shared application state — one singleton, injected at startup.

All Firestore stores are scoped to businesses/{FIRESTORE_BUSINESS_ID}/.
If Firestore is unavailable at startup, the stores are None and the system
falls back to file-based YAML configuration.
"""
from typing import Optional

from core.adapters.base import BaseLLMAdapter
from core.retrieval.qdrant_client import QdrantRetriever
from core.logging.tracker import TokenTracker
from config.loader import ConfigLoader
from core.agents.factory import AgentFactory
from core.orchestration.engine import WorkflowEngine
from core.orchestration.node_registry import NodeRegistry
from core.tools.registry import ToolRegistry
from core.services.todo_sheet_store import TodoSheetStore


class AppState:
    # ── Core services ─────────────────────────────────────────────────────────
    adapter: Optional[BaseLLMAdapter] = None
    retriever: Optional[QdrantRetriever] = None
    token_tracker: Optional[TokenTracker] = None
    config_loader: Optional[ConfigLoader] = None
    tool_registry: Optional[ToolRegistry] = None
    agent_factory: Optional[AgentFactory] = None
    workflow_engine: Optional[WorkflowEngine] = None
    node_registry: Optional[NodeRegistry] = None
    todo_store: Optional[TodoSheetStore] = None

    # ── Firestore stores (None if Firebase not configured) ────────────────────
    # businesses/{FIRESTORE_BUSINESS_ID}/agents
    agent_store: Optional[object] = None
    # businesses/{FIRESTORE_BUSINESS_ID}/workflows
    workflow_store: Optional[object] = None
    # businesses/{FIRESTORE_BUSINESS_ID}/remote_tools
    tool_store: Optional[object] = None
    # businesses/{FIRESTORE_BUSINESS_ID}/agent_directory
    directory_store: Optional[object] = None

    # Local in-process Velit multi-location batch scheduler
    velit_batch_scheduler: Optional[object] = None


state = AppState()


def get_state() -> AppState:
    return state
