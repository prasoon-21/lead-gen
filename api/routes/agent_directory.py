"""
Agent Directory — CRUD + execution endpoints (Firestore-backed).

Agents are stored in Firestore under:
    businesses/{FIRESTORE_BUSINESS_ID}/agent_directory/{agent_id}

Scopes:
    generic          — usable by any business / tenant
    business         — bound to a specific business_id
    business_tenant  — bound to business_id + tenant_id

Endpoints:
    POST   /api/directory/agents                    register new agent
    GET    /api/directory/agents                    list (filterable)
    GET    /api/directory/agents/{agent_id}         get by ID
    GET    /api/directory/agents/slug/{slug}        get by slug
    PUT    /api/directory/agents/{agent_id}         update config
    PATCH  /api/directory/agents/{agent_id}/status  activate / deactivate
    DELETE /api/directory/agents/{agent_id}         delete
    POST   /api/directory/agents/{agent_id}/run     execute an active agent
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, model_validator

from api.state import get_state
from config.schemas import AgentSpec

router = APIRouter()

VALID_SCOPES = {"generic", "business", "business_tenant"}
VALID_STATUSES = {"draft", "active", "inactive"}


# ── Request / Response schemas ────────────────────────────────────────────────

class AgentCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    slug: Optional[str] = Field(None, max_length=128)
    description: Optional[str] = None
    scope: str = Field("generic")
    business_id: Optional[str] = None
    tenant_id: Optional[str] = None

    welcome_message: Optional[str] = None
    placeholder_text: Optional[str] = Field(None, max_length=256)
    tags: List[str] = []

    model_profile: Optional[str] = None
    system_prompt: Optional[str] = None
    developer_prompt: Optional[str] = None
    response_mode: str = Field("text", pattern="^(text|json)$")
    allowed_tools: List[str] = []
    default_context: Dict[str, Any] = {}
    output_schema: Optional[Dict[str, Any]] = None

    memory_window: int = Field(6, ge=1, le=50)
    max_steps: int = Field(8, ge=1, le=50)
    max_tool_calls: int = Field(6, ge=0, le=50)
    max_runtime_seconds: int = Field(45, ge=5, le=300)

    created_by: Optional[str] = None
    metadata: Dict[str, Any] = {}
    version: str = Field("1.0", max_length=16)

    @model_validator(mode="after")
    def _validate_scope(self) -> "AgentCreateRequest":
        if self.scope not in VALID_SCOPES:
            raise ValueError(f"scope must be one of {VALID_SCOPES}")
        if self.scope in {"business", "business_tenant"} and not self.business_id:
            raise ValueError("business_id is required for scope 'business' or 'business_tenant'")
        if self.scope == "business_tenant" and not self.tenant_id:
            raise ValueError("tenant_id is required for scope 'business_tenant'")
        return self


class AgentUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    scope: Optional[str] = None
    business_id: Optional[str] = None
    tenant_id: Optional[str] = None
    welcome_message: Optional[str] = None
    placeholder_text: Optional[str] = None
    tags: Optional[List[str]] = None
    model_profile: Optional[str] = None
    system_prompt: Optional[str] = None
    developer_prompt: Optional[str] = None
    response_mode: Optional[str] = None
    allowed_tools: Optional[List[str]] = None
    default_context: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    memory_window: Optional[int] = None
    max_steps: Optional[int] = None
    max_tool_calls: Optional[int] = None
    max_runtime_seconds: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    version: Optional[str] = None


class AgentStatusUpdate(BaseModel):
    status: str

    @model_validator(mode="after")
    def _validate(self) -> "AgentStatusUpdate":
        if self.status not in VALID_STATUSES:
            raise ValueError(f"status must be one of {VALID_STATUSES}")
        return self


class AgentRunRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: Optional[str] = None
    context: Dict[str, Any] = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_directory_store():
    state = get_state()
    if not getattr(state, "directory_store", None):
        raise HTTPException(
            status_code=503,
            detail="Firestore not initialised. Check Firebase credentials in .env",
        )
    return state.directory_store


def _record_to_spec(record: Dict[str, Any]) -> AgentSpec:
    return AgentSpec(
        agent_id=record["agent_id"],
        version=record.get("version", "1.0"),
        description=record.get("description") or "",
        model_profile=record.get("model_profile"),
        system_prompt=record.get("system_prompt") or "",
        developer_prompt=record.get("developer_prompt") or "",
        response_mode=record.get("response_mode", "text"),
        output_schema=record.get("output_schema"),
        allowed_tools=record.get("allowed_tools") or [],
        memory_window=record.get("memory_window", 6),
        max_steps=record.get("max_steps", 8),
        max_tool_calls=record.get("max_tool_calls", 6),
        max_runtime_seconds=record.get("max_runtime_seconds", 45),
        default_context=record.get("default_context") or {},
        metadata=record.get("metadata") or {},
    )


# ── CRUD endpoints ────────────────────────────────────────────────────────────

@router.post("/agents", status_code=201, summary="Register a new agent")
async def create_agent(body: AgentCreateRequest) -> Dict[str, Any]:
    store = _get_directory_store()
    data = body.model_dump()
    return await store.create(data)


@router.get("/agents", summary="List agents")
async def list_agents(
    scope: Optional[str] = Query(None),
    business_id: Optional[str] = Query(None),
    tenant_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> List[Dict[str, Any]]:
    store = _get_directory_store()
    return await store.list_all(
        scope=scope,
        business_id=business_id,
        tenant_id=tenant_id,
        status=status,
        limit=limit,
        offset=offset,
    )


@router.get("/agents/slug/{slug}", summary="Get agent by slug")
async def get_agent_by_slug(slug: str) -> Dict[str, Any]:
    store = _get_directory_store()
    record = await store.get_by_slug(slug)
    if not record:
        raise HTTPException(status_code=404, detail=f"Agent with slug '{slug}' not found")
    return record


@router.get("/agents/{agent_id}", summary="Get agent by ID")
async def get_agent(agent_id: str) -> Dict[str, Any]:
    store = _get_directory_store()
    record = await store.get(agent_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    return record


@router.put("/agents/{agent_id}", summary="Update agent configuration")
async def update_agent(agent_id: str, body: AgentUpdateRequest) -> Dict[str, Any]:
    store = _get_directory_store()
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    record = await store.update(agent_id, updates)
    if not record:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    return record


@router.patch("/agents/{agent_id}/status", summary="Activate, deactivate, or revert to draft")
async def update_status(agent_id: str, body: AgentStatusUpdate) -> Dict[str, Any]:
    store = _get_directory_store()
    record = await store.set_status(agent_id, body.status)
    if not record:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    return record


@router.delete("/agents/{agent_id}", status_code=204, summary="Delete an agent")
async def delete_agent(agent_id: str) -> None:
    store = _get_directory_store()
    if not await store.exists(agent_id):
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    await store.delete(agent_id)


# ── Execution endpoint ────────────────────────────────────────────────────────

@router.post("/agents/{agent_id}/run", summary="Execute an active agent")
async def run_agent(agent_id: str, body: AgentRunRequest) -> Dict[str, Any]:
    """Run any active agent in the directory.

    Pass agent_id in your chat widget so the correct agent executes for
    that page / product — without hardcoding agent logic in the frontend.
    """
    store = _get_directory_store()
    record = await store.get(agent_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    if record.get("status") != "active":
        raise HTTPException(
            status_code=409,
            detail=f"Agent '{agent_id}' is not active (status={record.get('status')}). Activate it first.",
        )

    app_state = get_state()
    if not app_state.agent_factory:
        raise HTTPException(status_code=503, detail="Agent factory not initialised")

    spec = _record_to_spec(record)
    merged_context = {**(record.get("default_context") or {}), **body.context}

    result = await app_state.agent_factory.run_with_spec(
        spec=spec,
        message=body.message,
        session_id=body.session_id,
        context=merged_context,
    )

    return {
        "agent_id": agent_id,
        "session_id": result.get("session_id", ""),
        "trace_id": result.get("trace_id"),
        "text": result.get("text"),
        "json_result": result.get("json"),
        "steps": result.get("steps", 0),
        "metadata": result.get("metadata", {}),
    }
