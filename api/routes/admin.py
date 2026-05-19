"""
Admin API — CRUD for agents, workflows, and remote tools stored in Firestore.

All endpoints are scoped to the business set by FIRESTORE_BUSINESS_ID.

Endpoints:
    Agents:
        GET    /api/admin/agents              list all
        GET    /api/admin/agents/{agent_id}   get one
        POST   /api/admin/agents              create / update
        DELETE /api/admin/agents/{agent_id}   delete

    Workflows:
        GET    /api/admin/workflows
        GET    /api/admin/workflows/{workflow_id}
        POST   /api/admin/workflows
        DELETE /api/admin/workflows/{workflow_id}

    Remote Tools:
        GET    /api/admin/tools
        GET    /api/admin/tools/{tool_name}
        POST   /api/admin/tools
        DELETE /api/admin/tools/{tool_name}

    Utilities:
        POST   /api/admin/seed               seed standard library → Firestore
        POST   /api/admin/reload             reload Firestore records into memory cache
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.state import get_state

router = APIRouter()


# ── Request / Response schemas ────────────────────────────────────────────────

class AgentPayload(BaseModel):
    agent_id: str
    description: str = ""
    system_prompt: str
    model_profile: str = "large"
    response_mode: str = "text"
    allowed_tools: List[str] = []
    allowed_sub_agents: List[str] = []
    memory_window: int = 6
    max_steps: int = 8
    max_tool_calls: int = 6
    max_runtime_seconds: int = 45
    default_context: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}
    is_standard: bool = False


class WorkflowNodePayload(BaseModel):
    node_id: str
    node_type: str
    config: Dict[str, Any] = {}
    input_map: Dict[str, str] = {}
    output_map: Dict[str, str] = {}
    transitions: List[Dict[str, str]] = []


class WorkflowPayload(BaseModel):
    workflow_id: str
    description: str = ""
    start_at: str
    nodes: List[WorkflowNodePayload]
    metadata: Dict[str, Any] = {}
    is_standard: bool = False


class RemoteToolPayload(BaseModel):
    tool_name: str
    description: str = ""
    endpoint_url: str
    endpoint_url_env: Optional[str] = None
    input_schema: Dict[str, Any] = {"type": "object", "properties": {}}
    auth_header_name: Optional[str] = None
    auth_header_env: Optional[str] = None
    auth_header_prefix: str = ""
    timeout_seconds: int = 30
    method: str = "POST"
    request_body_mode: str = "wrapped"
    static_headers: Dict[str, str] = {}
    argument_headers: Dict[str, str] = {}
    omit_arguments_from_body: List[str] = []


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_stores():
    state = get_state()
    if not state.agent_store or not state.workflow_store or not state.tool_store:
        raise HTTPException(
            status_code=503,
            detail="Firestore is not initialised. Check FIREBASE_CREDENTIALS_PATH in .env",
        )
    return state.agent_store, state.workflow_store, state.tool_store


# ── Agent endpoints ───────────────────────────────────────────────────────────

@router.get("/agents", summary="List all agents")
async def list_agents() -> List[Dict[str, Any]]:
    agent_store, _, _ = _get_stores()
    return await agent_store.list_all()


@router.get("/agents/{agent_id}", summary="Get a single agent")
async def get_agent(agent_id: str) -> Dict[str, Any]:
    agent_store, _, _ = _get_stores()
    record = await agent_store.get(agent_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")
    return record


@router.post("/agents", summary="Create or update an agent")
async def save_agent(payload: AgentPayload) -> Dict[str, Any]:
    agent_store, _, _ = _get_stores()
    data = payload.model_dump()
    await agent_store.save(payload.agent_id, data)

    # Invalidate the in-memory config cache so the new version is picked up
    state = get_state()
    state.config_loader._agents_cache.pop(payload.agent_id, None)

    return {"ok": True, "agent_id": payload.agent_id}


@router.delete("/agents/{agent_id}", summary="Delete an agent")
async def delete_agent(agent_id: str) -> Dict[str, Any]:
    agent_store, _, _ = _get_stores()
    await agent_store.delete(agent_id)

    state = get_state()
    state.config_loader._agents_cache.pop(agent_id, None)

    return {"ok": True, "agent_id": agent_id}


# ── Workflow endpoints ────────────────────────────────────────────────────────

@router.get("/workflows", summary="List all workflows")
async def list_workflows() -> List[Dict[str, Any]]:
    _, workflow_store, _ = _get_stores()
    return await workflow_store.list_all()


@router.get("/workflows/{workflow_id}", summary="Get a single workflow")
async def get_workflow(workflow_id: str) -> Dict[str, Any]:
    _, workflow_store, _ = _get_stores()
    record = await workflow_store.get(workflow_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Workflow not found: {workflow_id}")
    return record


@router.post("/workflows", summary="Create or update a workflow")
async def save_workflow(payload: WorkflowPayload) -> Dict[str, Any]:
    _, workflow_store, _ = _get_stores()
    data = payload.model_dump()
    await workflow_store.save(payload.workflow_id, data)

    state = get_state()
    state.config_loader._workflow_specs_cache.pop(payload.workflow_id, None)

    return {"ok": True, "workflow_id": payload.workflow_id}


@router.delete("/workflows/{workflow_id}", summary="Delete a workflow")
async def delete_workflow(workflow_id: str) -> Dict[str, Any]:
    _, workflow_store, _ = _get_stores()
    await workflow_store.delete(workflow_id)

    state = get_state()
    state.config_loader._workflow_specs_cache.pop(workflow_id, None)

    return {"ok": True, "workflow_id": workflow_id}


# ── Remote tool endpoints ─────────────────────────────────────────────────────

@router.get("/tools", summary="List all remote tools")
async def list_tools() -> List[Dict[str, Any]]:
    _, _, tool_store = _get_stores()
    return await tool_store.list_all()


@router.get("/tools/{tool_name}", summary="Get a single remote tool")
async def get_tool(tool_name: str) -> Dict[str, Any]:
    _, _, tool_store = _get_stores()
    record = await tool_store.get(tool_name)
    if not record:
        raise HTTPException(status_code=404, detail=f"Tool not found: {tool_name}")
    return record


@router.post("/tools", summary="Register or update a remote tool")
async def save_tool(payload: RemoteToolPayload) -> Dict[str, Any]:
    _, _, tool_store = _get_stores()
    data = payload.model_dump()
    await tool_store.save(payload.tool_name, data)

    # Dynamically register the new tool in the live ToolRegistry
    state = get_state()
    if state.tool_registry:
        from core.tools.external.remote_tool import RemoteToolWrapper
        tool = RemoteToolWrapper(
            name=payload.tool_name,
            description=payload.description,
            endpoint_url=payload.endpoint_url,
            input_schema=payload.input_schema,
            auth_header_name=payload.auth_header_name,
            auth_header_env=payload.auth_header_env,
            auth_header_prefix=payload.auth_header_prefix,
            timeout_seconds=payload.timeout_seconds,
            method=payload.method,
            request_body_mode=payload.request_body_mode,
            static_headers=payload.static_headers,
            argument_headers=payload.argument_headers,
            omit_arguments_from_body=payload.omit_arguments_from_body,
        )
        state.tool_registry.register(tool)

    return {"ok": True, "tool_name": payload.tool_name}


@router.delete("/tools/{tool_name}", summary="Delete a remote tool")
async def delete_tool(tool_name: str) -> Dict[str, Any]:
    _, _, tool_store = _get_stores()
    await tool_store.delete(tool_name)
    return {"ok": True, "tool_name": tool_name}


# ── Utility endpoints ─────────────────────────────────────────────────────────

@router.post("/seed", summary="Seed standard library (YAML files → Firestore)")
async def seed_standard_library(force: bool = False) -> Dict[str, Any]:
    """Push all YAML agent and workflow specs to Firestore.

    Set force=true to overwrite existing records.
    This is equivalent to running: python scripts/seed_firestore.py [--force]
    """
    agent_store, workflow_store, _ = _get_stores()
    state = get_state()
    loader = state.config_loader

    seeded_agents = []
    seeded_workflows = []
    skipped = []

    # Agents
    for meta in loader.list_agent_specs():
        agent_id = meta["agent_id"]
        try:
            spec = loader.load_agent_spec(agent_id)
            system_prompt = spec.system_prompt or ""
            if not system_prompt and spec.system_prompt_path:
                try:
                    system_prompt = loader.load_prompt(spec.system_prompt_path)
                except FileNotFoundError:
                    pass

            if not force and await agent_store.exists(agent_id):
                skipped.append(agent_id)
                continue

            from core.db.agent_store import AgentStore
            record = AgentStore.from_yaml_spec({
                **spec.__dict__,
                "system_prompt": system_prompt,
                "is_standard": True,
            })
            await agent_store.save(agent_id, record)
            loader._agents_cache.pop(agent_id, None)
            seeded_agents.append(agent_id)
        except Exception as exc:
            skipped.append(f"{agent_id} (error: {exc})")

    # Workflows
    for meta in loader.list_workflow_specs():
        workflow_id = meta["workflow_id"]
        try:
            spec = loader.load_workflow_spec(workflow_id)
            if not force and await workflow_store.exists(workflow_id):
                skipped.append(workflow_id)
                continue

            from core.db.workflow_store import WorkflowStore
            record = WorkflowStore.from_yaml_spec({
                "workflow_id": spec.workflow_id,
                "version": spec.version,
                "description": spec.description,
                "start_at": spec.start_at,
                "nodes": [
                    {
                        "node_id": n.node_id,
                        "node_type": n.node_type,
                        "config": n.config,
                        "input_map": n.input_map,
                        "output_map": n.output_map,
                        "transitions": [{"to": t.to, "when": t.when} for t in n.transitions],
                    }
                    for n in spec.nodes
                ],
                "metadata": spec.metadata,
                "is_standard": True,
            })
            await workflow_store.save(workflow_id, record)
            loader._workflow_specs_cache.pop(workflow_id, None)
            seeded_workflows.append(workflow_id)
        except Exception as exc:
            skipped.append(f"{workflow_id} (error: {exc})")

    return {
        "ok": True,
        "seeded_agents": seeded_agents,
        "seeded_workflows": seeded_workflows,
        "skipped": skipped,
    }


@router.post("/reload", summary="Reload Firestore records into memory cache")
async def reload_from_db() -> Dict[str, Any]:
    """Force a re-read of all agents and workflows from Firestore into the in-memory cache.

    Useful after making direct edits in Firestore console or via the admin API
    from another instance.
    """
    state = get_state()
    state.config_loader.clear_cache()
    await state.config_loader.preload_from_db()
    return {"ok": True, "message": "Cache reloaded from Firestore"}
