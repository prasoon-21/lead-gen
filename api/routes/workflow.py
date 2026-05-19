from typing import Any, Dict, List, Optional
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

from api.state import get_state


router = APIRouter()
runtime_logger = logging.getLogger("agent.runtime")


@router.get("/list")
async def list_workflows():
    state = get_state()
    if not state.config_loader:
        raise HTTPException(status_code=500, detail="Config loader not initialized")
    return {"workflows": state.config_loader.list_workflow_specs()}


@router.get("/{workflow_id}")
async def get_workflow(workflow_id: str):
    """
    Returns the full workflow spec with each node enriched:
    - agent nodes  → agent spec + resolved system prompt text
    - classification nodes → profile instructions + all dimensions
    - llm / tool / rag nodes → their config as-is
    """
    app_state = get_state()
    loader = app_state.config_loader
    if not loader:
        raise HTTPException(status_code=500, detail="Config loader not initialized")

    try:
        spec = loader.load_workflow_spec(workflow_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Workflow not found: {workflow_id}")

    nodes: List[Dict[str, Any]] = []
    for node in spec.nodes:
        enriched: Dict[str, Any] = {
            "node_id":     node.node_id,
            "node_type":   node.node_type,
            "config":      node.config,
            "input_map":   node.input_map,
            "output_map":  node.output_map,
            "transitions": [{"to": t.to, "when": t.when} for t in node.transitions],
            "details":     {},   # resolved extra info added below
        }

        node_type = node.node_type.strip().lower()

        if node_type == "agent":
            agent_id = (node.config or {}).get("agent_id")
            if agent_id:
                try:
                    agent_spec = loader.load_agent_spec(agent_id)
                    system_prompt = agent_spec.system_prompt or ""
                    if not system_prompt and agent_spec.system_prompt_path:
                        try:
                            system_prompt = loader.load_prompt(agent_spec.system_prompt_path)
                        except Exception:
                            system_prompt = f"(prompt file not found: {agent_spec.system_prompt_path})"
                    enriched["details"] = {
                        "agent_id":      agent_spec.agent_id,
                        "description":   agent_spec.description,
                        "model_profile": agent_spec.model_profile,
                        "allowed_tools": agent_spec.allowed_tools,
                        "max_steps":     agent_spec.max_steps,
                        "max_tool_calls": agent_spec.max_tool_calls,
                        "response_mode": agent_spec.response_mode,
                        "system_prompt": system_prompt,
                    }
                except Exception as e:
                    enriched["details"] = {"error": str(e)}

        elif node_type == "classification":
            profile_id = (node.config or {}).get("profile_id")
            if profile_id:
                try:
                    profile = loader.load_classification_profile(profile_id)
                    enriched["details"] = {
                        "profile_id":   profile.profile_id,
                        "description":  profile.description,
                        "instructions": profile.instructions,
                        "dimensions": [
                            {
                                "name":           d.name,
                                "description":    d.description,
                                "field_type":     d.field_type,
                                "allowed_values": d.allowed_values,
                                "required":       d.required,
                                "allow_null":     d.allow_null,
                            }
                            for d in profile.dimensions
                        ],
                    }
                except Exception as e:
                    enriched["details"] = {"error": str(e)}

        elif node_type == "llm":
            enriched["details"] = {
                "system_prompt": (node.config or {}).get("system_prompt", ""),
                "temperature":   (node.config or {}).get("temperature", 0.7),
                "max_tokens":    (node.config or {}).get("max_tokens", 1024),
            }

        elif node_type == "tool":
            enriched["details"] = {
                "tool_name": (node.config or {}).get("tool_name", ""),
                "arguments": (node.config or {}).get("arguments", {}),
            }

        elif node_type == "rag":
            enriched["details"] = {
                "system_prompt":    (node.config or {}).get("system_prompt", ""),
                "use_retrieval":    (node.config or {}).get("use_retrieval", True),
                "collection_name":  (node.config or {}).get("collection_name", ""),
                "max_chunks":       (node.config or {}).get("max_chunks", 5),
                "rewrite_query":    (node.config or {}).get("rewrite_query", True),
            }

        nodes.append(enriched)

    return {
        "workflow_id": spec.workflow_id,
        "version":     spec.version,
        "description": spec.description,
        "start_at":    spec.start_at,
        "nodes":       nodes,
    }


class WorkflowRunRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    workflow_id: str
    payload: Dict[str, Any] = {}
    session_id: Optional[str] = None


class WorkflowRunResponse(BaseModel):
    success: bool = True
    workflow_id: str
    trace_id: str
    session_id: str
    state: Dict[str, Any]
    metadata: Dict[str, Any] = {}


@router.post("/run", response_model=WorkflowRunResponse)
async def run_workflow(request: WorkflowRunRequest):
    state = get_state()
    if not state.workflow_engine:
        raise HTTPException(status_code=500, detail="Workflow engine not initialized")
    runtime_logger.info(
        "workflow_api_call",
        extra={
            "event": "workflow_api_call",
            "trace_id": "-",
            "session_id": request.session_id or "-",
            "agent_id": "-",
            "workflow_id": request.workflow_id,
            "step": 0,
            "tool_name": "-",
            "status": "received",
        },
    )

    try:
        result = await state.workflow_engine.run(
            workflow_id=request.workflow_id,
            payload=request.payload,
            session_id=request.session_id,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail=f"Unknown workflow_id: {request.workflow_id}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return WorkflowRunResponse(
        success=result.get("success", True),
        workflow_id=result.get("workflow_id"),
        trace_id=result.get("trace_id"),
        session_id=result.get("session_id"),
        state=result.get("state", {}),
        metadata=result.get("metadata", {}),
    )
