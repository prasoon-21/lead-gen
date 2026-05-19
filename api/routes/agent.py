import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from api.state import get_state
from core.agents.agent_runner import AgentRunner
from core.agents.memory import MemoryStore
from core.adapters.router import build_adapter_from_env
from config.schemas import ModelProfile, FilterConfig


router = APIRouter()
_memory_store = MemoryStore()


class AgentChatRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())

    message: str
    session_id: Optional[str] = None
    context: Dict[str, Any] = {}
    response_mode: str = "text"  # text | json
    schema_: Optional[str] = Field(default=None, alias="schema")
    system_prompt: Optional[str] = None
    system_prompt_path: Optional[str] = None
    model_profile: Optional[str] = None
    workflow_name: Optional[str] = None
    collection_name: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None
    use_retrieval: bool = True
    rewrite_query: bool = True
    max_chunks: int = 5
    memory_window: int = 6
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class AgentChatResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    success: bool = True
    session_id: str
    response_mode: str
    text: Optional[str] = None
    json_: Optional[Dict[str, Any]] = Field(default=None, alias="json", serialization_alias="json")
    metadata: Dict[str, Any] = {}


def _get_model_profile(state, profile_name: Optional[str]) -> Optional[ModelProfile]:
    if not profile_name:
        return None
    profiles = state.config_loader.load_model_profiles()
    profile = profiles.get(profile_name)
    if not profile:
        raise HTTPException(status_code=400, detail=f"Unknown model_profile: {profile_name}")
    return profile


def _get_adapter_for_profile(state, profile: Optional[ModelProfile]):
    if not profile:
        return state.adapter
    adapter = build_adapter_from_env(
        provider_override=profile.provider,
        model_override=profile.model,
        embedding_model_override=profile.embedding_model,
    )
    if not adapter:
        raise HTTPException(
            status_code=500,
            detail=f"LLM adapter not configured for provider: {profile.provider}",
        )
    return adapter


def _resolve_vector_store(
    state,
    workflow_name: Optional[str],
    collection_name: Optional[str],
    filters: Optional[Dict[str, Any]],
    context: Dict[str, Any],
) -> tuple[str, Optional[FilterConfig]]:
    workflow = None
    if workflow_name:
        try:
            workflow = state.config_loader.load_workflow(workflow_name)
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail=f"Unknown workflow: {workflow_name}")

    resolved_collection = collection_name
    if not resolved_collection and workflow and workflow.vector_store:
        resolved_collection = workflow.vector_store.collection

    if not resolved_collection:
        raise HTTPException(status_code=400, detail="collection_name is required for RAG retrieval")

    filter_config = None
    if filters:
        filter_config = state.config_loader.parse_filter_config(filters)
    elif workflow and workflow.vector_store and workflow.vector_store.filters:
        filter_config = workflow.vector_store.filters

    if filter_config:
        filter_config = state.config_loader.resolve_filter_placeholders(filter_config, context)

    return resolved_collection, filter_config


@router.post("/chat", response_model=AgentChatResponse)
async def agent_chat(request: AgentChatRequest):
    state = get_state()
    if not state.adapter:
        raise HTTPException(status_code=500, detail="LLM adapter not initialized")

    response_mode = request.response_mode.lower().strip()
    if response_mode not in {"text", "json"}:
        raise HTTPException(status_code=400, detail="response_mode must be text or json")
    if response_mode == "json" and not request.schema_:
        raise HTTPException(status_code=400, detail="schema is required for response_mode=json")

    session_id = request.session_id or f"SESS_{uuid.uuid4()}"

    profile = _get_model_profile(state, request.model_profile)
    adapter = _get_adapter_for_profile(state, profile)
    if not adapter:
        raise HTTPException(status_code=500, detail="LLM adapter not initialized")

    system_prompt = request.system_prompt
    if not system_prompt and request.system_prompt_path:
        system_prompt = state.config_loader.load_prompt(request.system_prompt_path)

    runner = AgentRunner(
        llm=adapter,
        retriever=state.retriever,
        memory_store=_memory_store,
        memory_window=request.memory_window,
    )

    try:
        collection_name = None
        filter_config = None
        if request.use_retrieval:
            collection_name, filter_config = _resolve_vector_store(
                state,
                request.workflow_name,
                request.collection_name,
                request.filters,
                request.context,
            )
        result = await runner.run(
            session_id=session_id,
            message=request.message,
            context=request.context,
            system_prompt=system_prompt,
            response_mode=response_mode,
            schema=request.schema_,
            use_retrieval=request.use_retrieval,
            rewrite_query=request.rewrite_query,
            max_chunks=request.max_chunks,
            response_temperature=request.temperature if request.temperature is not None else (profile.temperature if profile else 0.7),
            response_max_tokens=request.max_tokens if request.max_tokens is not None else (profile.max_tokens if profile else 4096),
            collection_name=collection_name,
            filter_config=filter_config,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return AgentChatResponse(
        success=True,
        session_id=session_id,
        response_mode=response_mode,
        text=result.get("text"),
        json_=result.get("json"),
        metadata=result.get("metadata", {}),
    )
