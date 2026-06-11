import os
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, ConfigDict, Field

from api.schemas import DocumentPayload
from api.state import get_state
from core.adapters.router import build_adapter_from_env
from config.schemas import ModelProfile, FilterConfig
from core.agents.rag_agent import RAGAgent
from core.extraction.json_extractor import JSONExtractor
from core.extraction.image_metadata_extractor import ImageMetadataExtractor
from core.services.file_analyzer import FileAnalyzer


router = APIRouter()


class LLMRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = None
    system_prompt_path: Optional[str] = None
    model_profile: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class LLMResponse(BaseModel):
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


class RAGRequest(BaseModel):
    query: str
    context: Dict[str, Any] = {}
    system_prompt: Optional[str] = None
    system_prompt_path: Optional[str] = None
    model_profile: Optional[str] = None
    workflow_name: Optional[str] = None
    collection_name: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None
    use_retrieval: bool = True
    rewrite_query: bool = True
    include_retrieval: bool = False
    max_chunks: int = 5
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class RAGResponse(BaseModel):
    text: str
    search_query: str
    retrieval_used: bool
    retrieval_count: int
    input_tokens: int
    output_tokens: int
    latency_ms: float
    retrieval_results: Optional[list] = None


class JSONExtractRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    text: str
    schema_: str = Field(alias="schema")
    system_prompt: Optional[str] = None
    model_profile: Optional[str] = None


class ImageMetadataRequest(BaseModel):
    text: str
    topic: Optional[str] = None
    level: Optional[str] = None
    board: Optional[str] = None
    model_profile: Optional[str] = None


class FileAnalyzeRequest(BaseModel):
    document: DocumentPayload
    instruction: str
    user_query: Optional[str] = None
    model_profile: Optional[str] = None


class FileAnalyzeResponse(BaseModel):
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


class TranscribeResponse(BaseModel):
    text: str
    model: str
    language: Optional[str] = None


def _sanitize_api_key(value: str) -> str:
    cleaned = value.strip()
    cleaned = cleaned.encode("ascii", "ignore").decode("ascii")
    return cleaned


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


@router.post("/llm", response_model=LLMResponse)
async def run_llm(request: LLMRequest):
    state = get_state()
    profile = _get_model_profile(state, request.model_profile)
    adapter = _get_adapter_for_profile(state, profile)
    if not adapter:
        raise HTTPException(status_code=500, detail="LLM adapter not initialized")
    system_prompt = request.system_prompt
    if not system_prompt and request.system_prompt_path:
        system_prompt = state.config_loader.load_prompt(request.system_prompt_path)
    response = await adapter.generate(
        prompt=request.prompt,
        system_prompt=system_prompt,
        temperature=request.temperature if request.temperature is not None else (profile.temperature if profile else 0.7),
        max_tokens=request.max_tokens if request.max_tokens is not None else (profile.max_tokens if profile else 1024),
    )
    return LLMResponse(
        text=response.text,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        latency_ms=response.latency_ms,
    )


@router.post("/rag", response_model=RAGResponse)
async def run_rag(request: RAGRequest):
    state = get_state()
    profile = _get_model_profile(state, request.model_profile)
    adapter = _get_adapter_for_profile(state, profile)
    if not adapter:
        raise HTTPException(status_code=500, detail="LLM adapter not initialized")
    system_prompt = request.system_prompt
    if not system_prompt and request.system_prompt_path:
        system_prompt = state.config_loader.load_prompt(request.system_prompt_path)
    agent = RAGAgent(
        llm=adapter,
        retriever=state.retriever,
        rewrite_query=request.rewrite_query,
        max_chunks=request.max_chunks,
        response_temperature=request.temperature if request.temperature is not None else (profile.temperature if profile else 0.7),
        response_max_tokens=request.max_tokens if request.max_tokens is not None else (profile.max_tokens if profile else 4096),
    )
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
    result = await agent.run(
        query=request.query,
        context=request.context,
        system_prompt=system_prompt,
        use_retrieval=request.use_retrieval,
        include_retrieval=request.include_retrieval,
        collection_name=collection_name,
        filter_config=filter_config,
    )
    return RAGResponse(**result)


@router.post("/extract/json")
async def extract_json(request: JSONExtractRequest):
    state = get_state()
    profile = _get_model_profile(state, request.model_profile)
    adapter = _get_adapter_for_profile(state, profile)
    if not adapter:
        raise HTTPException(status_code=500, detail="LLM adapter not initialized")
    extractor = JSONExtractor(adapter)
    return await extractor.extract(
        text=request.text,
        schema_description=request.schema_,
        system_prompt=request.system_prompt,
    )


@router.post("/extract/image-metadata")
async def extract_image_metadata(request: ImageMetadataRequest):
    state = get_state()
    profile = _get_model_profile(state, request.model_profile)
    adapter = _get_adapter_for_profile(state, profile)
    if not adapter:
        raise HTTPException(status_code=500, detail="LLM adapter not initialized")
    extractor = ImageMetadataExtractor(adapter)
    return await extractor.extract(
        text=request.text,
        topic=request.topic,
        level=request.level,
        board=request.board,
    )


@router.post("/file/analyze", response_model=FileAnalyzeResponse)
async def analyze_file(request: FileAnalyzeRequest):
    state = get_state()
    profile = _get_model_profile(state, request.model_profile)
    adapter = _get_adapter_for_profile(state, profile)
    if not adapter:
        raise HTTPException(status_code=500, detail="LLM adapter not initialized")
    analyzer = FileAnalyzer(adapter)
    response = await analyzer.analyze(
        document=request.document.model_dump(),
        instruction=request.instruction,
        user_query=request.user_query,
    )
    return FileAnalyzeResponse(
        text=response.text,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        latency_ms=response.latency_ms,
    )

@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    provider: Optional[str] = Form(None),
):
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    use_provider = (provider or "gemini").strip().lower()
    google_api_key = os.getenv("GOOGLE_API_KEY")

    if use_provider != "gemini":
        raise HTTPException(
            status_code=400,
            detail="Only the Gemini transcription provider is enabled for this project.",
        )

    if google_api_key:
        try:
            from google import genai as _genai
            from google.genai import types as _genai_types

            _client = _genai.Client(api_key=google_api_key)
            mime_type = file.content_type or "audio/mpeg"
            model_name = model or os.getenv("GEMINI_TRANSCRIPTION_MODEL", "gemini-2.5-flash")

            transcribe_prompt = "Transcribe this audio file accurately. Return only the transcription text, no additional commentary."
            if language:
                transcribe_prompt += f" The audio is in {language}."
            if prompt:
                transcribe_prompt += f" Context: {prompt}"

            response = _client.models.generate_content(
                model=model_name,
                contents=[
                    transcribe_prompt,
                    _genai_types.Part(
                        inline_data=_genai_types.Blob(
                            mime_type=mime_type,
                            data=file_bytes,
                        )
                    )
                ]
            )

            text = response.text.strip() if response.text else ""
            if not text:
                raise HTTPException(status_code=500, detail="Gemini returned empty transcription")

            return TranscribeResponse(text=text, model=model_name, language=language)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Gemini transcription failed: {str(e)}")

    raise HTTPException(status_code=500, detail="No Gemini transcription provider configured. Set GOOGLE_API_KEY.")

