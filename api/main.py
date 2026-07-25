import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from dotenv import load_dotenv

from api.auth import api_key_middleware
from api.request_logging import request_logger, setup_request_logging
from api.state import state
from api.routes import (
    health, learning, image, blocks, agent,
    campus_learning_agent, agent_kernel, workflow,
    admin, agent_directory, studio, leads, email_verification,
    velit_batch_scheduler,
)
from core.adapters.router import build_adapter_from_env
from core.retrieval.embeddings import EmbeddingService
from core.logging.tracker import TokenTracker
from core.logging.runtime_logger import setup_runtime_logging
from config.loader import ConfigLoader
from core.tools.defaults import build_default_tool_registry
from core.agents.factory import AgentFactory
from core.orchestration.engine import WorkflowEngine
from core.orchestration.node_registry import build_default_node_registry
from core.services.todo_sheet_store import TodoSheetStore
from core.services.velit.batch_scheduler import VelitBatchScheduler

load_dotenv()
setup_runtime_logging()


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _production_mode() -> bool:
    return (os.getenv("APP_ENV") or os.getenv("ENVIRONMENT") or "").strip().lower() in {"prod", "production"}


def _parse_allowed_origins(raw_origins: str | None) -> list[str]:
    origins = [origin.strip() for origin in (raw_origins or "").split(",") if origin.strip()]
    if origins:
        return origins
    if _production_mode():
        public_base_url = (os.getenv("PUBLIC_BASE_URL") or "").strip().rstrip("/")
        railway_domain = (os.getenv("RAILWAY_PUBLIC_DOMAIN") or "").strip()
        production_origins = []
        if public_base_url:
            production_origins.append(public_base_url)
        if railway_domain:
            production_origins.append(f"https://{railway_domain}")
        return production_origins
    return ["*"]


def _include_optional_route(feature_name: str, default: bool = False) -> bool:
    return _env_bool(feature_name, default)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── LLM adapter ──────────────────────────────────────────────────────────
    state.adapter = build_adapter_from_env()
    if not state.adapter:
        print("WARNING: No LLM adapter configured in environment")

    # ── Qdrant vector store ───────────────────────────────────────────────────
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    collection_name = os.getenv("QDRANT_COLLECTION", "aurika_campus_content")
    embedding_service = EmbeddingService(state.adapter) if state.adapter else None

    try:
        if embedding_service:
            from core.retrieval.qdrant_client import QdrantRetriever  # noqa: lazy import
            state.retriever = QdrantRetriever(
                url=qdrant_url,
                api_key=qdrant_api_key,
                collection_name=collection_name,
                embedding_service=embedding_service,
            )
            print(f"Connected to Qdrant: {qdrant_url}")
        else:
            state.retriever = None
            print("Qdrant disabled (no embedding service available)")
    except Exception as e:
        print(f"Qdrant connection failed (may be expected on Vercel): {e}")
        state.retriever = None

    # ── Logging ───────────────────────────────────────────────────────────────
    state.token_tracker = TokenTracker(
        output_path=os.getenv("TOKEN_LOG_PATH", "logs/token_usage.csv"),
        jsonl_path=os.getenv("JSONL_LOG_PATH", "logs/tutor_logs.jsonl"),
        steps_jsonl_path=os.getenv("AGENT_STEPS_LOG_PATH", "logs/agent_steps.jsonl"),
    )

    # ── Config loader (file-based YAML — DB preload happens after Firestore init)
    state.config_loader = ConfigLoader(
        workflows_dir=os.getenv("WORKFLOWS_DIR", "config/workflow_settings"),
        agents_dir=os.getenv("AGENTS_DIR", "config/agents"),
        workflow_specs_dir=os.getenv("WORKFLOW_SPECS_DIR", "config/workflows"),
        workflow_profiles_dir=os.getenv("WORKFLOW_PROFILES_DIR", "config/workflow_profiles"),
    )

    # ── Firestore — single database for everything ────────────────────────────
    try:
        from core.db.firestore_client import FirestoreClient
        from core.db.agent_store import AgentStore
        from core.db.workflow_store import WorkflowStore
        from core.db.tool_store import ToolStore
        from core.db.directory_store import DirectoryStore

        business_id = os.getenv("FIRESTORE_BUSINESS_ID", "aurika-agentic-core")
        fs_client = FirestoreClient(business_id=business_id)

        state.agent_store = AgentStore(fs_client)
        state.workflow_store = WorkflowStore(fs_client)
        state.tool_store = ToolStore(fs_client)
        state.directory_store = DirectoryStore(fs_client)

        # Inject DB stores into ConfigLoader so Firestore records are served
        # from the in-memory cache (same sync API, no other code changes needed)
        state.config_loader.set_db_stores(state.agent_store, state.workflow_store)
        await state.config_loader.preload_from_db()

        print(f"Firestore connected — business: {business_id}")
    except Exception as exc:
        print(f"Firestore init failed — running with file-based config only: {exc}")
        state.agent_store = None
        state.workflow_store = None
        state.tool_store = None
        state.directory_store = None

    # ── Google Sheets (todo store) ────────────────────────────────────────────
    try:
        state.todo_store = TodoSheetStore(
            sheet_id=os.getenv("TODO_SHEET_ID"),
            sheet_name=os.getenv("TODO_SHEET_FILE_NAME", "Aurika Agent Todo"),
            folder_id=os.getenv("GOOGLE_DRIVE_FOLDER_ID"),
            tasks_worksheet_name=os.getenv("TODO_WORKSHEET_NAME", "tasks"),
            audit_worksheet_name=os.getenv("TODO_AUDIT_WORKSHEET", "audit_log"),
        )
        # LEADS_WORKSHEET_NAME is read inside TodoSheetStore via os.getenv automatically
    except Exception as exc:
        print(f"Todo store initialization failed: {exc}")
        state.todo_store = None

    # ── Tool registry (local + remote tools from files + Firestore) ───────────
    state.tool_registry = build_default_tool_registry()
    if state.tool_store:
        from core.tools.remote_loader import load_remote_tools_from_db
        for tool in await load_remote_tools_from_db(state.tool_store):
            state.tool_registry.register(tool)

    # ── Agent factory + orchestration engine ──────────────────────────────────
    state.agent_factory = AgentFactory(
        config_loader=state.config_loader,
        base_adapter=state.adapter,
        retriever=state.retriever,
        token_tracker=state.token_tracker,
        tool_registry=state.tool_registry,
        todo_store=state.todo_store,
    )
    state.node_registry = build_default_node_registry(resources={})
    state.workflow_engine = WorkflowEngine(
        config_loader=state.config_loader,
        agent_factory=state.agent_factory,
        node_registry=state.node_registry,
        token_tracker=state.token_tracker,
    )

    # Local Velit multi-location queue scheduler. It runs only while this
    # FastAPI process is alive and persists queue state to data/*.json.
    state.velit_batch_scheduler = VelitBatchScheduler()
    await state.velit_batch_scheduler.start_runtime()

    print("Agentic Core initialised")
    yield
    if state.velit_batch_scheduler:
        await state.velit_batch_scheduler.stop_runtime()
    print("Agentic Core shutting down")


app = FastAPI(
    title="Agentic Core",
    description="Unified AI Microservice for Aurika",
    version="1.0.0",
    lifespan=lifespan,
)

setup_request_logging()

allowed_origins = _parse_allowed_origins(os.getenv("ALLOWED_ORIGINS"))
allow_credentials = "*" not in allowed_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.middleware("http")(request_logger)
app.middleware("http")(api_key_middleware)


@app.get("/", include_in_schema=False)
async def dashboard():
    return FileResponse("ui/dashboard.html")


@app.get("/studio", include_in_schema=False)
async def studio_page():
    return FileResponse("ui/studio.html")


@app.get("/leads", include_in_schema=False)
async def leads_page():
    return FileResponse("ui/leadgen.html")


@app.get("/email-verifier", include_in_schema=False)
async def email_verifier_page():
    return FileResponse("ui/email_verifier.html")


@app.get("/velit-batch-scheduler", include_in_schema=False)
async def velit_batch_scheduler_page():
    return FileResponse("ui/velit_batch_scheduler.html")


@app.get("/hunter-search", include_in_schema=False)  # TEST FEATURE
async def hunter_search_page():
    if not _include_optional_route("ENABLE_HUNTER_TEST_PIPELINE", True):
        raise HTTPException(status_code=404, detail="Hunter test pipeline is disabled")
    return FileResponse("ui/hunter_search.html")


if _include_optional_route("ENABLE_COMPANY_CHAT_INTEL", False):
    @app.get("/company-chat", include_in_schema=False)
    async def company_chat_page():
        return FileResponse("ui/company_chat.html")


app.include_router(health.router,                                              tags=["Health"])
app.include_router(learning.router,          prefix="/api/learning/chat",     tags=["Learning"])
app.include_router(image.router,             prefix="/api/image",             tags=["Image"])
app.include_router(blocks.router,            prefix="/api/blocks",            tags=["Blocks"])
app.include_router(agent.router,             prefix="/api/agent",             tags=["Agent"])
app.include_router(agent_kernel.router,      prefix="/api/agent",             tags=["Agent Kernel"])
app.include_router(campus_learning_agent.router, prefix="/api/campus",        tags=["Campus"])
app.include_router(workflow.router,          prefix="/api/workflow",          tags=["Workflow"])
app.include_router(admin.router,             prefix="/api/admin",             tags=["Admin"])
app.include_router(agent_directory.router,   prefix="/api/directory",         tags=["Agent Directory"])
app.include_router(studio.router,            prefix="/api/studio",            tags=["Studio"])
app.include_router(leads.router,             prefix="/api/leads",             tags=["Leads"])
app.include_router(email_verification.router, prefix="/api/email-verification", tags=["Email Verification"])
app.include_router(velit_batch_scheduler.router, prefix="/api/velit-batch", tags=["Velit Batch Scheduler"])

if _include_optional_route("ENABLE_HUNTER_TEST_PIPELINE", True):
    try:
        from api.routes._test_features import hunter_search as hunter_search_route  # noqa: WPS433

        app.include_router(hunter_search_route.router, prefix="/api/hunter", tags=["Hunter Search (Test)"])
    except Exception as exc:
        print(f"Hunter test pipeline disabled because it could not be imported: {exc}")

if _include_optional_route("ENABLE_COMPANY_CHAT_INTEL", False):
    try:
        from api.routes import company_intel as company_intel_route  # noqa: WPS433

        app.include_router(company_intel_route.router, prefix="/api/company-intel", tags=["Company Intel"])
    except Exception as exc:
        print(f"Company intel chat disabled because it could not be imported: {exc}")

# ── Removable Vento Influencer Lead Pipeline ─────────────────────────────────
if _include_optional_route("ENABLE_VENTO_PIPELINE", True):
    try:
        from api.routes import vento_leads as vento_leads_route  # noqa: WPS433

        app.include_router(vento_leads_route.router, prefix="/api/vento", tags=["Vento Leads"])
    except Exception as exc:
        print(f"Vento pipeline disabled because it could not be imported: {exc}")
