# Agentic Core

AI-powered Learning Tutor backend with image generation and Qdrant vector search.

## Features

- 🎓 **AI Learning Tutor** - Personalized tutoring with memory and context
- 🖼️ **Image Generation** - Auto-generate educational diagrams with Gemini
- 📚 **Qdrant Search** - Vector-based content retrieval
- 👤 **Student Profiles** - Google Sheets integration for personalization
- 📦 **R2 Storage** - Image upload and CDN delivery

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
# Edit .env with your API keys

# Run server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/learning/chat/{student_id}` | POST | Chat with tutor |
| `/api/image/generate` | POST | Generate image |
| `/api/blocks/llm` | POST | Run a generic LLM prompt |
| `/api/blocks/rag` | POST | Run RAG agent (query + retrieval) |
| `/api/blocks/extract/json` | POST | Extract JSON from text |
| `/api/blocks/extract/image-metadata` | POST | Extract image metadata JSON |
| `/api/blocks/file/analyze` | POST | Analyze a document with vision |
| `/api/blocks/transcribe` | POST | Transcribe audio (Gemini) |
| `/api/agent/chat` | POST | Generic agent (memory + RAG + JSON output) |
| `/api/agent/run` | POST | Tool-calling agent kernel run by `agent_id`; accepts JSON or multipart file uploads |
| `/api/workflow/run` | POST | Hybrid fixed + flexible workflow run by `workflow_id` |

## Agent Kernel and Workflow Engine

New additive components:

- `config/agents/*.yaml` defines reusable agent specs (`allowed_tools`, limits, prompts, response mode).
- `core/agents/kernel.py` runs decision loop (`tool_call` / `final_answer`) with step limits and tracing.
- `core/tools/*` provides a pluggable tool registry and executor.
- `config/workflows/*.yaml` defines fixed-flow orchestration with agent/tool/llm/rag nodes.
- `config/workflow_profiles/*.yaml` defines data-owned classification dimensions and allowed values.
- `core/orchestration/engine.py` executes workflow transitions.

This does not replace existing blocks/routes; it adds an opt-in orchestration layer.

### Profile-Driven Conversation Classification

The workflow engine now supports a `classification` node type for cases where you do not need a full autonomous agent.

- The workflow points to a profile in `config/workflow_profiles/*.yaml`.
- The profile defines dimensions, allowed values, and instructions.
- The engine builds the JSON schema and classification prompt from the profile.
- Adding a new dimension or enum value becomes a profile change, not an engine change.

Included V1 example:

- Workflow id: `email_auto_ticketing_v1`
- Profile id: `email_auto_ticketing_v1`
- Purpose: classify inbound heater/AC/support emails into structured ticket fields

### Task Manager Agent (Sheets-backed + Aurika Tasks)

Added a first task-management agent:

- Agent id: `task_manager_agent`
- Prompt: `config/prompts/agents/task_manager_agent.txt`
- Workflow id: `task_manager_pipeline`
- Tools: `todo_ensure_store`, `todo_create`, `todo_list`, `todo_update`, `todo_complete`, `aurika_tasks_create_task`

Behavior:
- Reads pending tasks
- Proposes writes first
- Requires explicit confirmation before create/update/complete (`confirmed=true`)
- Can create formal Aurika tasks via `POST https://engage.aurika.ai/api/internal/tasks`

### Live Agent Runtime Logs (Terminal + File)

Runtime tracing is now emitted in real time:

- Logger: `agent.runtime`
- Terminal: structured logs for agent start/end, decisions, tool calls, workflow nodes
- File: `logs/agent_runtime.log` (rotates daily)
- Step JSONL: `logs/agent_steps.jsonl`
- Remote tool debug events: `remote_tool_request`, `remote_tool_response`, `remote_tool_http_error`, `remote_tool_timeout`

Useful environment variables:

- `AGENT_RUNTIME_LOG_PATH` (default `logs/agent_runtime.log`)
- `AGENT_STEPS_LOG_PATH` (default `logs/agent_steps.jsonl`)

### Chat Request

```json
{
  "query": "explain photosynthesis",
  "studentId": "student-001",
  "businessId": "B001"
}
```

### Chat Response

```json
{
  "success": true,
  "response": {
    "text": "Photosynthesis is the process..."
  },
  "metadata": {
    "image_url": "https://pub-xxx.r2.dev/diagram.png",
    "image_generated": true
  }
}
```

## Environment Variables

See `.env.example` for all required variables:

| Variable | Description |
|----------|-------------|
| `GOOGLE_API_KEY` | Gemini API key |
| `QDRANT_URL` | Qdrant vector database URL |
| `QDRANT_API_KEY` | Qdrant API key |
| `QDRANT_COLLECTION` | Default collection for image upserts (RAG calls pass collection per request) |
| `GOOGLE_SHEETS_STUDENTS_ID` | Student profiles sheet ID |
| `GOOGLE_SERVICE_ACCOUNT_JSON` | Service account JSON |
| `R2_ENDPOINT` | Cloudflare R2 endpoint |
| `R2_ACCESS_KEY_ID` | R2 access key |
| `R2_SECRET_ACCESS_KEY` | R2 secret key |
| `R2_BUCKET_NAME` | R2 bucket name |
| `R2_PUBLIC_URL` | R2 public URL prefix |
| `ALLOWED_ORIGINS` | Comma-separated CORS origins |
| `REQUEST_LOG_PATH` | Request log file path (default: logs/requests.log) |
| `LLM_PROVIDER` | Primary LLM provider (Gemini by default) |
| `LLM_FALLBACKS` | Comma-separated fallback providers |
| `GEMINI_MODEL` | Gemini model name |
| `GEMINI_EMBEDDING_MODEL` | Gemini embedding model |
| `GEMINI_TRANSCRIPTION_MODEL` | Gemini audio transcription model |
| `ANTHROPIC_API_KEY` | Anthropic API key (optional) |
| `ANTHROPIC_MODEL` | Anthropic model name |
| `AGENTS_DIR` | Agent specs directory (default: `config/agents`) |
| `WORKFLOW_SPECS_DIR` | Workflow specs directory (default: `config/workflows`) |
| `WORKFLOW_PROFILES_DIR` | Workflow profile directory (default: `config/workflow_profiles`) |
| `AGENT_STEPS_LOG_PATH` | Step-by-step agent decision/tool log path |
| `AURIKA_TASKS_API_URL` | Optional override for the Aurika task API endpoint, useful for local testing against `http://localhost:3000/api/internal/tasks` |
| `AURIKA_ENGAGE_INTERNAL_API_KEY` | Bearer token used by `aurika_tasks_create_task` |
| `AGENTIC_CORE_API_KEY` | Optional inbound API key; when set, `/api/*` requires `X-API-Key` or `Authorization: Bearer` |
| `AGENTIC_CORE_API_KEY_HEADER` | Optional inbound API-key header name (default: `X-API-Key`) |
| `AGENT_UPLOAD_MAX_FILE_MB` | Max single attachment size for `/api/agent/run` uploads (default: 30) |
| `AGENT_UPLOAD_MAX_TOTAL_MB` | Max combined attachment size for `/api/agent/run` uploads (default: 60) |
| `AGENT_ATTACHMENT_MAX_EXTRACTED_CHARS` | Max extracted text characters injected per attachment (default: 120000) |
| `TAVILY_API_KEY` | Tavily API key for `web_search`/`web_research` tools |
| `ALLOWED_HTTP_TOOL_DOMAINS` | Optional comma-separated allowlist for `http_request` tool |
| `GOOGLE_DRIVE_FOLDER_ID` | Drive folder id where todo spreadsheet is created |
| `GOOGLE_SERVICE_ACCOUNT_JSON_BASE64` | Optional base64-encoded service account JSON |
| `TODO_SHEET_ID` | Optional fixed spreadsheet id for todo store (recommended to avoid Drive create/quota issues) |
| `TODO_SHEET_FILE_NAME` | Spreadsheet file name for todo storage |
| `TODO_WORKSHEET_NAME` | Todo worksheet/tab name |
| `TODO_AUDIT_WORKSHEET` | Audit worksheet/tab name |

## Model Profiles

Define per-task model configs in `config/models.yaml` and reference them with `model_profile`
in block or agent requests.

Example:

```yaml
profiles:
  small:
    provider: gemini
    model: gemini-2.5-flash
    temperature: 0.2
    max_tokens: 512
  large:
    provider: gemini
    model: gemini-2.5-pro
    temperature: 0.5
    max_tokens: 4096
```

Use it in a request:

```bash
curl -X POST http://localhost:8000/api/blocks/llm \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Summarize Newton\\u2019s laws","model_profile":"small"}'
```

## Vector Store Configuration

Each workflow can define its own vector store and metadata filters in
`config/workflow_settings/<workflow>.json`. Every RAG/Agent call must provide a
`collection_name`, either explicitly in the request or via the workflow config.

Example workflow config:

```json
{
  "workflow": "campus_learning_agent",
  "vector_store": {
    "provider": "qdrant",
    "collection": "aurika_campus",
    "filters": {
      "must": [
        { "field": "course", "op": "eq", "value": "{{course}}" },
        { "field": "level", "op": "eq", "value": "{{level}}" },
        { "field": "subject", "op": "eq", "value": "{{subject}}" }
      ],
      "should": [
        { "field": "topic", "op": "eq", "value": "{{topic}}" }
      ],
      "minimum_should_match": 1
    }
  }
}
```

Example request override:

```bash
curl -X POST http://localhost:8000/api/blocks/rag \
  -H "Content-Type: application/json" \
  -d '{
    "query":"Explain photosynthesis",
    "collection_name":"aurika_campus",
    "filters":{"must":[{"field":"subject","op":"eq","value":"Biology"}]}
  }'
```

Use a workflow config instead:

```bash
curl -X POST http://localhost:8000/api/blocks/rag \
  -H "Content-Type: application/json" \
  -d '{
    "query":"Explain photosynthesis",
    "workflow_name":"learning_tutor",
    "context":{"course":"NCERT","level":"Class_10","subject":"Science"}
  }'
```

## Project Structure

```
agentic-core/
├── api/
│   ├── main.py           # FastAPI app
│   ├── routes/
│   │   ├── health.py     # Health check
│   │   ├── learning.py   # Learning tutor
│   │   └── image.py      # Image generation
│   └── schemas.py        # Pydantic models
├── core/
│   ├── adapters/
│   │   └── gemini.py     # Gemini LLM adapter
│   ├── retrieval/
│   │   └── qdrant_client.py  # Qdrant search
│   └── services/
│       ├── image_generator.py   # Image generation
│       ├── r2_uploader.py       # R2 upload
│       ├── student_profile.py   # Google Sheets
│       └── subject_analyzer.py  # Subject detection
├── workflows/
│   ├── learning_tutor.py   # Main tutor workflow
│   └── image_workflow.py   # Image generation workflow
└── config/
    ├── prompts/
    │   └── learning_tutor.txt  # System prompt
    └── workflow_settings/
        └── learning_tutor.json # Workflow config
```

## Development

```bash
# Run with hot reload
uvicorn api.main:app --reload --port 8000

# Run tests (if any)
pytest
```

## Audio Transcription

The `/api/blocks/transcribe` endpoint uses Gemini in batch mode (no storage).

```bash
curl -X POST http://localhost:8000/api/blocks/transcribe \
  -F "file=@/path/to/audio.m4a" \
  -F "language=en"
```

## License

MIT
