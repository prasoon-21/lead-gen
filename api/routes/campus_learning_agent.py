import uuid
from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from api.schemas import CampusLearningRequest, ChatResponse, ResponseContent, ErrorResponse
from api.state import get_state
from core.services.subject_analyzer import SubjectAnalyzer
from core.services.file_analyzer import FileAnalyzer

router = APIRouter()

# Subject memory: session_id → last detected subject
_subject_memory: Dict[str, str] = {}


@router.post("/learning_agent", response_model=ChatResponse, responses={500: {"model": ErrorResponse}})
async def campus_learning_agent(request: CampusLearningRequest):
    try:
        state = get_state()

        if not state.adapter:
            raise HTTPException(status_code=500, detail="LLM adapter not initialized")

        session_id = request.session_id or f"SESS_{uuid.uuid4()}"

        # ── 1. Build student context from request ─────────────────────────────
        context: Dict[str, Any] = {
            "name": request.student_name or "Student",
            "class": request.level,
            "level": request.level,
            "board": request.board or "CBSE",
            "medium": request.medium,
            "course_type": request.course_type,
            "institute_type": request.course_type,
            "topic": request.topic or "",
            "subtopic": request.subtopic or "",
            "learning_preferences": request.learning_preferences or "Standard learning style",
            "learning_goal": "conceptual understanding",
            "student_id": request.student_id,
            "business_id": request.business_id,
        }

        # ── 2. Vector store config ────────────────────────────────────────────
        try:
            workflow_cfg = state.config_loader.load_workflow("campus_learning_agent")
        except FileNotFoundError:
            raise HTTPException(
                status_code=500,
                detail="Workflow settings not found for campus_learning_agent",
            )

        if not workflow_cfg.vector_store or not workflow_cfg.vector_store.collection:
            raise HTTPException(
                status_code=500,
                detail="Vector store collection not configured for campus_learning_agent",
            )
        context["collection_name"] = workflow_cfg.vector_store.collection

        # ── 3. Subject detection ──────────────────────────────────────────────
        subject_analyzer = SubjectAnalyzer(state.adapter)
        last_subject = _subject_memory.get(session_id)
        detected_subject = await subject_analyzer.classify_subject(
            query=request.query,
            current_subject=last_subject,
        )
        _subject_memory[session_id] = detected_subject
        context["subject"] = detected_subject

        # ── 4. RAG filters ────────────────────────────────────────────────────
        if workflow_cfg.vector_store.filters:
            context["filter_config"] = state.config_loader.resolve_filter_placeholders(
                workflow_cfg.vector_store.filters,
                context,
            )

        # ── 5. Build message (documents take priority) ────────────────────────
        documents: List[Dict[str, Any]] = [
            {"base64": doc.base64, "mimeType": doc.mimeType, "filename": doc.filename}
            for doc in request.documents
        ]
        if request.file_base64:
            documents.append({
                "base64": request.file_base64,
                "mimeType": request.file_mime_type or "application/octet-stream",
                "filename": request.file_name or "uploaded-file",
            })

        if documents:
            file_analyzer = FileAnalyzer(state.adapter)
            doc = documents[0]
            base64_data = doc["base64"]
            if "," in base64_data:
                base64_data = base64_data.split(",")[1]
            analysis = await file_analyzer.analyze_for_tutoring(
                document={"base64": base64_data, "mimeType": doc["mimeType"], "filename": doc["filename"]},
                user_query=request.query,
            )
            message = f"Based on this document analysis: {analysis}\n\nUser's question: {request.query}"
        else:
            message = request.query

        # ── 6. Run via workflow engine ────────────────────────────────────────
        result = await state.workflow_engine.run(
            workflow_id="campus_learning_agent",
            payload={"message": message, "context": context},
            session_id=session_id,
        )

        wf_state = result.get("state", {})
        outputs = wf_state.get("outputs", {})
        response_text = outputs.get("text", "")
        agent_metadata = outputs.get("metadata", {}) or {}

        return ChatResponse(
            success=True,
            sessionId=session_id,
            studentId=request.student_id,
            businessId=request.business_id,
            detectedSubject=detected_subject,
            response=ResponseContent(
                text=response_text,
                studentName=context.get("name", "Student"),
                subject=detected_subject,
                class_=context.get("level", "N/A"),
                chapter=context.get("topic", "N/A") or "N/A",
            ),
            metadata={
                "isGroup": False,
                "messageType": "document" if documents else "database",
                "timestamp": datetime.now().isoformat(),
                "retrieval_used": agent_metadata.get("retrieval_used", False),
                "retrieval_count": agent_metadata.get("retrieval_count", 0),
                "image_url": agent_metadata.get("image_url"),
                "image_generated": agent_metadata.get("image_generated", False),
                "trace_id": result.get("trace_id"),
            },
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/learning_agent/clear-memory")
async def clear_session_memory(session_id: str):
    _subject_memory.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}
