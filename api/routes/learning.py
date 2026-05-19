from datetime import datetime

from fastapi import APIRouter, HTTPException

from api.schemas import ChatRequest, ChatResponse, ResponseContent, ErrorResponse
from api.state import get_state
from core.services.subject_analyzer import SubjectAnalyzer
from core.services.file_analyzer import FileAnalyzer
from core.services.student_profile import StudentProfileService

router = APIRouter()

_student_service = StudentProfileService()
# Subject memory: session_id → last detected subject
_subject_memory: dict[str, str] = {}


@router.post("/{student_id}", response_model=ChatResponse, responses={500: {"model": ErrorResponse}})
async def learning_chat(student_id: str, request: ChatRequest):
    try:
        state = get_state()

        if not state.adapter:
            raise HTTPException(status_code=500, detail="LLM adapter not initialized")

        session_id = request.sessionId or f"SESS_{student_id}_{int(datetime.now().timestamp())}"

        # ── 1. Student profile ────────────────────────────────────────────────
        student_profile = await _student_service.get_student_profile(student_id)

        context: dict = {
            "name": student_profile.get("name", "Student"),
            "class": student_profile.get("class", "Class_10"),
            "level": student_profile.get("class", "Class_10"),
            "board": student_profile.get("board", "CBSE"),
            "medium": student_profile.get("medium", "English"),
            "chapter": student_profile.get("chapter_name", "N/A"),
            "topic": student_profile.get("topic", "N/A"),
            "learning_preferences": student_profile.get("learning_preferences", "Standard learning style"),
            "learning_goal": student_profile.get("learning_goal", "conceptual understanding"),
            "institute_type": student_profile.get("institute_type", "School"),
            "course": student_profile.get("course", ""),
            "student_id": student_id,
            "batch": student_profile.get("class", "Class_10"),
        }

        # ── 2. Vector store config from workflow settings ─────────────────────
        try:
            workflow_cfg = state.config_loader.load_workflow("learning_tutor")
            if not workflow_cfg.vector_store or not workflow_cfg.vector_store.collection:
                raise HTTPException(
                    status_code=500,
                    detail="Vector store collection not configured for learning_tutor",
                )
            context["collection_name"] = workflow_cfg.vector_store.collection
        except FileNotFoundError:
            raise HTTPException(
                status_code=500,
                detail="Workflow settings not found for learning_tutor",
            )

        # ── 3. Build message payload (document or text query) ─────────────────
        if request.documents and len(request.documents) > 0:
            doc = request.documents[0]
            file_analyzer = FileAnalyzer(state.adapter)
            base64_data = doc.base64
            if "," in base64_data:
                base64_data = base64_data.split(",")[1]
            analysis = await file_analyzer.analyze_for_tutoring(
                document={"base64": base64_data, "mimeType": doc.mimeType, "filename": doc.filename},
                user_query=request.query,
            )
            message = f"Based on this document analysis: {analysis}\n\nUser's question: {request.query}"
            detected_subject = context.get("subject", "Document Analysis")
        else:
            # ── 4. Subject detection ──────────────────────────────────────────
            subject_analyzer = SubjectAnalyzer(state.adapter)
            last_subject = _subject_memory.get(session_id)
            detected_subject = await subject_analyzer.classify_subject(
                query=request.query,
                current_subject=last_subject,
            )
            _subject_memory[session_id] = detected_subject
            context["subject"] = detected_subject

            # ── 5. RAG filters ────────────────────────────────────────────────
            if workflow_cfg.vector_store.filters:
                context["filter_config"] = state.config_loader.resolve_filter_placeholders(
                    workflow_cfg.vector_store.filters,
                    context,
                )
            message = request.query

        # ── 6. Run via workflow engine ────────────────────────────────────────
        result = await state.workflow_engine.run(
            workflow_id="learning_tutor",
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
            studentId=request.studentId,
            businessId=request.businessId,
            detectedSubject=detected_subject,
            response=ResponseContent(
                text=response_text,
                studentName=context.get("name", "Student"),
                subject=detected_subject,
                class_=context.get("class", "N/A"),
                chapter=context.get("chapter", "N/A"),
            ),
            metadata={
                "isGroup": False,
                "messageType": "document" if (request.documents and len(request.documents) > 0) else "database",
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


@router.post("/learning/clear-memory")
async def clear_session_memory(session_id: str):
    _subject_memory.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}
