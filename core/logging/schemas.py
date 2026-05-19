from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List


@dataclass
class RetrievalLog:
    query: str
    collection: str
    results_count: int
    top_scores: List[float] = field(default_factory=list)
    filter_used: Optional[Dict[str, Any]] = None


@dataclass
class AgentLog:
    timestamp: str
    session_id: str
    student_id: str
    student_class: str
    workflow: str
    user_query: str
    detected_subject: str
    detected_intent: str
    retrieval: Optional[RetrievalLog]
    ai_response: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: float
    student_rating: Optional[int] = None
    teacher_approved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        data = {
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "student_id": self.student_id,
            "student_class": self.student_class,
            "workflow": self.workflow,
            "user_query": self.user_query,
            "detected_subject": self.detected_subject,
            "detected_intent": self.detected_intent,
            "ai_response": self.ai_response,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
            "student_rating": self.student_rating,
            "teacher_approved": self.teacher_approved
        }
        
        if self.retrieval:
            data["retrieval"] = {
                "query": self.retrieval.query,
                "collection": self.retrieval.collection,
                "results_count": self.retrieval.results_count,
                "top_scores": self.retrieval.top_scores,
                "filter_used": self.retrieval.filter_used
            }
        
        return data
    
    @classmethod
    def create(
        cls,
        session_id: str,
        student_id: str,
        student_class: str,
        workflow: str,
        user_query: str,
        detected_subject: str,
        detected_intent: str,
        ai_response: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        retrieval: Optional[RetrievalLog] = None
    ) -> "AgentLog":
        return cls(
            timestamp=datetime.now().isoformat(),
            session_id=session_id,
            student_id=student_id,
            student_class=student_class,
            workflow=workflow,
            user_query=user_query,
            detected_subject=detected_subject,
            detected_intent=detected_intent,
            retrieval=retrieval,
            ai_response=ai_response,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            latency_ms=latency_ms
        )


@dataclass
class AgentStepLog:
    timestamp: str
    trace_id: str
    session_id: str
    agent_id: str
    step: int
    event: str
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "step": self.step,
            "event": self.event,
            "payload": self.payload,
        }

    @classmethod
    def create(
        cls,
        trace_id: str,
        session_id: str,
        agent_id: str,
        step: int,
        event: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> "AgentStepLog":
        return cls(
            timestamp=datetime.now().isoformat(),
            trace_id=trace_id,
            session_id=session_id,
            agent_id=agent_id,
            step=step,
            event=event,
            payload=payload or {},
        )
