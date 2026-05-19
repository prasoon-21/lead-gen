import csv
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from core.logging.schemas import AgentLog, RetrievalLog, AgentStepLog


@dataclass
class NodeLog:
    workflow: str
    node_name: str
    node_type: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_text: str
    output_text: str
    latency_ms: float
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class TokenTracker:
    
    CSV_HEADERS = [
        "timestamp", "workflow", "node_name", "node_type", "model",
        "input_tokens", "output_tokens", "total_tokens", "latency_ms"
    ]
    
    def __init__(
        self,
        output_path: str = "logs/token_usage.csv",
        jsonl_path: str = "logs/tutor_logs.jsonl",
        steps_jsonl_path: str = "logs/agent_steps.jsonl",
    ):
        import os
        if os.getenv("VERCEL") == "1":
            output_path = os.path.join("/tmp", os.path.basename(output_path))
            jsonl_path = os.path.join("/tmp", os.path.basename(jsonl_path))
            steps_jsonl_path = os.path.join("/tmp", os.path.basename(steps_jsonl_path))
        self.output_path = Path(output_path)
        self.jsonl_path = Path(jsonl_path)
        self.steps_jsonl_path = Path(steps_jsonl_path)
        self.logs: List[NodeLog] = []
        self._ensure_log_dir()
    
    def _ensure_log_dir(self):
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            self.steps_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.output_path.exists():
                with open(self.output_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(self.CSV_HEADERS)
        except Exception as exc:
            print(f"Warning: Could not initialize token tracker directories or files: {exc}")
    
    def log(self, log: NodeLog):
        self.logs.append(log)
        self._append_to_csv(log)
    
    def _append_to_csv(self, log: NodeLog):
        row = [
            log.timestamp,
            log.workflow,
            log.node_name,
            log.node_type,
            log.model,
            log.input_tokens,
            log.output_tokens,
            log.total_tokens,
            log.latency_ms
        ]
        
        try:
            with open(self.output_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except Exception as exc:
            print(f"Warning: Could not write token usage to CSV: {exc}")
    
    def log_event(self, event: str, data: Dict[str, Any]):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
            **data
        }
        try:
            with open(self.jsonl_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        except Exception as exc:
            print(f"Warning: Could not write event log to JSONL: {exc}")
    
    def log_agent_interaction(
        self,
        session_id: str,
        student_id: str,
        student_class: str,
        workflow: str,
        user_query: str,
        detected_subject: str,
        ai_response: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        retrieval_count: int = 0,
        retrieval_scores: List[float] = None
    ):
        retrieval = None
        if retrieval_count > 0:
            retrieval = RetrievalLog(
                query=user_query,
                collection="aurika_campus_content",
                results_count=retrieval_count,
                top_scores=retrieval_scores or []
            )
        
        agent_log = AgentLog.create(
            session_id=session_id,
            student_id=student_id,
            student_class=student_class,
            workflow=workflow,
            user_query=user_query,
            detected_subject=detected_subject,
            detected_intent="academic",
            ai_response=ai_response[:500],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            retrieval=retrieval
        )
        
        try:
            with open(self.jsonl_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(agent_log.to_dict(), ensure_ascii=False) + '\n')
        except Exception as exc:
            print(f"Warning: Could not write agent interaction log to JSONL: {exc}")
    
    def log_from_response(
        self,
        workflow: str,
        node_name: str,
        node_type: str,
        model: str,
        input_text: str,
        output_text: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float
    ):
        node_log = NodeLog(
            workflow=workflow,
            node_name=node_name,
            node_type=node_type,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            input_text=input_text,
            output_text=output_text,
            latency_ms=latency_ms
        )
        self.log(node_log)

    def log_agent_step(
        self,
        trace_id: str,
        session_id: str,
        agent_id: str,
        step: int,
        event: str,
        payload: Optional[Dict[str, Any]] = None,
    ):
        step_log = AgentStepLog.create(
            trace_id=trace_id,
            session_id=session_id,
            agent_id=agent_id,
            step=step,
            event=event,
            payload=payload,
        )
        try:
            with open(self.steps_jsonl_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(step_log.to_dict(), ensure_ascii=False) + '\n')
        except Exception as exc:
            print(f"Warning: Could not write agent step log to JSONL: {exc}")
    
    def get_workflow_summary(self, workflow: str) -> Dict[str, Any]:
        workflow_logs = [log for log in self.logs if log.workflow == workflow]
        
        if not workflow_logs:
            return {"workflow": workflow, "total_calls": 0}
        
        total_input = sum(log.input_tokens for log in workflow_logs)
        total_output = sum(log.output_tokens for log in workflow_logs)
        total_latency = sum(log.latency_ms for log in workflow_logs)
        
        return {
            "workflow": workflow,
            "total_calls": len(workflow_logs),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "total_latency_ms": total_latency,
            "avg_latency_ms": total_latency / len(workflow_logs)
        }
    
    def export_json(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            for log in self.logs:
                f.write(json.dumps(asdict(log), ensure_ascii=False) + '\n')
