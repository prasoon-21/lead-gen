import time
import uuid
import logging
from typing import Any, Dict, Optional

from config.loader import ConfigLoader
from config.schemas import WorkflowNodeSpec, WorkflowSpec
from core.agents.factory import AgentFactory
from core.orchestration.node_registry import NodeRegistry
from core.orchestration.state import get_path, map_inputs, map_outputs, set_path


class WorkflowEngine:
    def __init__(
        self,
        config_loader: ConfigLoader,
        agent_factory: AgentFactory,
        node_registry: NodeRegistry,
        token_tracker=None,
    ):
        self.config_loader = config_loader
        self.agent_factory = agent_factory
        self.node_registry = node_registry
        self.token_tracker = token_tracker
        self._runtime_logger = logging.getLogger("agent.runtime")

        # Shared resources passed into every node executor at call time.
        self._resources = {
            "adapter": agent_factory.base_adapter,
            "retriever": agent_factory.retriever,
            "tool_registry": agent_factory.tool_registry,
            "config_loader": config_loader,
            "token_tracker": token_tracker,
            "agent_factory": agent_factory,
            "todo_store": agent_factory.todo_store,
        }

    def _log(self, trace_id: str, event: str, data: Dict[str, Any]) -> None:
        self._runtime_logger.info(
            event,
            extra={
                "event": event,
                "trace_id": trace_id,
                "session_id": data.get("session_id", "-"),
                "agent_id": data.get("agent_id", "-"),
                "workflow_id": data.get("workflow_id", "-"),
                "step": data.get("step", 0),
                "tool_name": data.get("tool_name", "-"),
                "status": data.get("status", "-"),
            },
        )
        if self.token_tracker:
            self.token_tracker.log_event(event, {"trace_id": trace_id, **data})

    def _next_node(self, node: WorkflowNodeSpec, state: Dict[str, Any]) -> Optional[str]:
        for transition in node.transitions:
            if self._check_condition(transition.when, state):
                return transition.to
        return None

    @staticmethod
    def _check_condition(condition: str, state: Dict[str, Any]) -> bool:
        cond = (condition or "always").strip()
        if cond == "always":
            return True
        if cond.startswith("truthy:"):
            return bool(get_path(state, cond.split(":", 1)[1], None))
        if cond.startswith("exists:"):
            return get_path(state, cond.split(":", 1)[1], None) is not None
        if cond.startswith("equals:"):
            _, path, expected = cond.split(":", 2)
            return str(get_path(state, path, "")) == expected
        if cond.startswith("not_equals:"):
            _, path, expected = cond.split(":", 2)
            return str(get_path(state, path, "")) != expected
        return False

    async def _run_node(
        self,
        node: WorkflowNodeSpec,
        state: Dict[str, Any],
        trace_id: str,
    ) -> Dict[str, Any]:
        node_input = map_inputs(state, node.input_map)
        executor = self.node_registry.get(node.node_type)
        return await executor.execute(node, node_input, state, trace_id, self._resources)

    async def run(
        self,
        workflow_id: str,
        payload: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        spec: WorkflowSpec = self.config_loader.load_workflow_spec(workflow_id)
        if not spec.start_at:
            raise ValueError(f"start_at is required in workflow spec: {workflow_id}")

        trace_id = f"WTRACE_{uuid.uuid4()}"
        state: Dict[str, Any] = {
            "workflow_id": workflow_id,
            "session_id": session_id or f"SESS_{uuid.uuid4()}",
            "input": payload,
            "nodes": {},
        }
        if isinstance(payload, dict):
            state.update(payload)

        current_node_id = spec.start_at
        visited_steps = 0
        max_steps = max(20, len(spec.nodes) * 4)
        started = time.perf_counter()
        aggregated_tool_calls = 0
        aggregated_input_tokens = 0
        aggregated_output_tokens = 0
        node_summaries = []

        while current_node_id and visited_steps < max_steps:
            visited_steps += 1
            node = spec.get_node(current_node_id)
            if not node:
                raise ValueError(f"Node not found in workflow '{workflow_id}': {current_node_id}")

            self._log(
                trace_id,
                "workflow_node_start",
                {
                    "workflow_id": workflow_id,
                    "session_id": state.get("session_id"),
                    "status": f"node={node.node_id} type={node.node_type}",
                },
            )
            node_output = await self._run_node(node, state, trace_id)
            set_path(state, f"nodes.{node.node_id}.output", node_output)
            map_outputs(state, node_output if isinstance(node_output, dict) else {"result": node_output}, node.output_map)
            if isinstance(node_output, dict):
                node_meta = node_output.get("metadata", {}) if isinstance(node_output.get("metadata"), dict) else {}
                aggregated_tool_calls += int(node_meta.get("tool_calls", 0) or 0)
                aggregated_input_tokens += int(node_meta.get("input_tokens", 0) or 0)
                aggregated_output_tokens += int(node_meta.get("output_tokens", 0) or 0)
                node_summaries.append(
                    {
                        "node_id": node.node_id,
                        "node_type": node.node_type,
                        "tool_calls": int(node_meta.get("tool_calls", 0) or 0),
                        "input_tokens": int(node_meta.get("input_tokens", 0) or 0),
                        "output_tokens": int(node_meta.get("output_tokens", 0) or 0),
                        "latency_ms": float(node_meta.get("latency_ms", 0) or 0),
                        "manual_flow": bool(node_meta.get("manual_flow", False)),
                    }
                )
            self._log(
                trace_id,
                "workflow_node_end",
                {
                    "workflow_id": workflow_id,
                    "session_id": state.get("session_id"),
                    "status": f"node={node.node_id} type={node.node_type}",
                },
            )

            current_node_id = self._next_node(node, state)

        return {
            "success": True,
            "trace_id": trace_id,
            "workflow_id": workflow_id,
            "session_id": state.get("session_id"),
            "state": state,
            "metadata": {
                "steps": visited_steps,
                "latency_ms": (time.perf_counter() - started) * 1000,
                "tool_calls": aggregated_tool_calls,
                "input_tokens": aggregated_input_tokens,
                "output_tokens": aggregated_output_tokens,
                "total_tokens": aggregated_input_tokens + aggregated_output_tokens,
                "node_summaries": node_summaries,
            },
        }
