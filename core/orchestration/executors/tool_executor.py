import uuid
from typing import Any, Dict

from config.schemas import WorkflowNodeSpec
from core.orchestration.executors.base import BaseNodeExecutor
from core.orchestration.state import get_path
from core.tools import ToolContext, ToolExecutor


class ToolNodeExecutor(BaseNodeExecutor):
    node_type = "tool"

    async def execute(
        self,
        node: WorkflowNodeSpec,
        node_input: Dict[str, Any],
        state: Dict[str, Any],
        trace_id: str,
        resources: Dict[str, Any],
    ) -> Dict[str, Any]:
        config = node.config or {}

        tool_name = config.get("tool_name")
        if not tool_name:
            raise ValueError(f"tool_name is required for node '{node.node_id}'")

        arguments = dict(config.get("arguments", {}) or {})
        arguments.update(node_input)

        tool_context = ToolContext(
            session_id=get_path(state, "session_id", f"SESS_{uuid.uuid4()}"),
            trace_id=trace_id,
            resources={
                "adapter": resources.get("adapter"),
                "retriever": resources.get("retriever"),
                "config_loader": resources.get("config_loader"),
                "token_tracker": resources.get("token_tracker"),
                "agent_factory": resources.get("agent_factory"),
                "todo_store": resources.get("todo_store"),
            },
            state=state,
        )

        tool_executor = ToolExecutor(resources["tool_registry"])
        result = await tool_executor.execute(tool_name, arguments, tool_context)
        return result.to_dict()
