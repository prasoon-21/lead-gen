import uuid
from typing import Any, Dict

from config.schemas import WorkflowNodeSpec
from core.orchestration.executors.base import BaseNodeExecutor
from core.orchestration.state import get_path


class AgentNodeExecutor(BaseNodeExecutor):
    node_type = "agent"

    async def execute(
        self,
        node: WorkflowNodeSpec,
        node_input: Dict[str, Any],
        state: Dict[str, Any],
        trace_id: str,
        resources: Dict[str, Any],
    ) -> Dict[str, Any]:
        agent_factory = resources["agent_factory"]
        config = node.config or {}

        agent_id = config.get("agent_id")
        if not agent_id:
            raise ValueError(f"agent_id is required for node '{node.node_id}'")

        message = (
            node_input.get("message")
            or config.get("message")
            or get_path(state, "input.message", "")
        )
        context = node_input.get("context") or config.get("context") or {}
        session_id = (
            node_input.get("session_id")
            or get_path(state, "session_id")
            or f"SESS_{uuid.uuid4()}"
        )

        return await agent_factory.run(
            agent_id=agent_id,
            message=message,
            session_id=session_id,
            context=context,
        )
