from typing import Any, Dict

from config.schemas import WorkflowNodeSpec
from core.orchestration.classification import (
    build_classification_prompt,
    build_classification_system_prompt,
)
from core.orchestration.executors.base import BaseNodeExecutor
from core.orchestration.state import get_path


class ClassificationNodeExecutor(BaseNodeExecutor):
    node_type = "classification"

    async def execute(
        self,
        node: WorkflowNodeSpec,
        node_input: Dict[str, Any],
        state: Dict[str, Any],
        trace_id: str,
        resources: Dict[str, Any],
    ) -> Dict[str, Any]:
        adapter = resources.get("adapter")
        if not adapter:
            raise ValueError("LLM adapter not configured")

        config = node.config or {}
        config_loader = resources.get("config_loader")

        profile_id = config.get("profile_id")
        if not profile_id:
            raise ValueError(f"profile_id is required for node '{node.node_id}'")

        profile = config_loader.load_classification_profile(profile_id)

        conversation_text = (
            node_input.get("conversation_text")
            or config.get("conversation_text")
            or get_path(state, "input.conversation_text", "")
        )
        if not conversation_text:
            raise ValueError(f"conversation_text is required for node '{node.node_id}'")

        additional_context = {
            k: v for k, v in node_input.items()
            if k != "conversation_text" and v is not None
        }

        result = await adapter.generate_json(
            prompt=build_classification_prompt(
                profile=profile,
                conversation_text=str(conversation_text),
                additional_context=additional_context,
            ),
            system_prompt=build_classification_system_prompt(profile),
            temperature=float(config.get("temperature", 0.1)),
        )

        return {"result": result, "profile_id": profile.profile_id}
