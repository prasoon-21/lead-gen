from typing import Any, Dict

from config.schemas import WorkflowNodeSpec
from core.orchestration.executors.base import BaseNodeExecutor


class LLMNodeExecutor(BaseNodeExecutor):
    node_type = "llm"

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
        prompt = node_input.get("prompt") or config.get("prompt", "")
        system_prompt = node_input.get("system_prompt") or config.get("system_prompt")

        response = await adapter.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=float(config.get("temperature", 0.7)),
            max_tokens=int(config.get("max_tokens", 1024)),
        )

        return {
            "text": response.text,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "latency_ms": response.latency_ms,
            "model": response.model,
        }
