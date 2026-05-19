from typing import Dict, Type

from core.orchestration.executors.base import BaseNodeExecutor


class NodeRegistry:
    """
    Maps node_type strings to executor instances.

    Usage:
        registry = NodeRegistry()
        registry.register(AgentNodeExecutor())
        registry.register(ToolNodeExecutor())

        # In engine:
        executor = registry.get("agent")
        output = await executor.execute(node, node_input, state, trace_id, resources)

    To add a completely new node type without touching the engine:
        registry.register(MyCustomNodeExecutor())
    """

    def __init__(self) -> None:
        self._executors: Dict[str, BaseNodeExecutor] = {}

    def register(self, executor: BaseNodeExecutor) -> None:
        self._executors[executor.node_type] = executor

    def get(self, node_type: str) -> BaseNodeExecutor:
        executor = self._executors.get(node_type.strip().lower())
        if executor is None:
            registered = list(self._executors.keys())
            raise ValueError(
                f"Unknown node_type '{node_type}'. Registered types: {registered}"
            )
        return executor

    def registered_types(self) -> list:
        return list(self._executors.keys())


def build_default_node_registry(resources: dict) -> NodeRegistry:
    """
    Builds and returns the registry pre-loaded with all built-in node types.
    Call this once at startup and pass the registry into WorkflowEngine.
    """
    from core.orchestration.executors.agent_executor import AgentNodeExecutor
    from core.orchestration.executors.tool_executor import ToolNodeExecutor
    from core.orchestration.executors.llm_executor import LLMNodeExecutor
    from core.orchestration.executors.rag_executor import RAGNodeExecutor
    from core.orchestration.executors.classification_executor import ClassificationNodeExecutor

    registry = NodeRegistry()
    registry.register(AgentNodeExecutor())
    registry.register(ToolNodeExecutor())
    registry.register(LLMNodeExecutor())
    registry.register(RAGNodeExecutor())
    registry.register(ClassificationNodeExecutor())
    return registry
