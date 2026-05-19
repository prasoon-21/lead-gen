from abc import ABC, abstractmethod
from typing import Any, Dict

from config.schemas import WorkflowNodeSpec


class BaseNodeExecutor(ABC):
    """
    Every node type implements this.  The engine calls execute() — it knows
    nothing about what happens inside.  Add a new node type by writing a new
    class that inherits from this and registering it.
    """

    # Override in subclasses to declare what node_type string this handles.
    node_type: str = ""

    @abstractmethod
    async def execute(
        self,
        node: WorkflowNodeSpec,
        node_input: Dict[str, Any],
        state: Dict[str, Any],
        trace_id: str,
        resources: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Args:
            node        – the node spec (config, input_map, output_map, etc.)
            node_input  – already-resolved inputs (after input_map is applied)
            state       – full workflow state at this point in time
            trace_id    – for logging / tracing
            resources   – shared objects: adapter, agent_factory, tool_registry,
                          config_loader, token_tracker, todo_store, retriever, …

        Returns a plain dict that will be merged into state via output_map.
        """
