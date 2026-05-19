from typing import Any, Dict

from config.schemas import WorkflowNodeSpec
from core.agents.rag_agent import RAGAgent
from core.orchestration.executors.base import BaseNodeExecutor


class RAGNodeExecutor(BaseNodeExecutor):
    node_type = "rag"

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
        rag = RAGAgent(
            llm=adapter,
            retriever=resources.get("retriever"),
            rewrite_query=bool(config.get("rewrite_query", True)),
            max_chunks=int(config.get("max_chunks", 5)),
            response_temperature=float(config.get("temperature", 0.7)),
            response_max_tokens=int(config.get("max_tokens", 4096)),
        )

        config_loader = resources.get("config_loader")
        filters = node_input.get("filters") or config.get("filters")
        filter_config = config_loader.parse_filter_config(filters) if filters and config_loader else None

        return await rag.run(
            query=node_input.get("query") or config.get("query") or "",
            context=node_input.get("context") or config.get("context") or {},
            system_prompt=config.get("system_prompt"),
            use_retrieval=bool(config.get("use_retrieval", True)),
            include_retrieval=bool(config.get("include_retrieval", False)),
            collection_name=node_input.get("collection_name") or config.get("collection_name"),
            filter_config=filter_config,
        )
