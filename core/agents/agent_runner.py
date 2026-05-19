import json
from typing import Dict, Any, Optional

from core.adapters.base import BaseLLMAdapter
from core.agents.memory import MemoryStore
from core.agents.rag_agent import RAGAgent
from core.extraction.json_extractor import JSONExtractor
from core.retrieval.qdrant_client import QdrantRetriever


class AgentRunner:
    def __init__(
        self,
        llm: BaseLLMAdapter,
        retriever: Optional[QdrantRetriever] = None,
        memory_store: Optional[MemoryStore] = None,
        memory_window: int = 6,
    ):
        self.llm = llm
        self.retriever = retriever
        self.memory_store = memory_store or MemoryStore(max_turns=memory_window)
        self.memory_window = memory_window

    async def run(
        self,
        session_id: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        response_mode: str = "text",
        schema: Optional[str] = None,
        use_retrieval: bool = True,
        rewrite_query: bool = True,
        max_chunks: int = 5,
        response_temperature: float = 0.7,
        response_max_tokens: int = 4096,
        collection_name: Optional[str] = None,
        filter_config: Optional["FilterConfig"] = None,
    ) -> Dict[str, Any]:
        context = context or {}
        memory_context = self.memory_store.get_context(session_id, limit=self.memory_window)

        rag_agent = RAGAgent(
            llm=self.llm,
            retriever=self.retriever,
            rewrite_query=rewrite_query,
            max_chunks=max_chunks,
            response_temperature=response_temperature,
            response_max_tokens=response_max_tokens,
        )

        rag_result = await rag_agent.run(
            query=message,
            context=context,
            system_prompt=system_prompt,
            use_retrieval=use_retrieval,
            include_retrieval=False,
            memory_context=memory_context,
            collection_name=collection_name,
            filter_config=filter_config,
        )

        response_text = rag_result.get("text", "")
        response_json = None

        if response_mode == "json":
            if not schema:
                raise ValueError("schema is required for response_mode=json")
            extractor = JSONExtractor(self.llm)
            response_json = await extractor.extract(
                text=response_text,
                schema_description=schema,
            )

        self.memory_store.add_message(session_id, "user", message)
        if response_mode == "json":
            self.memory_store.add_message(
                session_id,
                "assistant",
                json.dumps(response_json or {}, ensure_ascii=False),
            )
        else:
            self.memory_store.add_message(session_id, "assistant", response_text)

        metadata = {
            "search_query": rag_result.get("search_query"),
            "retrieval_used": rag_result.get("retrieval_used", False),
            "retrieval_count": rag_result.get("retrieval_count", 0),
            "input_tokens": rag_result.get("input_tokens", 0),
            "output_tokens": rag_result.get("output_tokens", 0),
            "latency_ms": rag_result.get("latency_ms", 0),
        }

        return {
            "text": response_text if response_mode == "text" else None,
            "json": response_json if response_mode == "json" else None,
            "metadata": metadata,
        }
