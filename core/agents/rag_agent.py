import logging
from typing import Dict, Any, List, Optional

from core.adapters.base import BaseLLMAdapter, LLMResponse
from core.retrieval.qdrant_client import QdrantRetriever


class RAGAgent:
    DEFAULT_REWRITE_PROMPT = (
        "You are rewriting a student question into a short search query for a vector database.\n"
        "Return only the search query, no quotes or extra text.\n\n"
        "Question: {query}\n"
        "Context: subject={subject}, class={class_level}, board={board}\n"
    )

    DEFAULT_RESPONSE_PROMPT = (
        "Use the retrieved context to answer the question clearly and directly.\n"
        "If context is empty, answer using your own knowledge.\n\n"
        "Question: {query}\n\n"
        "{retrieved_context}\n"
    )

    def __init__(
        self,
        llm: BaseLLMAdapter,
        retriever: Optional[QdrantRetriever] = None,
        rewrite_query: bool = True,
        max_chunks: int = 5,
        response_temperature: float = 0.7,
        response_max_tokens: int = 4096,
    ):
        self.llm = llm
        self.retriever = retriever
        self.rewrite_query = rewrite_query
        self.max_chunks = max_chunks
        self.response_temperature = response_temperature
        self.response_max_tokens = response_max_tokens
        self._logger = logging.getLogger("core.rag_agent")

    async def run(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        use_retrieval: bool = True,
        include_retrieval: bool = False,
        memory_context: Optional[str] = None,
        collection_name: Optional[str] = None,
        filter_config: Optional["FilterConfig"] = None,
    ) -> Dict[str, Any]:
        context = context or {}
        search_query = query

        if use_retrieval and self.retriever and self.rewrite_query:
            search_query = await self._rewrite_query(query, context)

        retrieval_results: List[Dict[str, Any]] = []
        if use_retrieval and self.retriever:
            if not collection_name:
                raise ValueError("collection_name is required for RAG retrieval")
            retrieval_results = await self.retriever.search(
                search_query,
                context,
                top_k=self.max_chunks,
                collection_name=collection_name,
                filter_config=filter_config,
            )

        retrieved_context = self._format_results(retrieval_results)
        prompt = self._build_prompt(query, retrieved_context, memory_context)

        response = await self.llm.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=self.response_temperature,
            max_tokens=self.response_max_tokens,
        )

        result = {
            "text": response.text,
            "search_query": search_query,
            "retrieval_used": bool(retrieval_results),
            "retrieval_count": len(retrieval_results),
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "latency_ms": response.latency_ms,
        }

        if include_retrieval:
            result["retrieval_results"] = retrieval_results

        return result

    async def _rewrite_query(self, query: str, context: Dict[str, Any]) -> str:
        prompt = self.DEFAULT_REWRITE_PROMPT.format(
            query=query,
            subject=context.get("subject", "N/A"),
            class_level=context.get("class", "N/A"),
            board=context.get("board", "N/A"),
        )
        try:
            response = await self.llm.generate(prompt=prompt, temperature=0.1, max_tokens=64)
            return response.text.strip() or query
        except Exception as exc:
            self._logger.warning("Query rewrite failed, using original query: %s", exc)
            return query

    def _format_results(self, results: List[Dict[str, Any]]) -> str:
        if not results:
            return "Retrieved context: (none)\n"
        lines = ["Retrieved context:"]
        for idx, item in enumerate(results, 1):
            content = item.get("content", "")
            meta = item.get("metadata", {}) or {}
            topic = meta.get("topic_name", "") or meta.get("chapter_name", "")
            lines.append(f"[{idx}] {topic}\n{content}")
        return "\n".join(lines)

    def _build_prompt(
        self,
        query: str,
        retrieved_context: str,
        memory_context: Optional[str],
    ) -> str:
        parts = []
        if memory_context:
            parts.append(memory_context)
        parts.append(
            self.DEFAULT_RESPONSE_PROMPT.format(
                query=query,
                retrieved_context=retrieved_context,
            )
        )
        return "\n\n".join(parts)
