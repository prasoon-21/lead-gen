from typing import List, Optional

from core.adapters.base import BaseLLMAdapter


class EmbeddingService:
    
    def __init__(self, adapter: BaseLLMAdapter):
        self.adapter = adapter
    
    async def embed(self, text: str) -> List[float]:
        return await self.adapter.embed(text)
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            embedding = await self.embed(text)
            embeddings.append(embedding)
        return embeddings
