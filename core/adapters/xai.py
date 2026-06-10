from typing import Optional

from core.adapters.openai import OpenAIAdapter


class XAIAdapter(OpenAIAdapter):
    DEFAULT_MODEL = "grok-4.3"
    BASE_URL = "https://api.x.ai/v1"

    def __init__(
        self,
        api_key: str,
        model_name: str = DEFAULT_MODEL,
        embedding_model: str = "",
    ):
        super().__init__(
            api_key=api_key,
            model_name=model_name,
            embedding_model=embedding_model or "",
            base_url=self.BASE_URL,
        )

    async def embed(self, text: str):
        raise NotImplementedError("xAI embeddings are not configured; fall back to another provider for embeddings.")
