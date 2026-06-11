from typing import Any, Dict, List, Optional

from core.adapters.base import BaseLLMAdapter, LLMResponse


class OpenAIAdapter(BaseLLMAdapter):
    """
    OpenAI support has been disabled for this project.

    The class remains as a stub so older imports fail clearly instead of
    breaking module resolution elsewhere in the codebase.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str,
        embedding_model: str,
        base_url: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
    ):
        super().__init__(api_key, model_name)
        self.embedding_model = embedding_model
        self.base_url = base_url
        self.default_headers = default_headers or {}
        raise NotImplementedError("OpenAI support is disabled. Configure GOOGLE_API_KEY and use Gemini instead.")

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        raise NotImplementedError("OpenAI support is disabled. Use Gemini instead.")

    async def generate_with_vision(
        self,
        prompt: str,
        images: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
    ) -> LLMResponse:
        raise NotImplementedError("OpenAI support is disabled. Use Gemini instead.")

    async def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
    ) -> Dict[str, Any]:
        raise NotImplementedError("OpenAI support is disabled. Use Gemini instead.")

    async def embed(self, text: str) -> List[float]:
        raise NotImplementedError("OpenAI support is disabled. Use Gemini embeddings instead.")
