from core.adapters.base import BaseLLMAdapter


class XAIAdapter(BaseLLMAdapter):
    DEFAULT_MODEL = "grok-4.3"
    BASE_URL = "https://api.x.ai/v1"

    def __init__(
        self,
        api_key: str,
        model_name: str = DEFAULT_MODEL,
        embedding_model: str = "",
    ):
        super().__init__(api_key, model_name)
        self.embedding_model = embedding_model
        raise NotImplementedError("xAI support is disabled. Configure GOOGLE_API_KEY and use Gemini instead.")
