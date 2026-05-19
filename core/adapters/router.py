import logging
import os
from typing import List, Optional

from core.adapters.base import BaseLLMAdapter, LLMResponse
from core.adapters.gemini import GeminiAdapter

logger = logging.getLogger("core.llm.router")


def _normalize_provider(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def _parse_provider_list(value: str) -> List[str]:
    if not value:
        return []
    return [_normalize_provider(part) for part in value.split(",") if part.strip()]


def _should_fallback(exc: Exception) -> bool:
    message = str(exc).lower()
    if isinstance(exc, NotImplementedError):
        return True
    signals = [
        "rate limit",
        "quota",
        "429",
        "timeout",
        "timed out",
        "deadline",
        "overloaded",
        "capacity",
        "context length",
        "token limit",
        "max tokens",
        "too many tokens",
        "service unavailable",
        "503",
        "bad gateway",
        "502",
        "not implemented",
        "finish_reason",
        "no text",
    ]
    return any(signal in message for signal in signals)


class LLMRouter(BaseLLMAdapter):
    def __init__(self, adapters: List[BaseLLMAdapter]):
        if not adapters:
            raise ValueError("LLMRouter requires at least one adapter")
        super().__init__(api_key="", model_name=adapters[0].model_name)
        self.adapters = adapters

    async def _with_fallback(self, method_name: str, *args, **kwargs):
        last_exc: Optional[Exception] = None
        for adapter in self.adapters:
            try:
                method = getattr(adapter, method_name)
                return await method(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                if _should_fallback(exc):
                    logger.warning(
                        "LLM fallback triggered for %s (%s): %s",
                        method_name,
                        adapter.model_name,
                        exc,
                    )
                    continue
                raise
        if last_exc:
            raise last_exc
        raise RuntimeError("LLMRouter failed without an exception")

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        return await self._with_fallback(
            "generate",
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def generate_with_vision(
        self,
        prompt: str,
        images: List[dict],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
    ) -> LLMResponse:
        return await self._with_fallback(
            "generate_with_vision",
            prompt=prompt,
            images=images,
            system_prompt=system_prompt,
            temperature=temperature,
        )

    async def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
    ) -> dict:
        return await self._with_fallback(
            "generate_json",
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
        )

    async def embed(self, text: str) -> List[float]:
        return await self._with_fallback("embed", text=text)


def _build_gemini_adapter(
    model_name: Optional[str] = None,
    embedding_model: Optional[str] = None,
) -> Optional[BaseLLMAdapter]:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    model_name = model_name or os.getenv("GEMINI_MODEL", GeminiAdapter.DEFAULT_MODEL)
    embedding_model = embedding_model or os.getenv(
        "GEMINI_EMBEDDING_MODEL",
        GeminiAdapter.EMBEDDING_MODEL,
    )
    return GeminiAdapter(api_key=api_key, model_name=model_name, embedding_model=embedding_model)


def _build_openai_adapter(
    model_name: Optional[str] = None,
    embedding_model: Optional[str] = None,
) -> Optional[BaseLLMAdapter]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from core.adapters.openai import OpenAIAdapter
    except ImportError as exc:
        logger.warning("OpenAI adapter unavailable: %s", exc)
        return None

    model_name = model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    embedding_model = embedding_model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    return OpenAIAdapter(api_key=api_key, model_name=model_name, embedding_model=embedding_model)


def _build_anthropic_adapter(
    model_name: Optional[str] = None,
) -> Optional[BaseLLMAdapter]:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        from core.adapters.anthropic import AnthropicAdapter
    except ImportError as exc:
        logger.warning("Anthropic adapter unavailable: %s", exc)
        return None

    model_name = model_name or os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
    return AnthropicAdapter(api_key=api_key, model_name=model_name)


def build_adapter_from_env(
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
    embedding_model_override: Optional[str] = None,
) -> Optional[BaseLLMAdapter]:
    primary = _normalize_provider(provider_override or os.getenv("LLM_PROVIDER", "gemini"))
    fallbacks = _parse_provider_list(os.getenv("LLM_FALLBACKS", ""))
    providers = [primary] + [p for p in fallbacks if p and p != primary]

    adapters: List[BaseLLMAdapter] = []
    for provider in providers:
        adapter = None
        if provider == "gemini":
            adapter = _build_gemini_adapter(
                model_name=model_override if provider == primary else None,
                embedding_model=embedding_model_override if provider == primary else None,
            )
        elif provider == "openai":
            adapter = _build_openai_adapter(
                model_name=model_override if provider == primary else None,
                embedding_model=embedding_model_override if provider == primary else None,
            )
        elif provider == "anthropic":
            adapter = _build_anthropic_adapter(
                model_name=model_override if provider == primary else None,
            )
        else:
            logger.warning("Unknown LLM provider: %s", provider)
            continue

        if adapter:
            adapters.append(adapter)
        else:
            logger.warning("LLM provider not configured: %s", provider)

    if not adapters:
        return None
    if len(adapters) == 1:
        return adapters[0]
    return LLMRouter(adapters)
