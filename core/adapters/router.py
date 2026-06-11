import logging
import os
from typing import Optional

from core.adapters.base import BaseLLMAdapter
from core.adapters.gemini import GeminiAdapter

logger = logging.getLogger("core.llm.router")


def _normalize_provider(name: str) -> str:
    return name.strip().lower().replace("-", "_")


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


def build_adapter_from_env(
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
    embedding_model_override: Optional[str] = None,
) -> Optional[BaseLLMAdapter]:
    requested_provider = _normalize_provider(provider_override or os.getenv("LLM_PROVIDER", "gemini"))
    configured_fallbacks = os.getenv("LLM_FALLBACKS", "").strip()

    if requested_provider != "gemini":
        logger.warning(
            "Non-Gemini provider '%s' was requested, but this project is locked to Gemini. Using Gemini instead.",
            requested_provider,
        )
    if configured_fallbacks:
        logger.warning(
            "Ignoring LLM_FALLBACKS=%s because only Gemini is enabled for this project.",
            configured_fallbacks,
        )

    adapter = _build_gemini_adapter(
        model_name=model_override,
        embedding_model=embedding_model_override,
    )
    if not adapter:
        logger.warning("Gemini adapter not configured. Set GOOGLE_API_KEY or GEMINI_API_KEY.")
    return adapter
