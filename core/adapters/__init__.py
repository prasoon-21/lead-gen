"""
LLM Adapters - Model-agnostic interfaces
"""

from core.adapters.base import BaseLLMAdapter, LLMResponse
from core.adapters.gemini import GeminiAdapter
from core.adapters.router import LLMRouter, build_adapter_from_env

try:  # optional providers
    from core.adapters.openai import OpenAIAdapter
except ImportError:  # pragma: no cover
    OpenAIAdapter = None

try:  # optional providers
    from core.adapters.anthropic import AnthropicAdapter
except ImportError:  # pragma: no cover
    AnthropicAdapter = None

__all__ = [
    "BaseLLMAdapter",
    "LLMResponse",
    "GeminiAdapter",
    "OpenAIAdapter",
    "AnthropicAdapter",
    "LLMRouter",
    "build_adapter_from_env",
]
