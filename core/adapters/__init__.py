"""
LLM Adapters - Model-agnostic interfaces
"""

from core.adapters.base import BaseLLMAdapter, LLMResponse
from core.adapters.gemini import GeminiAdapter
from core.adapters.router import build_adapter_from_env

__all__ = [
    "BaseLLMAdapter",
    "LLMResponse",
    "GeminiAdapter",
    "build_adapter_from_env",
]
