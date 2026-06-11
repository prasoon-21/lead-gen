from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    model: str
    latency_ms: float
    raw_response: Optional[Any] = None


class BaseLLMAdapter(ABC):
    
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.last_usage: Dict[str, Any] = {
            "model": model_name,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "latency_ms": 0.0,
            "mode": "",
        }

    def _set_last_usage(
        self,
        *,
        mode: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        model: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.last_usage = {
            "model": model or self.model_name,
            "input_tokens": int(input_tokens or 0),
            "output_tokens": int(output_tokens or 0),
            "total_tokens": int(input_tokens or 0) + int(output_tokens or 0),
            "latency_ms": float(latency_ms or 0.0),
            "mode": mode,
            **(extra or {}),
        }
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096
    ) -> LLMResponse:
        pass
    
    @abstractmethod
    async def generate_with_vision(
        self,
        prompt: str,
        images: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7
    ) -> LLMResponse:
        pass
    
    @abstractmethod
    async def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        pass
