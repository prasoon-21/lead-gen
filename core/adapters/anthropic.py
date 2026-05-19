import json
import time
from typing import Dict, Any, Optional, List

from core.adapters.base import BaseLLMAdapter, LLMResponse

try:
    from anthropic import Anthropic
except ImportError:  # pragma: no cover - optional dependency
    Anthropic = None


class AnthropicAdapter(BaseLLMAdapter):
    def __init__(self, api_key: str, model_name: str):
        if Anthropic is None:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
        super().__init__(api_key, model_name)
        self.client = Anthropic(api_key=api_key)

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        start_time = time.time()
        response = self.client.messages.create(
            model=self.model_name,
            system=system_prompt or "",
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )

        latency_ms = (time.time() - start_time) * 1000
        text = ""
        if response.content:
            text = "".join(block.text for block in response.content if hasattr(block, "text"))
        return LLMResponse(
            text=text,
            input_tokens=getattr(response.usage, "input_tokens", 0) or 0,
            output_tokens=getattr(response.usage, "output_tokens", 0) or 0,
            model=self.model_name,
            latency_ms=latency_ms,
            raw_response=response,
        )

    async def generate_with_vision(
        self,
        prompt: str,
        images: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
    ) -> LLMResponse:
        start_time = time.time()
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img in images:
            base64_data = img.get("base64", "")
            if "," in base64_data:
                base64_data = base64_data.split(",")[1]
            mime_type = img.get("mimeType", "image/png")
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": base64_data,
                    },
                }
            )

        response = self.client.messages.create(
            model=self.model_name,
            system=system_prompt or "",
            max_tokens=4096,
            temperature=temperature,
            messages=[{"role": "user", "content": content}],
        )

        latency_ms = (time.time() - start_time) * 1000
        text = ""
        if response.content:
            text = "".join(block.text for block in response.content if hasattr(block, "text"))
        return LLMResponse(
            text=text,
            input_tokens=getattr(response.usage, "input_tokens", 0) or 0,
            output_tokens=getattr(response.usage, "output_tokens", 0) or 0,
            model=self.model_name,
            latency_ms=latency_ms,
            raw_response=response,
        )

    async def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
    ) -> Dict[str, Any]:
        json_system = (system_prompt or "") + "\n\nReturn valid JSON only."
        response = await self.generate(
            prompt=prompt,
            system_prompt=json_system,
            temperature=temperature,
            max_tokens=2048,
        )
        text = response.text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
        return json.loads(text)

    async def embed(self, text: str) -> List[float]:
        raise NotImplementedError("Anthropic embeddings are not configured yet.")
