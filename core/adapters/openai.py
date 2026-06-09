import json
import re
import time
from typing import Dict, Any, Optional, List

from core.adapters.base import BaseLLMAdapter, LLMResponse
from core.extraction.json_repair import safe_json_loads

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None


class OpenAIAdapter(BaseLLMAdapter):
    def __init__(
        self,
        api_key: str,
        model_name: str,
        embedding_model: str,
        base_url: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
    ):
        if OpenAI is None:
            raise ImportError("openai package not installed. Run: pip install openai")
        api_key = self._sanitize_api_key(api_key)
        super().__init__(api_key, model_name)
        self.embedding_model = embedding_model
        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        if default_headers:
            client_kwargs["default_headers"] = default_headers
        self.client = OpenAI(**client_kwargs)

    @staticmethod
    def _sanitize_api_key(value: str) -> str:
        cleaned = value.strip()
        cleaned = cleaned.encode("ascii", "ignore").decode("ascii")
        return cleaned

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        start_time = time.time()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        latency_ms = (time.time() - start_time) * 1000
        text = response.choices[0].message.content if response.choices else ""
        usage = response.usage or {}
        return LLMResponse(
            text=text or "",
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
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
            data_url = f"data:{mime_type};base64,{base64_data}"
            content.append({"type": "image_url", "image_url": {"url": data_url}})

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=4096,
        )

        latency_ms = (time.time() - start_time) * 1000
        text = response.choices[0].message.content if response.choices else ""
        usage = response.usage or {}
        return LLMResponse(
            text=text or "",
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
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
        if not text:
            return {}

        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        try:
            matches = re.finditer(r"([\[\{])", text)
            for match in matches:
                start_char = match.group(1)
                end_char = "}" if start_char == "{" else "]"
                start_pos = match.start()

                depth = 0
                for i in range(start_pos, len(text)):
                    if text[i] == start_char:
                        depth += 1
                    elif text[i] == end_char:
                        depth -= 1
                        if depth == 0:
                            candidate = text[start_pos:i + 1]
                            try:
                                return safe_json_loads(candidate)
                            except Exception:
                                continue

            return safe_json_loads(text)
        except Exception:
            return {"error": "malformed_json", "raw": text[:500]}

    async def embed(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )
        return response.data[0].embedding
