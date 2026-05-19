import time
import json
import base64
import re
from typing import Dict, Any, Optional, List

from google import genai
from google.genai import types as genai_types

from core.extraction.json_repair import safe_json_loads
from core.adapters.base import BaseLLMAdapter, LLMResponse


class GeminiAdapter(BaseLLMAdapter):
    
    DEFAULT_MODEL = "gemini-2.5-flash"
    EMBEDDING_MODEL = "models/text-embedding-004"
    
    def __init__(
        self,
        api_key: str,
        model_name: str = DEFAULT_MODEL,
        embedding_model: str = EMBEDDING_MODEL
    ):
        super().__init__(api_key, model_name)
        self.embedding_model = embedding_model
        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name

    def _extract_text(self, response) -> str:
        try:
            text = response.text
            if text:
                return text
        except Exception:
            pass

        text_parts = []
        finish_reason = None
        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            if finish_reason is None:
                finish_reason = getattr(candidate, "finish_reason", None)
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) if content else None
            if parts:
                for part in parts:
                    part_text = getattr(part, "text", None)
                    if part_text:
                        text_parts.append(part_text)

        text = "".join(text_parts).strip()
        if text:
            return text

        if finish_reason is not None:
            try:
                finish_reason = finish_reason.name
            except Exception:
                finish_reason = str(finish_reason)
        raise RuntimeError(f"LLM response contained no text (finish_reason={finish_reason})")
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 8192
    ) -> LLMResponse:
        start_time = time.time()

        config = genai_types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system_prompt or "",
        )

        response = self._client.models.generate_content(
            model=self._model_name,
            contents=prompt,
            config=config,
        )

        latency_ms = (time.time() - start_time) * 1000

        input_tokens = 0
        output_tokens = 0
        if hasattr(response, 'usage_metadata'):
            input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0) or 0
            output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0) or 0

        text = self._extract_text(response)
        return LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model_name,
            latency_ms=latency_ms,
            raw_response=response
        )
    
    async def generate_with_vision(
        self,
        prompt: str,
        images: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7
    ) -> LLMResponse:
        start_time = time.time()

        parts = []
        if system_prompt:
            parts.append(genai_types.Part(text=system_prompt + "\n\n"))
        parts.append(genai_types.Part(text=prompt))

        for img in images:
            base64_data = img.get("base64", "")
            if "," in base64_data:
                base64_data = base64_data.split(",")[1]
            mime_type = img.get("mimeType", "image/png")
            image_bytes = base64.b64decode(base64_data)
            parts.append(genai_types.Part(
                inline_data=genai_types.Blob(mime_type=mime_type, data=image_bytes)
            ))

        config = genai_types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=8192,
        )

        response = self._client.models.generate_content(
            model=self._model_name,
            contents=parts,
            config=config,
        )

        latency_ms = (time.time() - start_time) * 1000

        input_tokens = 0
        output_tokens = 0
        if hasattr(response, 'usage_metadata'):
            input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0) or 0
            output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0) or 0

        text = self._extract_text(response)
        return LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model_name,
            latency_ms=latency_ms,
            raw_response=response
        )
    
    async def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        json_system = (system_prompt or "") + "\n\nRespond with valid JSON only. No markdown fences, no explanation. Just the JSON object/array."
        
        try:
            response = await self.generate(
                prompt=prompt,
                system_prompt=json_system,
                temperature=temperature,
                max_tokens=2048
            )
            
            text = response.text.strip()
            if not text:
                return {}

            # Attempt 1: Direct parse
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

            # Attempt 2: Extract from markdown fences or bare braces
            # We look for the first { or [ and then find the matching closing character
            try:
                # Find all potential JSON blocks
                matches = re.finditer(r'([\[\{])', text)
                for match in matches:
                    start_char = match.group(1)
                    end_char = '}' if start_char == '{' else ']'
                    start_pos = match.start()
                    
                    # Find matching closing bracket by counting nesting level
                    depth = 0
                    for i in range(start_pos, len(text)):
                        if text[i] == start_char:
                            depth += 1
                        elif text[i] == end_char:
                            depth -= 1
                            if depth == 0:
                                candidate = text[start_pos:i+1]
                                try:
                                    return safe_json_loads(candidate)
                                except Exception:
                                    continue # Try next block if this one fails
                
                # Fallback to repair the whole text if no perfect block found
                return safe_json_loads(text)
            except Exception:
                pass

            return {"error": "malformed_json", "raw": text[:200]}
            
        except Exception as e:
            print(f"Gemini generate_json failed: {e}")
            return {"error": str(e)}
    
    async def embed(self, text: str) -> List[float]:
        result = self._client.models.embed_content(
            model=self.embedding_model,
            contents=text,
        )
        return result.embeddings[0].values
