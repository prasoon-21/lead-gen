import time
import json
import base64
import re
import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable, Awaitable

from google.api_core import exceptions as google_exceptions
from google import genai
from google.genai import types as genai_types

from core.extraction.json_repair import safe_json_loads
from core.adapters.base import BaseLLMAdapter, LLMResponse

logger = logging.getLogger(__name__)


class GeminiAdapter(BaseLLMAdapter):
    DEFAULT_MODEL = "gemini-1.5-flash"
    EMBEDDING_MODEL = "models/text-embedding-004"

    def __init__(
        self,
        api_key: str,
        model_name: str = DEFAULT_MODEL,
        embedding_model: str = EMBEDDING_MODEL,
    ):
        super().__init__(api_key, model_name)
        self.embedding_model = embedding_model
        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name

    async def _with_retry(
        self, func: Callable[..., Awaitable[Any]], *args, **kwargs
    ) -> Any:
        """
        Retry an async function with exponential backoff.
        """
        max_retries = 5
        base_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                return await func(*args, **kwargs)
            except (
                google_exceptions.ResourceExhausted,
                google_exceptions.ServiceUnavailable,
                google_exceptions.DeadlineExceeded,
            ) as e:
                if attempt == max_retries - 1:
                    logger.error("Gemini API max retries reached. Failing permanently.")
                    raise e  # Raise the original, helpful Google API exception instead of generic string
                
                delay = base_delay * (2**attempt)
                logger.warning(
                    "Gemini API transient error: %s. Retrying in %d seconds (Attempt %d/%d)...", 
                    e, delay, attempt + 1, max_retries
                )
                await asyncio.sleep(delay)

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
        raise RuntimeError(
            f"LLM response contained no text (finish_reason={finish_reason})"
        )

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 8192,
        response_mime_type: Optional[str] = None,  # Added to support native structural modes
    ) -> LLMResponse:
        start_time = time.time()

        config = genai_types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system_prompt,
            response_mime_type=response_mime_type,
        )

        response = await self._with_retry(
            self._client.aio.models.generate_content,
            model=self._model_name,
            contents=prompt,
            config=config,
        )

        latency_ms = (time.time() - start_time) * 1000

        input_tokens = 0
        output_tokens = 0
        if hasattr(response, "usage_metadata"):
            input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            output_tokens = (
                getattr(response.usage_metadata, "candidates_token_count", 0) or 0
            )

        text = self._extract_text(response)
        self._set_last_usage(
            mode="generate",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            model=self.model_name,
            extra={"response_mime_type": response_mime_type or "text/plain"},
        )
        return LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
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

        parts = []
        parts.append(genai_types.Part(text=prompt))

        for img in images:
            base64_data = img.get("base64", "")
            if "," in base64_data:
                base64_data = base64_data.split(",")[1]
            mime_type = img.get("mimeType", "image/png")
            image_bytes = base64.b64decode(base64_data)
            parts.append(
                genai_types.Part(
                    inline_data=genai_types.Blob(mime_type=mime_type, data=image_bytes)
                )
            )

        config = genai_types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=8192,
            system_instruction=system_prompt,
        )

        response = await self._with_retry(
            self._client.aio.models.generate_content,
            model=self._model_name,
            contents=parts,
            config=config,
        )

        latency_ms = (time.time() - start_time) * 1000

        input_tokens = 0
        output_tokens = 0
        if hasattr(response, "usage_metadata"):
            input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            output_tokens = (
                getattr(response.usage_metadata, "candidates_token_count", 0) or 0
            )

        text = self._extract_text(response)
        self._set_last_usage(
            mode="vision",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            model=self.model_name,
            extra={"image_count": len(images or [])},
        )
        return LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
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
        # Using native SDK application/json constraint guarantees structural validity
        response = await self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=8192,
            response_mime_type="application/json",
        )

        text = response.text.strip()
        if not text:
            return {}

        # Attempt 1: Direct structural parse
        try:
            parsed = json.loads(text)
            self.last_usage["mode"] = "generate_json"
            self.last_usage["parse_status"] = "direct_json"
            return parsed
        except json.JSONDecodeError:
            pass

        # Attempt 2: Fallback extraction mechanism if schema structure got modified downstream
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
                            candidate = text[start_pos : i + 1]
                            try:
                                parsed = safe_json_loads(candidate)
                                self.last_usage["mode"] = "generate_json"
                                self.last_usage["parse_status"] = "extracted_json"
                                return parsed
                            except Exception:
                                continue

            parsed = safe_json_loads(text)
            self.last_usage["mode"] = "generate_json"
            self.last_usage["parse_status"] = "repaired_json"
            return parsed
        except Exception:
            pass

        self.last_usage["mode"] = "generate_json"
        self.last_usage["parse_status"] = "malformed_json"
        return {"error": "malformed_json", "raw": text[:200]}

    async def embed(self, text: str) -> List[float]:
        response = await self._with_retry(
            self._client.aio.models.embed_content,
            model=self.embedding_model,
            contents=text,
        )
        if response.embeddings:
            return response.embeddings[0].values
        return []
