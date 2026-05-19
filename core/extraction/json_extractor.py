from typing import Dict, Any, Optional

from core.adapters.base import BaseLLMAdapter


class JSONExtractor:
    def __init__(self, llm: BaseLLMAdapter):
        self.llm = llm

    async def extract(
        self,
        text: str,
        schema_description: str,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        prompt = (
            "Extract structured data from the text below.\n"
            "Return JSON that matches this schema description:\n"
            f"{schema_description}\n\n"
            f"Text:\n{text}"
        )
        return await self.llm.generate_json(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.2,
        )
