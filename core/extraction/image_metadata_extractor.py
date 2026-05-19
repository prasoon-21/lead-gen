from typing import Dict, Any, Optional

from core.adapters.base import BaseLLMAdapter


class ImageMetadataExtractor:
    def __init__(self, llm: BaseLLMAdapter):
        self.llm = llm

    async def extract(
        self,
        text: str,
        topic: Optional[str] = None,
        level: Optional[str] = None,
        board: Optional[str] = None,
    ) -> Dict[str, Any]:
        prompt = (
            "Create JSON for an educational image based on the text below.\n"
            "Return ONLY valid JSON with these keys:\n"
            "- image_description\n"
            "- image_caption\n"
            "- image_generation_prompt\n\n"
            f"Topic: {topic or 'General'}\n"
            f"Level: {level or 'N/A'}\n"
            f"Board: {board or 'N/A'}\n\n"
            f"Text:\n{text}"
        )
        return await self.llm.generate_json(prompt=prompt, temperature=0.3)
