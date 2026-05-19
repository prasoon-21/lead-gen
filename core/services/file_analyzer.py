import base64
from pathlib import Path
from typing import Dict, Any, Optional, List
import json

from core.adapters.base import BaseLLMAdapter, LLMResponse


class FileAnalyzer:
    
    SUPPORTED_MIME_TYPES = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".pdf": "application/pdf"
    }
    
    def __init__(self, adapter: BaseLLMAdapter):
        self.adapter = adapter
    
    async def analyze(
        self,
        document: Dict[str, Any],
        instruction: str,
        user_query: Optional[str] = None
    ) -> LLMResponse:
        prompt = self._build_analysis_prompt(instruction, user_query)
        
        images = [document]
        
        return await self.adapter.generate_with_vision(
            prompt=prompt,
            images=images
        )
    
    async def extract_text(self, document: Dict[str, Any]) -> str:
        instruction = "Extract all text from this document. Preserve structure and formatting."
        response = await self.analyze(document, instruction)
        return response.text
    
    async def extract_math(self, document: Dict[str, Any]) -> str:
        instruction = """Extract mathematical content from this image.
Use proper symbols: θ, √, ², π, →, ∫, Σ, ∞, ≤, ≥, ≠
Format equations clearly. Preserve all numbers and expressions."""
        
        response = await self.analyze(document, instruction)
        return response.text
    
    async def analyze_for_tutoring(
        self,
        document: Dict[str, Any],
        user_query: str
    ) -> str:
        instruction = """Analyze this document uploaded by a student.

SAFETY CHECK FIRST:
- 18+ adult content = REJECT
- Violence, gore, abuse = REJECT
- Hate speech, offensive content = REJECT
- Inappropriate or harmful content = REJECT

If INAPPROPRIATE/UNSAFE:
- Say: "I cannot process this type of content. Please only share educational materials."
- Do NOT describe or analyze the content

Then check if EDUCATIONAL:
- Math problems, science diagrams, textbook pages = EDUCATIONAL
- Random images, memes, selfies, unrelated content = NOT EDUCATIONAL

If EDUCATIONAL:
- Extract the content clearly
- Use proper math symbols (θ, √, ², π, →)
- Identify: equations, questions, diagrams, formulas
- Prepare to help solve/explain

If NOT EDUCATIONAL:
- Politely say: "This doesn't seem to be related to your studies. Please share something from your textbook or homework so I can help you learn!"
- Do NOT analyze or describe the irrelevant content"""
        
        response = await self.analyze(document, instruction, user_query)
        return response.text
    
    async def analyze_from_path(
        self,
        file_path: str,
        instruction: str
    ) -> LLMResponse:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(path, "rb") as f:
            data = f.read()
        
        base64_data = base64.b64encode(data).decode("utf-8")
        mime_type = self.SUPPORTED_MIME_TYPES.get(path.suffix.lower(), "image/png")
        
        document = {
            "base64": base64_data,
            "mimeType": mime_type,
            "filename": path.name
        }
        
        return await self.analyze(document, instruction)
    
    def _build_analysis_prompt(
        self,
        instruction: str,
        user_query: Optional[str] = None
    ) -> str:
        if user_query:
            return f"""{instruction}

User's question: "{user_query}"

Analyze the document and respond to the user's question."""
        
        return instruction
