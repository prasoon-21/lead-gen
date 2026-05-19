import json
from typing import Dict, Any, Optional
import base64

from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai


class ImageGenerator:
    
    IMAGE_MODEL = "gemini-2.5-flash-image"
    PROMPT_MODEL = "gemini-2.5-flash-lite"
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self._image_model = genai.GenerativeModel(self.IMAGE_MODEL)
        self._prompt_model = genai.GenerativeModel(self.PROMPT_MODEL)
    
    async def generate_image_metadata(
        self,
        query: str,
        ai_output: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        prompt = f"""You generate structured image generation metadata based on the user query and the agent's answer.
Your task is to infer the most appropriate educational or explanatory image that best supports understanding.

Query: {query}
AI Output: {ai_output}

Generate metadata for an educational image. Output ONLY valid JSON:

{{
  "image_prompt": "A clear, educational, high-resolution diagram of...",
  "image_name": "lowercase-hyphen-separated-name",
  "image_description": "Detailed description for embedding...",
  "image_caption": "Short one-line caption",
  "topic": "{context.get('subject', 'General')}",
  "subtopic": "Specific visual element or concept"
}}

Rules:
- image_prompt must be detailed for AI image generation
- image_name must be lowercase, hyphen-separated, no extension
- image_description must be detailed for vector embedding
- image_caption must be a short one-liner"""
        
        config = genai.GenerationConfig(temperature=0.3, max_output_tokens=500)
        response = self._prompt_model.generate_content(prompt, generation_config=config)
        
        text = response.text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
        
        try:
            metadata = json.loads(text)
        except json.JSONDecodeError:
            metadata = {
                "image_prompt": f"Educational diagram about {query}",
                "image_name": query.lower().replace(" ", "-")[:30],
                "image_description": query,
                "image_caption": query,
                "topic": context.get("subject", "General"),
                "subtopic": "Concept"
            }
        
        metadata["level"] = context.get("class", "Class_10")
        metadata["subject"] = context.get("subject", "General")
        metadata["board"] = context.get("board", "CBSE")
        metadata["medium"] = context.get("medium", "English")
        metadata["course_type"] = context.get("institute_type", "School")
        metadata["content_type"] = "image"
        
        return metadata
    
    async def generate_image(self, image_prompt: str) -> Optional[bytes]:
        try:
            generation_config = None
            try:
                generation_config = genai.GenerationConfig(
                    response_modalities=["image", "text"]
                )
            except TypeError:
                generation_config = None

            if generation_config:
                response = self._image_model.generate_content(
                    image_prompt,
                    generation_config=generation_config
                )
            else:
                response = self._image_model.generate_content(image_prompt)
            
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    return part.inline_data.data
            
            return None
            
        except Exception as e:
            print(f"   ⚠️ Image generation failed: {e}")
            return None
    
    async def generate_full(
        self,
        query: str,
        ai_output: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        print(f"\n   🎨 STEP: GENERATING IMAGE METADATA...")
        metadata = await self.generate_image_metadata(query, ai_output, context)
        print(f"   ✅ Prompt: {metadata.get('image_prompt', '')[:80]}...")
        print(f"   ✅ Name: {metadata.get('image_name', '')}")
        
        print(f"\n   🖼️ STEP: GENERATING IMAGE WITH GEMINI...")
        image_bytes = await self.generate_image(metadata.get("image_prompt", ""))
        
        if image_bytes:
            print(f"   ✅ Image generated: {len(image_bytes)} bytes")
        else:
            print(f"   ⚠️ Image generation failed")
        
        return {
            "metadata": metadata,
            "image_bytes": image_bytes
        }
