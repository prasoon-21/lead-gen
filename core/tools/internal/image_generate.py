import os
from typing import Any, Dict

from core.tools.base import BaseTool, ToolContext
from core.services.image_generator import ImageGenerator
from core.services.r2_uploader import R2Uploader


class ImageGenerateTool(BaseTool):
    """Generates an educational image and uploads it to R2 storage.

    Checks for an existing image URL first (unless force_generate=True).
    Uses ImageGenerator (Gemini) + R2Uploader directly — no workflow dependency.
    """

    name = "image_generate"
    description = "Generate an educational image and return its public URL."
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The user's question or topic"},
            "ai_output": {"type": "string", "description": "The AI's text response to pair with the image"},
            "context": {"type": "object", "description": "Student context (subject, class, board, etc.)"},
            "existing_image_url": {"type": "string", "description": "If set and force_generate is false, return this directly"},
            "force_generate": {"type": "boolean", "description": "Force new image generation even if URL exists"},
        },
        "required": ["query", "ai_output", "context"],
    }
    timeout_seconds = 80

    async def run(self, arguments: Dict[str, Any], context: ToolContext) -> Dict[str, Any]:
        existing_url = arguments.get("existing_image_url")
        force = bool(arguments.get("force_generate", False))

        # Return existing image if available and not forcing regeneration
        if existing_url and not force:
            return {
                "image_url": existing_url,
                "image_generated": False,
                "source": "existing",
            }

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return {"image_url": None, "image_generated": False, "error": "GOOGLE_API_KEY not set"}

        generator = ImageGenerator(api_key=api_key)
        result = await generator.generate_full(
            query=arguments["query"],
            ai_output=arguments["ai_output"],
            context=arguments["context"],
        )

        image_bytes = result.get("image_bytes")
        metadata = result.get("metadata", {})

        if not image_bytes:
            return {"image_url": None, "image_generated": False, "error": "Image generation returned no bytes"}

        uploader = R2Uploader()
        image_url = await uploader.upload(
            image_bytes=image_bytes,
            filename=metadata.get("image_name", "generated-image"),
        )

        if not image_url:
            return {"image_url": None, "image_generated": False, "error": "R2 upload failed"}

        # Optionally store in Qdrant if retriever is available
        retriever = context.resources.get("retriever")
        if retriever:
            try:
                metadata["image_url"] = image_url
                await retriever.upsert_image(metadata)
            except Exception:
                pass  # Qdrant upsert failure is non-fatal

        return {
            "image_url": image_url,
            "image_generated": True,
            "image_name": metadata.get("image_name"),
            "image_caption": metadata.get("image_caption"),
        }
