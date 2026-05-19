import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from api.state import get_state
from core.services.image_generator import ImageGenerator
from core.services.r2_uploader import R2Uploader


router = APIRouter()

_image_generator = None
_r2_uploader = None


def get_image_generator() -> ImageGenerator:
    global _image_generator
    if _image_generator is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        _image_generator = ImageGenerator(api_key=api_key)
    return _image_generator


def get_r2_uploader() -> R2Uploader:
    global _r2_uploader
    if _r2_uploader is None:
        _r2_uploader = R2Uploader()
    return _r2_uploader


class ImageRequest(BaseModel):
    message: str
    output: str
    student_id: Optional[str] = None
    subject: Optional[str] = None
    level: Optional[str] = "Class_10"
    board: Optional[str] = "CBSE"


class ImageResponse(BaseModel):
    success: bool
    image_url: Optional[str] = None
    error: Optional[str] = None


@router.post("/generate", response_model=ImageResponse)
async def generate_image(request: ImageRequest):
    try:
        print("\n" + "="*60)
        print(f"🎨 IMAGE GENERATION REQUEST")
        print(f"   Query: {request.message[:80]}...")
        print("="*60)
        
        context = {
            "class": request.level,
            "subject": request.subject or "General",
            "board": request.board,
            "medium": "English",
            "institute_type": "School"
        }
        
        generator = get_image_generator()
        result = await generator.generate_full(
            query=request.message,
            ai_output=request.output,
            context=context
        )
        
        metadata = result["metadata"]
        image_bytes = result["image_bytes"]
        
        if not image_bytes:
            return ImageResponse(
                success=False,
                error="Image generation failed"
            )
        
        print(f"\n   📤 STEP: UPLOADING TO R2...")
        uploader = get_r2_uploader()
        image_url = await uploader.upload(
            image_bytes=image_bytes,
            filename=metadata.get("image_name", "generated-image")
        )
        
        if not image_url:
            return ImageResponse(
                success=False,
                error="Upload failed"
            )
        
        metadata["image_url"] = image_url
        
        print(f"\n   📚 STEP: STORING IN QDRANT...")
        state = get_state()
        if state.retriever:
            await state.retriever.upsert_image(metadata)
        
        print(f"\n✅ IMAGE GENERATION COMPLETE")
        print(f"   URL: {image_url}")
        print("="*60 + "\n")
        
        return ImageResponse(
            success=True,
            image_url=image_url
        )
        
    except Exception as e:
        print(f"   ⚠️ Error: {e}")
        return ImageResponse(
            success=False,
            error=str(e)
        )
