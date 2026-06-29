from typing import Any, Dict

from fastapi import APIRouter
from pydantic import BaseModel


router = APIRouter()


class HunterSearchRequest(BaseModel):
    industry: str
    location: str
    seed_query: str = ""
    target_count: int = 15


@router.post("/search")
async def hunter_search(request: HunterSearchRequest) -> Dict[str, Any]:
    from core.services._test_features.hunter.hunter_search_pipeline import HunterSearchPipeline

    pipeline = HunterSearchPipeline()
    result = await pipeline.run(
        industry=request.industry,
        location=request.location,
        seed_query=request.seed_query,
        target_count=request.target_count,
    )
    return result
