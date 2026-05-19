from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "agentic_core",
        "version": "1.0.0"
    }


@router.get("/")
async def root():
    return {
        "message": "Agentic Core API",
        "docs": "/docs"
    }
