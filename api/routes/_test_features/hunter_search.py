import os
from typing import Any, Dict

from fastapi import APIRouter
from pydantic import BaseModel


router = APIRouter()

FALLBACK_WARNING = (
    "Hunter credits are low or unavailable. Leads shown are from normal search "
    "and may be less reliable."
)


class HunterSearchRequest(BaseModel):
    industry: str
    location: str
    seed_query: str = ""
    target_count: int = 15
    mode: str = "generic"


@router.post("/search")
async def hunter_search(request: HunterSearchRequest) -> Dict[str, Any]:
    from core.services._test_features.hunter.hunter_search_pipeline import HunterSearchPipeline

    request = _normalized_request(request)
    pipeline = HunterSearchPipeline()
    fallback_enabled = _env_bool("HUNTER_FALLBACK_ENABLED", True)
    preflight_enabled = _env_bool("HUNTER_CREDIT_PREFLIGHT_ENABLED", True)
    minimum_credits = _env_int("HUNTER_MIN_CREDITS_TO_RUN", 5)
    credit_status: Dict[str, Any] = {
        "ok": True,
        "available": None,
        "used": None,
        "raw": {},
        "reason": "hunter_credit_preflight_skipped",
    }

    if preflight_enabled:
        credit_status = await pipeline.get_credit_status()
        if not pipeline.has_enough_credits(credit_status, minimum=minimum_credits):
            return await _fallback_response(
                request,
                fallback_enabled=fallback_enabled,
                fallback_reason=credit_status.get("reason") or "hunter_low_credits",
                credit_status=credit_status,
            )

    result = await pipeline.run(
        industry=request.industry,
        location=request.location,
        seed_query=request.seed_query,
        target_count=request.target_count,
    )
    result.setdefault("metadata", {})
    result["metadata"]["hunter_credit_status"] = credit_status
    result["metadata"]["mode"] = _normalize_mode(request.mode)
    result["fallback_used"] = False
    result["warnings"] = result.get("warnings", [])

    credit_failure_reason = _hunter_credit_failure_reason(result)
    if credit_failure_reason:
        return await _fallback_response(
            request,
            fallback_enabled=fallback_enabled,
            fallback_reason=credit_failure_reason,
            credit_status=credit_status,
        )

    return result


async def _fallback_response(
    request: HunterSearchRequest,
    *,
    fallback_enabled: bool,
    fallback_reason: str,
    credit_status: Dict[str, Any],
) -> Dict[str, Any]:
    if not fallback_enabled:
        return {
            "success": False,
            "fallback_used": False,
            "leads": [],
            "steps": [
                {
                    "name": "hunter_credit_preflight",
                    "status": "failed",
                    "message": FALLBACK_WARNING,
                }
            ],
            "metadata": {
                "provider": "hunter",
                "fallback_enabled": False,
                "fallback_reason": fallback_reason,
                "mode": _normalize_mode(request.mode),
                "hunter_credit_status": credit_status,
            },
            "warnings": [],
            "errors": [fallback_reason],
        }

    pipeline_cls, pipeline_name = _fallback_pipeline_for_mode(request.mode)
    try:
        fallback_pipeline = pipeline_cls()
        prod_result = await fallback_pipeline.run(
            industry=request.industry,
            location=request.location,
            seed_query=request.seed_query,
            target_count=request.target_count,
        )
    except Exception as exc:
        return {
            "success": False,
            "fallback_used": True,
            "fallback_warning": FALLBACK_WARNING,
            "leads": [],
            "steps": [
                {
                    "name": "hunter_credit_preflight",
                    "status": "fallback",
                    "message": FALLBACK_WARNING,
                },
                {
                    "name": "normal_fallback",
                    "status": "failed",
                    "message": str(exc),
                },
            ],
            "metadata": {
                "provider": "normal_fallback",
                "fallback_used": True,
                "fallback_reason": fallback_reason,
                "fallback_pipeline": pipeline_name,
                "mode": _normalize_mode(request.mode),
                "hunter_credit_status": credit_status,
                "warning": FALLBACK_WARNING,
            },
            "warnings": [FALLBACK_WARNING],
            "errors": [str(exc)],
        }

    fallback_steps = [
        {
            "name": "hunter_credit_preflight",
            "status": "fallback",
            "message": FALLBACK_WARNING,
        },
        *(prod_result.get("steps") or []),
    ]
    return {
        "success": True,
        "leads": prod_result.get("leads", []),
        "steps": fallback_steps,
        "metadata": {
            **(prod_result.get("metadata") or {}),
            "provider": "normal_fallback",
            "fallback_used": True,
            "fallback_reason": fallback_reason,
            "fallback_pipeline": pipeline_name,
            "mode": _normalize_mode(request.mode),
            "hunter_credit_status": credit_status,
            "warning": FALLBACK_WARNING,
        },
        "errors": [],
        "warnings": [FALLBACK_WARNING],
        "fallback_used": True,
        "fallback_warning": FALLBACK_WARNING,
    }


def _fallback_pipeline_for_mode(mode: str):
    from core.services.production_lead_pipeline import ProductionLeadPipeline
    from core.services.velit.velit_lead_pipeline import VelitLeadPipeline

    normalized_mode = _normalize_mode(mode)
    if normalized_mode == "velit":
        return VelitLeadPipeline, "velit_lead_pipeline"
    return ProductionLeadPipeline, "production_lead_pipeline"


def _normalize_mode(mode: str) -> str:
    normalized = (mode or os.getenv("HUNTER_FALLBACK_MODE_DEFAULT", "generic")).strip().lower()
    return normalized if normalized in {"generic", "velit"} else "generic"


def _normalized_request(request: HunterSearchRequest) -> HunterSearchRequest:
    mode = _normalize_mode(request.mode)
    industry = (request.industry or "").strip()
    seed_query = (request.seed_query or "").strip()
    if mode == "velit":
        industry = "Van Upfitters / RV Builders"
        if not seed_query:
            seed_query = (
                f"Find premium van upfitters and RV builders around {request.location}. "
                "For each company, identify the best decision-maker and gather the strongest available contact path. "
                "Prioritize founders, owners, CEOs, operations leads, or other decision-makers. "
                "If direct email is unavailable, still keep the lead if you find LinkedIn, company phone, "
                "or a usable contact page. Search nearby cities or adjacent markets if the main location is thin."
            )
    return HunterSearchRequest(
        industry=industry,
        location=(request.location or "").strip(),
        seed_query=seed_query,
        target_count=request.target_count,
        mode=mode,
    )


def _hunter_credit_failure_reason(result: Dict[str, Any]) -> str:
    searchable_errors = []
    for value in result.get("errors") or []:
        searchable_errors.append(str(value))
    error = result.get("error")
    if error:
        searchable_errors.append(str(error))
    metadata = result.get("metadata") or {}
    for value in metadata.get("errors") or []:
        searchable_errors.append(str(value))

    joined = " ".join(searchable_errors)
    if "HUNTER_API_KEY is not configured" in joined:
        return "hunter_missing_api_key"
    for status_code in ("HTTP 401", "HTTP 403", "HTTP 429"):
        if status_code in joined:
            return f"hunter_credit_or_auth_failure_{status_code.lower().replace(' ', '_')}"
    return ""


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default
