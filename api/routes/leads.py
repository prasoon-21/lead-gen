from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from typing import List, Dict, Any
import inspect
import json
import os
from api.state import get_state
from core.services.lead_quality_service import score_lead
from core.services.lead_service import LeadService
from core.utils.email_verify import verify_email_legitimacy
from core.utils.spam_scorer import calculate_spam_score

router = APIRouter()

class LeadExportRequest(BaseModel):
    leads: List[Dict[str, Any]]
    mode: str = "generic"

class EmailVerifyRequest(BaseModel):
    email: str
    subject: str = ""
    body: str = ""
    lead: Dict[str, Any] = Field(default_factory=dict)

class LinkedInSessionRequest(BaseModel):
    cookie_value: str

@router.post("/export")
async def export_leads(request: LeadExportRequest):
    state = get_state()
    if not state.todo_store:
        raise HTTPException(status_code=500, detail="Google Sheets storage not configured")
    
    service = LeadService(state.todo_store)
    result = await service.export_leads(request.leads, mode=request.mode)
    return result

@router.post("/verify")
async def verify_lead(request: EmailVerifyRequest):
    try:
        verification = await run_in_threadpool(verify_email_legitimacy, request.email)
    except Exception as exc:
        verification = {
            "email": request.email,
            "status": "error",
            "confidence": 0,
            "message": f"Email verification failed internally: {type(exc).__name__}",
            "checks": [
                {
                    "name": "verification_runtime",
                    "status": "failed",
                    "detail": str(exc),
                }
            ],
        }
    lead_quality = {}
    if request.lead:
        lead_stub = dict(request.lead)
        lead_stub["contact_email"] = request.email
        lead_stub["verification_status"] = verification.get("status", "")
        lead_stub["verification_message"] = verification.get("message", "")
        lead_quality = score_lead(lead_stub, verification=verification)

    spam_analysis = {}
    if request.subject or request.body:
        spam_analysis = calculate_spam_score(request.subject, request.body)
    
    return {
        "verification": verification,
        "lead_quality": lead_quality,
        "spam_analysis": spam_analysis
    }

@router.get("/existing")
async def get_existing_leads(mode: str = "generic"):
    state = get_state()
    if not state.todo_store:
        return {"companies": []}
    
    service = LeadService(state.todo_store)
    try:
        companies = await service.get_existing_companies(mode=mode)
        return {"companies": companies}
    except Exception as e:
        import logging
        logging.getLogger("api.leads").warning(f"Failed to fetch existing companies: {e}")
        return {"companies": []}


@router.get("/details")
async def get_lead_details():
    state = get_state()
    if not state.todo_store:
        raise HTTPException(status_code=500, detail="Google Sheets storage not configured")

    list_leads = state.todo_store.list_leads
    if inspect.iscoroutinefunction(list_leads):
        return await list_leads()
    return await run_in_threadpool(list_leads)

@router.post("/linkedin_session")
async def update_linkedin_session(request: LinkedInSessionRequest):
    session_path = os.path.join(os.getcwd(), "linkedin_session.json")
    if os.getenv("VERCEL") == "1":
        session_path = os.path.join("/tmp", "linkedin_session.json")
    
    # Standard Playwright storage state format with the user-provided li_at cookie
    session_data = {
        "cookies": [
            {
                "name": "li_at",
                "value": request.cookie_value.strip(),
                "domain": ".linkedin.com",
                "path": "/",
                "expires": 1810667126.159561,
                "httpOnly": True,
                "secure": True,
                "sameSite": "None"
            },
            {
                "name": "JSESSIONID",
                "value": "ajax:0000000000000000000",
                "domain": ".linkedin.com",
                "path": "/",
                "expires": 1810667126.159561,
                "httpOnly": False,
                "secure": True,
                "sameSite": "None"
            }
        ],
        "origins": []
    }
    
    try:
        with open(session_path, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2)
        return {"status": "success", "message": "LinkedIn session updated successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save session: {str(e)}")

@router.get("/linkedin_session/status")
async def get_linkedin_session_status():
    session_path = os.path.join(os.getcwd(), "linkedin_session.json")
    if os.getenv("VERCEL") == "1":
        session_path = os.path.join("/tmp", "linkedin_session.json")
    if not os.path.exists(session_path):
        return {"status": "inactive", "message": "No saved session found"}
        
    try:
        with open(session_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            cookies = data.get("cookies", [])
            li_at = next((c for c in cookies if c.get("name") == "li_at"), None)
            if li_at and li_at.get("value"):
                return {"status": "active", "message": "Session is active"}
    except Exception:
        pass
        
    return {"status": "inactive", "message": "Invalid or expired session"}

