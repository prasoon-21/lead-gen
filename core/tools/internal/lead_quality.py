from core.tools.base import BaseTool, ToolContext
from core.services.lead_quality_service import score_lead
from core.utils.email_verify import verify_email_legitimacy
from core.utils.spam_scorer import calculate_spam_score
from typing import Any, Dict

class LeadQualityTool(BaseTool):
    """
    Verifies lead email legitimacy and evaluates outreach content for spam triggers.
    """
    name = "lead_quality_check"
    description = (
        "Verifies an email address exists (legitimacy) and calculates a spam score "
        "for the provided subject and body text."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address to verify."
            },
            "subject": {
                "type": "string",
                "description": "The subject line of the outreach email."
            },
            "body": {
                "type": "string",
                "description": "The body text of the outreach email."
            }
        },
        "required": ["email"]
    }

    async def run(self, arguments: Dict[str, Any], context: ToolContext) -> Dict[str, Any]:
        email = arguments.get("email")
        subject = arguments.get("subject", "")
        body = arguments.get("body", "")
        
        verification = verify_email_legitimacy(email)
        
        # Only calculate spam score if subject or body is provided
        spam_analysis = {}
        if subject or body:
            spam_analysis = calculate_spam_score(subject, body)

        lead_stub = {
            "contact_email": email,
            "contact_person_name": arguments.get("contact_person_name", ""),
            "contact_phone": arguments.get("contact_phone", ""),
            "linkedin_url": arguments.get("linkedin_url", ""),
            "contact_page": arguments.get("contact_page", ""),
            "company_website": arguments.get("company_website", ""),
            "company_name": arguments.get("company_name", ""),
            "source": arguments.get("source", "quality_check"),
        }
        quality = score_lead(lead_stub, verification=verification)

        result = {
            "email_verification": verification,
            "spam_analysis": spam_analysis,
            "lead_quality": quality,
            "overall_status": "Passed" if verification["status"] == "valid" and (not spam_analysis or spam_analysis["score"] >= 7) else "Needs Review"
        }
        
        return result
