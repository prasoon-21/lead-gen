from core.tools.base import BaseTool, ToolContext, ToolResult
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
            
        result = {
            "email_verification": verification,
            "spam_analysis": spam_analysis,
            "overall_status": "Passed" if verification["status"] == "valid" and (not spam_analysis or spam_analysis["score"] >= 7) else "Needs Review"
        }
        
        return result
