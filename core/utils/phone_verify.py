from __future__ import annotations

from typing import Any, Dict

from core.utils.phone_quality import normalize_phone_candidate


def verify_phone_legitimacy(value: Any, *, region: str = "US") -> Dict[str, Any]:
    raw = str(value or "").strip()
    if not raw or raw.lower() in {"-", "--", "n/a", "na", "none", "null", "not found"}:
        return {
            "status": "no_phone_found",
            "confidence": 0,
            "message": "No phone number was found in the selected cell.",
            "raw": raw,
        }

    candidate = normalize_phone_candidate(
        raw,
        source_type="excel_upload",
        source_url="",
        context="phone number from uploaded Excel cell",
        region=region,
    )
    if not candidate:
        return {
            "status": "invalid",
            "confidence": 0,
            "message": "This does not look like a valid phone number format.",
            "raw": raw,
        }

    validation_status = str(candidate.get("validation_status") or "").lower()
    status = "valid" if validation_status == "valid" else "risky"
    message = (
        "Phone number format is valid and normalized."
        if status == "valid"
        else "Phone number is possible, but needs manual review."
    )
    return {
        "status": status,
        "confidence": int(candidate.get("confidence") or 0),
        "message": message,
        "raw": raw,
        "normalized": candidate.get("display", ""),
        "e164": candidate.get("e164", ""),
        "phone_type": candidate.get("phone_type", ""),
        "validation_status": validation_status,
    }
