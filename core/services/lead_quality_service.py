from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse


GENERIC_EMAIL_PREFIXES = {
    "info",
    "support",
    "contact",
    "hello",
    "sales",
    "admin",
    "office",
    "team",
    "help",
}


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _first_non_empty(*values: Any) -> str:
    for value in values:
        cleaned = _clean(value)
        if cleaned:
            return cleaned
    return ""


def _domain_from_url(url: str) -> str:
    return (urlparse(url or "").hostname or "").lower()


def _domain_from_email(email: str) -> str:
    parts = (email or "").split("@", 1)
    return parts[1].lower().strip() if len(parts) == 2 else ""


def _is_generic_email(email: str) -> bool:
    local = (email or "").split("@", 1)[0].lower().strip()
    return local in GENERIC_EMAIL_PREFIXES


def _field_source_map(lead: Dict[str, Any]) -> Dict[str, str]:
    existing = lead.get("field_sources")
    if isinstance(existing, dict):
        normalized = {}
        for key, value in existing.items():
            cleaned = _clean(value)
            if cleaned:
                normalized[str(key)] = cleaned
        if normalized:
            return normalized

    source = _clean(lead.get("source")) or "unknown"
    field_sources: Dict[str, str] = {}

    if _clean(lead.get("company_name")):
        field_sources["company_name"] = source
    if _clean(lead.get("company_website")):
        field_sources["company_website"] = "website" if "website" in source or source == "unknown" else source
    if _clean(lead.get("contact_person_name")) or _clean(lead.get("founder_name")):
        field_sources["contact_person_name"] = "linkedin" if _clean(lead.get("linkedin_url")) else source
    if _clean(lead.get("contact_person_title")):
        field_sources["contact_person_title"] = "linkedin" if _clean(lead.get("linkedin_url")) else source
    if _clean(lead.get("contact_email")):
        field_sources["contact_email"] = "linkedin" if source == "linkedin" else source
    if _clean(lead.get("contact_phone")):
        field_sources["contact_phone"] = source
    if _clean(lead.get("linkedin_url")):
        field_sources["linkedin_url"] = "linkedin"
    if _clean(lead.get("contact_page")):
        field_sources["contact_page"] = "website"
    if _clean(lead.get("location")):
        field_sources["location"] = source
    return field_sources


def score_lead(lead: Dict[str, Any], verification: Dict[str, Any] | None = None) -> Dict[str, Any]:
    company_name = _clean(lead.get("company_name"))
    website = _clean(lead.get("company_website"))
    person_name = _first_non_empty(lead.get("contact_person_name"), lead.get("founder_name"))
    person_title = _clean(lead.get("contact_person_title"))
    email = _clean(lead.get("contact_email"))
    phone = _clean(lead.get("contact_phone"))
    linkedin_url = _clean(lead.get("linkedin_url"))
    contact_page = _clean(lead.get("contact_page"))
    value_prop = _clean(lead.get("value_proposition"))
    location = _clean(lead.get("location"))

    score = 0
    notes: List[str] = []
    warnings: List[str] = []

    if company_name:
        score += 15
    else:
        warnings.append("missing_company_name")

    if website:
        score += 18
    else:
        warnings.append("missing_company_website")

    if person_name:
        score += 16
    else:
        warnings.append("missing_contact_person")

    if person_title:
        score += 8

    email_domain = _domain_from_email(email)
    website_domain = _domain_from_url(website)

    if email:
        score += 14
        if _is_generic_email(email):
            score -= 4
            notes.append("generic_email_fallback")
        if website_domain and email_domain and website_domain in email_domain:
            score += 4
            notes.append("email_matches_company_domain")
    else:
        warnings.append("missing_email")

    if phone:
        score += 8
    if linkedin_url:
        score += 8
    if contact_page:
        score += 6
    if value_prop:
        score += 5
    if location:
        score += 4

    if verification:
        status = _clean(verification.get("status")).lower()
        if status == "valid":
            score += 14
            notes.append("email_verified_valid")
        elif status == "risky":
            score += 4
            notes.append("email_verification_risky")
        elif status == "invalid":
            score -= 14
            warnings.append("email_invalid")

    fallback_paths = lead.get("fallback_contact_paths")
    if isinstance(fallback_paths, list) and fallback_paths:
        score += min(6, len(fallback_paths) * 2)

    usable_contact_paths = sum(bool(value) for value in (email, phone, linkedin_url, contact_page))
    if usable_contact_paths == 0:
        score -= 18
        warnings.append("no_contact_path")

    if not website and not linkedin_url and not contact_page:
        score -= 20
        warnings.append("weak_public_presence")

    if person_name and not email and not phone and not linkedin_url:
        score -= 6
        notes.append("person_without_contact_path")

    score = max(score, 0)

    if score >= 70:
        confidence = "high"
    elif score >= 42:
        confidence = "medium"
    else:
        confidence = "low"

    rejected = (
        not company_name
        or not website
        or usable_contact_paths == 0
        or ("email_invalid" in warnings and usable_contact_paths < 2)
    )
    usable = not rejected and score >= 30

    if rejected:
        quality_status = "Rejected"
    elif confidence == "high":
        quality_status = "Strong"
    elif confidence == "medium":
        quality_status = "Usable"
    else:
        quality_status = "Partial"

    return {
        "quality_score": score,
        "confidence": confidence,
        "quality_status": quality_status,
        "usable": usable,
        "rejected": rejected,
        "quality_notes": notes,
        "quality_warnings": warnings,
        "field_sources": _field_source_map(lead),
    }


def rank_and_enrich_leads(leads: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    enriched: List[Dict[str, Any]] = []
    stats = {
        "input_count": len(leads),
        "rejected_count": 0,
        "usable_count": 0,
        "partial_count": 0,
    }

    for lead in leads:
        item = deepcopy(lead)
        quality = score_lead(item)
        item.update(quality)
        if item["rejected"]:
            stats["rejected_count"] += 1
            continue
        if item["quality_status"] == "Partial":
            stats["partial_count"] += 1
        else:
            stats["usable_count"] += 1
        enriched.append(item)

    enriched.sort(
        key=lambda lead: (
            lead.get("quality_score", 0),
            1 if lead.get("confidence") == "high" else 0,
            1 if _clean(lead.get("contact_email")) else 0,
            1 if _clean(lead.get("contact_person_name") or lead.get("founder_name")) else 0,
        ),
        reverse=True,
    )
    return enriched, stats
