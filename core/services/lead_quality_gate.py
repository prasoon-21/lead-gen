from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Tuple
from urllib.parse import urlparse


BLACKLISTED_DOMAIN_KEYWORDS = {
    "yelp",
    "yellowpages",
    "clutch",
    "upwork",
    "fiverr",
    "indeed",
    "glassdoor",
    "monster",
    "ziprecruiter",
    "careerbuilder",
    "linkedin.com/jobs",
    "facebook",
    "instagram",
    "youtube",
    "tiktok",
    "mapquest",
    "bbb",
    "chamberofcommerce",
    "thumbtack",
    "angi",
    "craigslist",
    "directory",
    "marketplace",
}

BLACKLISTED_URL_KEYWORDS = {
    "directory",
    "marketplace",
}

NOISE_PATH_KEYWORDS = {
    "/blog",
    "/blogs",
    "/news",
    "/article",
    "/articles",
    "/jobs",
    "/careers",
    "/privacy",
    "/terms",
    "/category",
    "/tag/",
    ".pdf",
}

GENERIC_EMAIL_PREFIXES = {
    "info",
    "sales",
    "support",
    "admin",
    "contact",
    "hello",
    "office",
    "team",
    "help",
    "service",
    "enquiries",
    "inquiries",
}

FREE_EMAIL_DOMAINS = {
    "gmail.com",
    "yahoo.com",
    "hotmail.com",
    "outlook.com",
    "aol.com",
    "icloud.com",
    "proton.me",
    "protonmail.com",
}

EXECUTIVE_TITLE_PATTERNS = (
    (re.compile(r"\b(co[-\s]?founder|founder|owner|chief executive officer|ceo|principal)\b", re.I), 25),
    (re.compile(r"\b(coo|chief operating officer|operations lead|managing director|general manager)\b", re.I), 15),
    (re.compile(r"\b(growth|sales)\s+(director|lead|head|vp|vice president)\b", re.I), 15),
    (re.compile(r"\b(vp|vice president)\s+(sales|growth|business development)\b", re.I), 15),
)

BANNED_TITLE_PATTERN = re.compile(
    r"\b(hr|human resources|recruiter|talent|intern|assistant|admin|administrator|"
    r"support|customer support|legal counsel|attorney|paralegal|bookkeeper)\b",
    re.I,
)

B2B_SERVICE_MARKERS = {
    "agency",
    "consulting",
    "consultant",
    "services",
    "solutions",
    "automation",
    "ai",
    "artificial intelligence",
    "software",
    "technology",
    "it services",
    "managed services",
    "development",
    "integration",
    "upfitter",
    "upfitting",
    "van conversion",
    "commercial vehicle",
    "fleet",
    "truck body",
}

CONSUMER_RETAIL_MARKERS = {
    "add to cart",
    "shopping cart",
    "checkout",
    "shop now",
    "buy now",
    "consumer products",
    "retail store",
    "online store",
}


def apply_quality_gate(
    leads: Iterable[Dict[str, Any]],
    *,
    drop_rejected: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Normalize, score, reject, and domain-dedupe raw lead rows.

    The returned records keep the old flat lead fields for the UI while adding
    the structured Apollo/Hunter-like fields requested by the pipeline.
    """
    merged_by_domain: Dict[str, Dict[str, Any]] = {}
    stats = {
        "input_count": 0,
        "approved_count": 0,
        "needs_review_count": 0,
        "rejected_count": 0,
        "merged_duplicate_domains": 0,
        "rejection_reasons": {},
    }

    for lead in leads or []:
        if not isinstance(lead, dict):
            continue
        stats["input_count"] += 1
        normalized = normalize_and_score_lead(lead)
        disposition = normalized.get("pipeline_disposition")
        if disposition == "Rejected":
            stats["rejected_count"] += 1
            reason = normalized.get("rejection_reason_code") or "ERR_LOW_SCORE"
            stats["rejection_reasons"][reason] = stats["rejection_reasons"].get(reason, 0) + 1
            if drop_rejected:
                continue

        domain = str(normalized.get("target_domain") or "").strip()
        if not domain:
            reason = "ERR_MISSING_DOMAIN"
            normalized["pipeline_disposition"] = "Rejected"
            normalized["rejection_reason_code"] = reason
            stats["rejected_count"] += 1
            stats["rejection_reasons"][reason] = stats["rejection_reasons"].get(reason, 0) + 1
            if drop_rejected:
                continue

        existing = merged_by_domain.get(domain)
        if existing:
            stats["merged_duplicate_domains"] += 1
            merged_by_domain[domain] = merge_company_records(existing, normalized)
        else:
            merged_by_domain[domain] = normalized

    final = list(merged_by_domain.values())
    for item in final:
        if item.get("pipeline_disposition") == "Approved":
            stats["approved_count"] += 1
        elif item.get("pipeline_disposition") == "Needs_Review":
            stats["needs_review_count"] += 1

    final.sort(
        key=lambda item: (
            int(item.get("quality_score") or 0),
            1 if item.get("pipeline_disposition") == "Approved" else 0,
            1 if item.get("contacts") else 0,
        ),
        reverse=True,
    )
    return final, stats


def normalize_and_score_lead(raw_lead: Dict[str, Any]) -> Dict[str, Any]:
    lead = deepcopy(raw_lead)
    website = _first_non_empty(
        lead.get("website_url"),
        lead.get("company_website"),
        lead.get("website"),
        lead.get("contact_page"),
    )
    domain = _domain_from_url(website) or _domain_from_email(_first_non_empty(lead.get("contact_email"), lead.get("email")))
    company_name = _sanitize_company_name(
        _first_non_empty(lead.get("company_name"), lead.get("company"), lead.get("name")) or _company_name_from_domain(domain)
    )

    rejection_reason = _immediate_rejection_reason(lead=lead, website=website, domain=domain)
    contacts, alternative_contacts = _build_contacts(lead, domain=domain)
    score, scoring_notes = _score_record(
        lead=lead,
        website=website,
        domain=domain,
        contacts=contacts,
        alternative_contacts=alternative_contacts,
    )
    disposition, threshold_reason = _disposition(score=score, rejection_reason=rejection_reason, contacts=contacts, alternative_contacts=alternative_contacts, website=website)
    if threshold_reason:
        rejection_reason = threshold_reason

    lead.update(
        {
            "company_name": company_name,
            "target_domain": domain,
            "website_url": website,
            "company_website": website,
            "quality_score": score,
            "pipeline_disposition": disposition,
            "rejection_reason_code": rejection_reason or "NULL",
            "contacts": contacts,
            "alternative_contacts": alternative_contacts,
            "quality_notes": _merge_unique(lead.get("quality_notes"), scoring_notes),
            "usable": disposition in {"Approved", "Needs_Review"},
            "rejected": disposition == "Rejected",
        }
    )

    # Backward-compatible flat fields for existing UI/export code.
    primary = contacts[0] if contacts else {}
    if primary:
        lead["contact_person_name"] = _join_name(primary.get("first_name"), primary.get("last_name")) or lead.get("contact_person_name") or ""
        lead["founder_name"] = lead.get("founder_name") or lead["contact_person_name"]
        lead["contact_person_title"] = primary.get("executive_title") or lead.get("contact_person_title") or ""
        lead["contact_email"] = primary.get("validated_email") or lead.get("contact_email") or ""
        lead["contact_phone"] = primary.get("direct_phone") or lead.get("contact_phone") or ""
        lead["linkedin_url"] = primary.get("personal_linkedin") or lead.get("linkedin_url") or ""
    elif _is_generic_email(_first_non_empty(lead.get("contact_email"), lead.get("email"))):
        lead["contact_email"] = ""

    lead["confidence"] = "high" if disposition == "Approved" else "medium" if disposition == "Needs_Review" else "low"
    lead["quality_status"] = disposition
    lead["acceptance_reason"] = _explain_disposition(lead)
    lead["confidence_explanation"] = lead["acceptance_reason"]
    return lead


def merge_company_records(existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(existing)
    merged["quality_score"] = max(int(existing.get("quality_score") or 0), int(incoming.get("quality_score") or 0))
    merged["pipeline_disposition"] = _best_disposition(existing.get("pipeline_disposition"), incoming.get("pipeline_disposition"))
    if merged["pipeline_disposition"] != "Rejected":
        merged["rejection_reason_code"] = "NULL"

    merged["contacts"] = _merge_contacts(existing.get("contacts") or [], incoming.get("contacts") or [])
    merged["alternative_contacts"] = _merge_alternative_contacts(
        existing.get("alternative_contacts") or {},
        incoming.get("alternative_contacts") or {},
    )
    for field in ("company_name", "website_url", "company_website", "location", "lead_summary", "value_proposition"):
        if not _clean(merged.get(field)) and _clean(incoming.get(field)):
            merged[field] = incoming[field]

    primary = merged["contacts"][0] if merged["contacts"] else {}
    if primary:
        merged["contact_person_name"] = _join_name(primary.get("first_name"), primary.get("last_name"))
        merged["contact_person_title"] = primary.get("executive_title") or ""
        merged["contact_email"] = primary.get("validated_email") or ""
        merged["contact_phone"] = primary.get("direct_phone") or ""
        merged["linkedin_url"] = primary.get("personal_linkedin") or ""
    return merged


def _build_contacts(lead: Dict[str, Any], *, domain: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    raw_email = _first_non_empty(lead.get("validated_email"), lead.get("contact_email"), lead.get("email"))
    emails = _merge_unique([raw_email], lead.get("official_site_emails"), lead.get("emails"))
    generic_emails = [email for email in emails if _is_generic_email(email)]
    direct_emails = [email for email in emails if email and not _is_generic_email(email)]

    person_name = _first_non_empty(lead.get("contact_person_name"), lead.get("founder_name"), lead.get("person_name"))
    title = _first_non_empty(lead.get("contact_person_title"), lead.get("executive_title"), lead.get("designation"), lead.get("role"))
    linkedin = _first_non_empty(lead.get("personal_linkedin"), lead.get("linkedin_url"), lead.get("person_linkedin_url"))
    phone = _normalize_phone(_first_non_empty(lead.get("direct_phone"), lead.get("contact_phone"), lead.get("phone")))

    company_linkedin = _first_non_empty(lead.get("company_linkedin"), lead.get("company_linkedin_url"))
    alternative_contacts = {
        "generic_emails": sorted(set(generic_emails)),
        "company_linkedin": company_linkedin or None,
    }

    contacts: List[Dict[str, Any]] = []
    if person_name or title or linkedin or direct_emails or phone:
        if _is_banned_title(title):
            return [], alternative_contacts
        title_points = _persona_points(title)
        # A direct email without a person is still a usable company contact, but
        # it must not be falsely attached to a named executive.
        if person_name or title_points > 0 or linkedin:
            first_name, last_name = _split_name(person_name)
            contacts.append(
                {
                    "first_name": first_name,
                    "last_name": last_name,
                    "executive_title": title or None,
                    "validated_email": direct_emails[0] if direct_emails else None,
                    "email_type": "direct" if direct_emails else None,
                    "direct_phone": phone or None,
                    "personal_linkedin": linkedin if _looks_like_person_linkedin(linkedin) else None,
                }
            )
        elif direct_emails or phone:
            contacts.append(
                {
                    "first_name": None,
                    "last_name": None,
                    "executive_title": None,
                    "validated_email": direct_emails[0] if direct_emails else None,
                    "email_type": "direct" if direct_emails else None,
                    "direct_phone": phone or None,
                    "personal_linkedin": None,
                }
            )

    if not contacts and phone:
        contacts.append(
            {
                "first_name": None,
                "last_name": None,
                "executive_title": None,
                "validated_email": None,
                "email_type": None,
                "direct_phone": phone,
                "personal_linkedin": None,
            }
        )

    return contacts, alternative_contacts


def _score_record(
    *,
    lead: Dict[str, Any],
    website: str,
    domain: str,
    contacts: List[Dict[str, Any]],
    alternative_contacts: Dict[str, Any],
) -> Tuple[int, List[str]]:
    score = 0
    notes: List[str] = []

    if website and domain:
        score += 20
        notes.append("official_reachable_website")

    company_text = " ".join(
        str(lead.get(key) or "").lower()
        for key in ("industry", "lead_summary", "value_proposition", "notes")
    )
    if any(_contains_marker(company_text, marker) for marker in B2B_SERVICE_MARKERS):
        score += 10
        notes.append("b2b_service_fit")

    primary = contacts[0] if contacts else {}
    email = _clean(primary.get("validated_email"))
    phone = _clean(primary.get("direct_phone"))
    linkedin = _clean(primary.get("personal_linkedin"))
    title = _clean(primary.get("executive_title"))

    if email:
        score += 40
        notes.append("direct_b2b_email")
    elif phone:
        score += 30
        notes.append("validated_phone")
    elif alternative_contacts.get("generic_emails"):
        score += 25
        notes.append("generic_corporate_email")

    persona_points = _persona_points(title)
    score += persona_points
    if persona_points == 25:
        notes.append("founder_or_c_level_persona")
    elif persona_points == 15:
        notes.append("operations_or_growth_persona")

    if linkedin:
        score += 15
        notes.append("personal_linkedin")
    elif alternative_contacts.get("company_linkedin"):
        score += 5
        notes.append("company_linkedin")

    if email and _domain_from_email(email) in FREE_EMAIL_DOMAINS and not _looks_like_solo_operator(lead):
        score -= 30
        notes.append("free_email_penalty")

    site_status = str(lead.get("site_status") or lead.get("verification_status") or "").lower()
    if any(marker in site_status for marker in ("dead", "broken", "bad gateway", "502", "parked")):
        score -= 50
        notes.append("dead_site_penalty")

    return max(0, min(score, 100)), notes


def _disposition(
    *,
    score: int,
    rejection_reason: str,
    contacts: List[Dict[str, Any]],
    alternative_contacts: Dict[str, Any],
    website: str,
) -> Tuple[str, str]:
    has_direct_channel = bool(
        contacts
        or alternative_contacts.get("generic_emails")
        or alternative_contacts.get("company_linkedin")
    )
    if rejection_reason:
        return "Rejected", rejection_reason
    if not website or not has_direct_channel:
        return "Rejected", "ERR_NO_DIRECT_CONTACT"
    if score >= 75:
        return "Approved", ""
    if score >= 50:
        return "Needs_Review", ""
    return "Rejected", "ERR_LOW_SCORE"


def _immediate_rejection_reason(*, lead: Dict[str, Any], website: str, domain: str) -> str:
    url_haystack = " ".join(
        str(value or "").lower()
        for value in (
            website,
            domain,
            lead.get("contact_page"),
        )
    )
    content_haystack = " ".join(
        str(value or "").lower()
        for value in (
            website,
            domain,
            lead.get("contact_page"),
            lead.get("company_name"),
            lead.get("lead_summary"),
            lead.get("value_proposition"),
        )
    )
    path = (urlparse(website or "").path or "").lower()
    if any(keyword in url_haystack for keyword in BLACKLISTED_URL_KEYWORDS):
        return "ERR_BANNED_INDUSTRY"
    if any(keyword in content_haystack for keyword in BLACKLISTED_DOMAIN_KEYWORDS - BLACKLISTED_URL_KEYWORDS):
        return "ERR_BANNED_INDUSTRY"
    if any(keyword in path for keyword in NOISE_PATH_KEYWORDS):
        return "ERR_BANNED_INDUSTRY"
    if any(marker in content_haystack for marker in CONSUMER_RETAIL_MARKERS) and not any(
        _contains_marker(content_haystack, marker) for marker in B2B_SERVICE_MARKERS
    ):
        return "ERR_BANNED_INDUSTRY"
    return ""


def _contains_marker(text: str, marker: str) -> bool:
    cleaned = str(marker or "").lower().strip()
    if not cleaned:
        return False
    if len(cleaned) <= 3 and cleaned.isalnum():
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(cleaned)}(?![a-z0-9])", text or ""))
    return cleaned in (text or "")


def _merge_contacts(existing: List[Dict[str, Any]], incoming: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    contacts: List[Dict[str, Any]] = []
    seen = set()
    for contact in list(existing or []) + list(incoming or []):
        if not isinstance(contact, dict):
            continue
        key = (
            _clean(contact.get("validated_email")).lower()
            or _clean(contact.get("personal_linkedin")).lower()
            or "|".join(
                part
                for part in (
                    _clean(contact.get("first_name")).lower(),
                    _clean(contact.get("last_name")).lower(),
                    _clean(contact.get("executive_title")).lower(),
                )
                if part
            )
        )
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        contacts.append(contact)
    return contacts


def _merge_alternative_contacts(existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    emails = sorted(set((existing.get("generic_emails") or []) + (incoming.get("generic_emails") or [])))
    return {
        "generic_emails": emails,
        "company_linkedin": existing.get("company_linkedin") or incoming.get("company_linkedin") or None,
    }


def _best_disposition(left: Any, right: Any) -> str:
    order = {"Rejected": 0, "Needs_Review": 1, "Approved": 2}
    return max((_clean(left) or "Rejected", _clean(right) or "Rejected"), key=lambda item: order.get(item, 0))


def _persona_points(title: str) -> int:
    if not title or _is_banned_title(title):
        return 0
    for pattern, points in EXECUTIVE_TITLE_PATTERNS:
        if pattern.search(title):
            return points
    return 0


def _is_banned_title(title: str) -> bool:
    return bool(title and BANNED_TITLE_PATTERN.search(title))


def _is_generic_email(email: str) -> bool:
    local = (email or "").split("@", 1)[0].lower().strip()
    return local in GENERIC_EMAIL_PREFIXES


def _looks_like_solo_operator(lead: Dict[str, Any]) -> bool:
    text = " ".join(
        str(lead.get(key) or "").lower()
        for key in ("company_size", "lead_summary", "value_proposition", "notes")
    )
    return any(marker in text for marker in ("solo", "independent", "owner-operated", "freelance", "one-person"))


def _looks_like_person_linkedin(url: str) -> bool:
    lowered = (url or "").lower()
    return "linkedin.com/in/" in lowered


def _normalize_phone(value: str) -> str:
    cleaned = _clean(value)
    if not cleaned:
        return ""
    digits = re.sub(r"\D", "", cleaned)
    if len(digits) == 11 and digits.startswith("1"):
        return f"+{digits}"
    if len(digits) == 10:
        return f"+1{digits}"
    if cleaned.startswith("+"):
        return cleaned
    return cleaned


def _split_name(name: str) -> Tuple[Any, Any]:
    cleaned = _clean(name)
    if not cleaned:
        return None, None
    parts = cleaned.split()
    if len(parts) == 1:
        return parts[0], None
    return parts[0], " ".join(parts[1:])


def _join_name(first: Any, last: Any) -> str:
    return " ".join(part for part in (_clean(first), _clean(last)) if part)


def _sanitize_company_name(value: str) -> str:
    cleaned = _clean(value)
    cleaned = re.sub(r"\b(llc|inc|ltd|limited|corp|corporation|pvt|private|plc)\.?\b", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,-")
    return cleaned


def _domain_from_url(url: str) -> str:
    host = (urlparse(url or "").hostname or "").lower().strip()
    return host[4:] if host.startswith("www.") else host


def _domain_from_email(email: str) -> str:
    parts = str(email or "").split("@", 1)
    return parts[1].lower().strip() if len(parts) == 2 else ""


def _company_name_from_domain(domain: str) -> str:
    label = (domain or "").split(".", 1)[0]
    return " ".join(part.capitalize() for part in re.split(r"[-_]+", label) if part)


def _clean(value: Any) -> str:
    cleaned = " ".join(str(value or "").split()).strip()
    return "" if cleaned.lower() in {"-", "—", "â€”", "n/a", "na", "none", "null"} else cleaned


def _first_non_empty(*values: Any) -> str:
    for value in values:
        cleaned = _clean(value)
        if cleaned:
            return cleaned
    return ""


def _merge_unique(*value_sets: Any) -> List[str]:
    values: List[str] = []
    for value_set in value_sets:
        if not value_set:
            continue
        if isinstance(value_set, str):
            iterable = [value_set]
        elif isinstance(value_set, dict):
            iterable = value_set.values()
        else:
            iterable = value_set
        for value in iterable:
            cleaned = _clean(value).lower() if "@" in _clean(value) else _clean(value)
            if cleaned and cleaned not in values:
                values.append(cleaned)
    return values


def _explain_disposition(lead: Dict[str, Any]) -> str:
    disposition = lead.get("pipeline_disposition")
    score = lead.get("quality_score")
    if disposition == "Rejected":
        return f"Rejected by strict quality gate: {lead.get('rejection_reason_code') or 'ERR_LOW_SCORE'}."
    company = lead.get("company_name") or "This company"
    return f"{company} marked {disposition} by strict quality gate with score {score}/100."
