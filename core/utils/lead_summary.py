from __future__ import annotations

import re
from typing import Any, Dict
from urllib.parse import urlparse


EMPTY_VALUES = {"", "-", "--", "\u2014", "n/a", "na", "none", "null", "not found"}


def _clean(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return "" if text.lower() in EMPTY_VALUES else text


def _shorten(value: str, limit: int = 220) -> str:
    text = _clean(value)
    if len(text) <= limit:
        return text
    return text[: limit - 1].rsplit(" ", 1)[0].rstrip(".,;:") + "."


def _clean_specialty(value: Any) -> str:
    text = _clean(value).replace("_", " ").replace("/", " / ")
    text = re.sub(r"\s+", " ", text).strip(" -")
    if not text:
        return "B2B services"
    return text.lower() if text.isupper() else text


def _lead_location(lead: Dict[str, Any]) -> str:
    explicit = _clean(lead.get("location"))
    if explicit:
        return explicit
    parts = [_clean(lead.get("city")), _clean(lead.get("state"))]
    return ", ".join(part for part in parts if part)


def _has_usable(value: Any) -> bool:
    return bool(_clean(value))


def _website_name(url: str) -> str:
    host = (urlparse(url or "").hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def build_plain_lead_summary(lead: Dict[str, Any]) -> str:
    """Create a short non-technical description for Excel/UI review."""
    existing = _clean(
        lead.get("lead_summary")
        or lead.get("business_summary")
        or lead.get("plain_summary")
        or lead.get("what_they_do")
    )
    if existing:
        return _shorten(existing, 260)

    company = _clean(lead.get("company_name")) or "This company"
    specialty = _clean_specialty(lead.get("specialty") or lead.get("industry"))
    location = _lead_location(lead)
    website = _clean(lead.get("company_website") or lead.get("website"))
    value_prop = _clean(lead.get("value_proposition") or lead.get("notes"))

    intro = f"{company} appears to provide {specialty}"
    if location:
        intro += f" in/around {location}"
    intro += "."

    detail = ""
    if value_prop and company.lower() not in value_prop.lower():
        detail = _shorten(value_prop, 120)
    elif value_prop and not value_prop.lower().startswith(f"{company.lower()} is a verified"):
        detail = _shorten(value_prop, 120)

    contact_signals = []
    if _has_usable(lead.get("contact_email") or lead.get("email")):
        contact_signals.append("email")
    if _has_usable(lead.get("contact_phone") or lead.get("phone")):
        contact_signals.append("phone")
    if _has_usable(lead.get("linkedin_url") or lead.get("company_linkedin_url")):
        contact_signals.append("LinkedIn")
    if website:
        contact_signals.append("website")

    signal_text = ""
    if contact_signals:
        signal_text = f" Contact path found via {', '.join(dict.fromkeys(contact_signals))}."
    elif website:
        signal_text = f" Website found: {_website_name(website)}."

    if detail:
        if detail[-1:] not in {".", "!", "?"}:
            detail += "."
        return _shorten(f"{intro} {detail}{signal_text}", 280)
    return _shorten(f"{intro} Useful for outreach because it matches the selected lead category.{signal_text}", 280)
