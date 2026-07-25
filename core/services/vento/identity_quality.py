from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Tuple
from urllib.parse import urlparse

from core.services.vento.discovery import extract_social_handles, normalize_social_handle


_EMAIL_RE = re.compile(r"^[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Z0-9.-]+\.[A-Z]{2,}$", re.I)
_MASKED_EMAIL_RE = re.compile(r"(?:\*{2,}|x{2,}|\.\.{2,}).*@", re.I)
_INSTAGRAM_RESERVED = {
    "about", "accounts", "developer", "direct", "directory", "explore", "legal", "p", "press",
    "privacy", "reel", "reels", "stories", "tags", "terms", "tv", "web",
}
_TIKTOK_RESERVED = {
    "about", "business", "discover", "explore", "legal", "live", "music", "search", "tag", "video",
}
_DIRECTORY_DOMAINS = {
    "creators.feedspot.com", "feedspot.com", "influencers.club", "modash.io", "favikon.com",
    "socialveins.com", "starngage.com", "hypeauditor.com",
}
_CORPORATE_HANDLES = {"roverdotcom", "rover", "wag", "petsmart", "petco", "amazon"}
_TITLE_PREFIXES = (
    "how to ", "best ", "top ", "keeping ", "becoming ", "understanding ", "transform ", "create ",
    "my journey", "life as ", "testimonial ", "official guide", "email template", "10 essential",
)
_TITLE_WORDS = {
    "journey", "review", "reviews", "story", "experience", "items", "accessories", "adventures",
    "template", "guide", "tips", "ideas", "calls", "cleanup", "upgrade", "need",
}
_STORE_MARKERS = {
    "shop", "store", "worldwide shipping", "order now", "buy now", "catalog", "wholesale", "retail",
    "official store", "product page",
}
_MEDIA_MARKERS = {"news", "newspaper", "publication", "correspondent", "times", "magazine", "media outlet"}
_AGENCY_MARKERS = {"agency", "social media management", "content studio", "marketing studio", "manager"}


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _host(url: str) -> str:
    return (urlparse(str(url or "")).hostname or "").lower().removeprefix("www.")


def _path_parts(url: str) -> list[str]:
    return [part for part in urlparse(str(url or "")).path.split("/") if part]


def _url_handle(url: str, platform: str) -> str:
    host = _host(url)
    parts = _path_parts(url)
    if platform == "instagram" and host.endswith("instagram.com") and parts:
        candidate = parts[0].lstrip("@").lower()
        if candidate not in _INSTAGRAM_RESERVED:
            return normalize_social_handle(candidate, "instagram")
    if platform == "tiktok" and host.endswith("tiktok.com") and parts:
        candidate = next((part[1:] for part in parts if part.startswith("@") and len(part) > 1), "")
        if candidate and candidate.lower() not in _TIKTOK_RESERVED:
            return normalize_social_handle(candidate, "tiktok")
    return ""


def _canonical_url(handle: str, platform: str) -> str:
    normalized = normalize_social_handle(handle, platform)
    if not normalized:
        return ""
    if platform == "instagram":
        return f"https://www.instagram.com/{normalized}/"
    if platform == "tiktok":
        return f"https://www.tiktok.com/@{normalized}"
    return ""


def social_url_scope(url: str) -> str:
    host = _host(url)
    parts = _path_parts(url)
    if host.endswith("instagram.com"):
        if parts and parts[0].lower() in {"p", "reel", "reels", "tv", "stories"}:
            return "post"
        if len(parts) == 1 and _url_handle(url, "instagram"):
            return "profile"
    if host.endswith("tiktok.com"):
        if "video" in [part.lower() for part in parts]:
            return "post"
        if len(parts) == 1 and _url_handle(url, "tiktok"):
            return "profile"
    return "other"


def looks_like_content_title(value: str) -> bool:
    text = _clean(value)
    lower = text.lower()
    words = re.findall(r"[a-z0-9]+", lower)
    if not text:
        return True
    if any(lower.startswith(prefix) for prefix in _TITLE_PREFIXES):
        return True
    if len(words) >= 7:
        return True
    if len(words) >= 4 and any(word in _TITLE_WORDS for word in words):
        return True
    if any(marker in lower for marker in (" | tiktok", " - tiktok", "instagram photos and videos", "tiktok influencers")):
        return True
    return False


def _name_from_handle(handle: str) -> str:
    value = normalize_social_handle(handle, "instagram") or normalize_social_handle(handle, "tiktok")
    return re.sub(r"[._-]+", " ", value).strip().title().replace(" ", "") if value else ""


def _valid_email(value: Any) -> str:
    email = _clean(value).lower()
    if not email or _MASKED_EMAIL_RE.search(email) or not _EMAIL_RE.fullmatch(email):
        return ""
    return email


def classify_creator_entity(lead: Dict[str, Any]) -> Tuple[str, bool, list[str]]:
    name = _clean(lead.get("creator_name"))
    handle = _clean(lead.get("instagram_handle") or lead.get("tiktok_handle")).lstrip("@").lower()
    source_url = _clean(lead.get("discovery_source_url") or lead.get("source_url"))
    host = _host(source_url)
    text = " ".join(
        _clean(lead.get(field))
        for field in ("creator_name", "raw_title", "raw_content", "bio_text", "website")
    ).lower()

    reasons: list[str] = []
    if host in _DIRECTORY_DOMAINS or any(marker in text for marker in ("top influencers", "influencers in 202", "influencer list", "directory of")):
        return "directory_or_listicle", False, ["directory_or_listicle"]
    if handle in _CORPORATE_HANDLES or any(marker in text for marker in ("official rover", "pet care company", "official corporate")):
        return "corporate_brand", False, ["corporate_brand"]
    if any(marker in text for marker in _MEDIA_MARKERS) and ("creator" not in text or "correspondent" in text):
        return "media_or_publisher", False, ["media_or_publisher"]
    if handle.endswith((".ae", "_shop", ".shop")) or any(marker in text for marker in _STORE_MARKERS):
        return "store_or_shop", False, ["store_or_shop"]
    if any(marker in text for marker in _AGENCY_MARKERS):
        return "agency_or_manager", False, ["agency_or_manager"]
    if any(marker in text for marker in ("ugc creator", "content creator", "brand content")):
        return "ugc_creator", True, ["individual_ugc_creator"]
    if any(marker in text for marker in ("pet creator", "pet influencer", "dog mom", "dog dad", "dog parent", "cat mom")):
        return "pet_creator", True, ["individual_pet_creator"]
    if any(marker in text for marker in ("product reviewer", "car reviewer", "car gadget", "product review")):
        return "review_creator", True, ["individual_review_creator"]
    if lead.get("instagram_handle") or lead.get("tiktok_handle"):
        reasons.append("social_identity_present")
        return "individual_creator", True, reasons
    return "unknown_entity", False, ["creator_identity_not_established"]


def canonicalize_creator_identity(raw: Dict[str, Any]) -> Dict[str, Any]:
    lead = dict(raw)
    discovery_url = _clean(lead.get("discovery_source_url") or lead.get("source_url"))
    combined = " ".join(
        _clean(lead.get(field)) for field in ("raw_title", "raw_content", "bio_text")
    )
    handles = extract_social_handles(combined, discovery_url, _clean(lead.get("instagram_url")), _clean(lead.get("tiktok_url")))

    instagram = (
        normalize_social_handle(_clean(lead.get("instagram_handle")), "instagram")
        or _url_handle(_clean(lead.get("instagram_url")), "instagram")
        or _url_handle(discovery_url, "instagram")
        or next(iter(handles.get("instagram") or []), "")
    )
    tiktok = (
        normalize_social_handle(_clean(lead.get("tiktok_handle")), "tiktok")
        or _url_handle(_clean(lead.get("tiktok_url")), "tiktok")
        or _url_handle(discovery_url, "tiktok")
        or next(iter(handles.get("tiktok") or []), "")
    )

    instagram = normalize_social_handle(instagram, "instagram")
    tiktok = normalize_social_handle(tiktok, "tiktok")
    lead["instagram_handle"] = f"@{instagram}" if instagram else ""
    lead["instagram_url"] = _canonical_url(instagram, "instagram")
    lead["tiktok_handle"] = f"@{tiktok}" if tiktok else ""
    lead["tiktok_url"] = _canonical_url(tiktok, "tiktok")
    lead["discovery_source_url"] = discovery_url
    lead["email"] = _valid_email(lead.get("email"))

    name = _clean(lead.get("creator_name"))
    if looks_like_content_title(name):
        name = ""
    if not name:
        name = _name_from_handle(instagram or tiktok)
    lead["creator_name"] = name

    source_scope = social_url_scope(discovery_url)
    direct_social = bool(_url_handle(discovery_url, "instagram") or _url_handle(discovery_url, "tiktok"))
    has_social = bool(instagram or tiktok)
    lead["identity_source_scope"] = source_scope
    lead["identity_confidence"] = "high" if direct_social and has_social else "medium" if has_social else "low"
    if str(lead.get("source") or "").lower() == "tavily_fallback" and has_social:
        lead["profile_search_indexed"] = True

    creator_type, individual, reasons = classify_creator_entity(lead)
    lead["creator_type"] = creator_type
    lead["individual_creator"] = individual
    lead["identity_reasons"] = reasons
    lead["canonical_social_present"] = has_social
    return lead


def canonicalize_creator_identities(leads: Iterable[Dict[str, Any]]) -> list[Dict[str, Any]]:
    return [canonicalize_creator_identity(lead) for lead in leads if isinstance(lead, dict)]
