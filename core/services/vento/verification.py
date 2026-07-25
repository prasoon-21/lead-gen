from __future__ import annotations

import asyncio
import os
import re
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import httpx

from core.utils.email_verify import verify_email_legitimacy


PROFILE_STATUSES = {
    "verified_active",
    "verified_inactive",
    "not_found",
    "private",
    "rate_limited",
    "unknown",
}
EMAIL_STATUSES = {"valid", "risky", "invalid", "unknown", "not_available"}
LOCATION_STATUSES = {"confirmed", "likely", "uncertain", "conflicting", "outside_target", "unknown"}
FOLLOWER_STATUSES = {
    "verified", "provider_verified", "provider_reported", "search_observed", "imported_unverified", "estimated", "unknown",
}

_EMAIL_RE = re.compile(r"^[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?(?:\.[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?)+$", re.I)
_US_STATE_CODES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA",
    "colorado": "CO", "connecticut": "CT", "delaware": "DE", "district of columbia": "DC",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID", "illinois": "IL",
    "indiana": "IN", "iowa": "IA", "kansas": "KS", "kentucky": "KY", "louisiana": "LA",
    "maine": "ME", "maryland": "MD", "massachusetts": "MA", "michigan": "MI", "minnesota": "MN",
    "mississippi": "MS", "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR",
    "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC", "south dakota": "SD",
    "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT", "virginia": "VA",
    "washington": "WA", "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
}
_US_CODES = set(_US_STATE_CODES.values())
_PNW_CITIES = {
    "seattle": ("Seattle", "WA"), "bellevue": ("Bellevue", "WA"), "tacoma": ("Tacoma", "WA"),
    "redmond": ("Redmond", "WA"), "kirkland": ("Kirkland", "WA"), "everett": ("Everett", "WA"),
    "portland": ("Portland", "OR"), "beaverton": ("Beaverton", "OR"), "hillsboro": ("Hillsboro", "OR"),
    "gresham": ("Gresham", "OR"), "lake oswego": ("Lake Oswego", "OR"), "eugene": ("Eugene", "OR"),
    "bend": ("Bend", "OR"),
}
_US_MAJOR_CITIES = {
    **_PNW_CITIES,
    "los angeles": ("Los Angeles", "CA"), "san diego": ("San Diego", "CA"),
    "san francisco": ("San Francisco", "CA"), "sacramento": ("Sacramento", "CA"),
    "san jose": ("San Jose", "CA"), "long beach": ("Long Beach", "CA"),
    "new york": ("New York", "NY"), "brooklyn": ("Brooklyn", "NY"),
    "miami": ("Miami", "FL"), "orlando": ("Orlando", "FL"), "tampa": ("Tampa", "FL"),
    "austin": ("Austin", "TX"), "dallas": ("Dallas", "TX"), "houston": ("Houston", "TX"),
    "san antonio": ("San Antonio", "TX"), "fort worth": ("Fort Worth", "TX"),
    "chicago": ("Chicago", "IL"), "denver": ("Denver", "CO"), "phoenix": ("Phoenix", "AZ"),
    "las vegas": ("Las Vegas", "NV"), "boston": ("Boston", "MA"), "atlanta": ("Atlanta", "GA"),
    "nashville": ("Nashville", "TN"), "charlotte": ("Charlotte", "NC"),
    "philadelphia": ("Philadelphia", "PA"), "minneapolis": ("Minneapolis", "MN"),
    "salt lake city": ("Salt Lake City", "UT"), "boise": ("Boise", "ID"),
}
_CANADA_MARKERS = {"canada", "canadian", "ontario", "british columbia", "alberta", "quebec", "vancouver", "toronto"}
_NON_US_LOCATIONS = {
    "london": ("London", "", "UK"), "united kingdom": ("", "", "UK"), "england": ("", "", "UK"),
    "mumbai": ("Mumbai", "", "India"), "delhi": ("Delhi", "", "India"), "bangalore": ("Bangalore", "", "India"),
    "india": ("", "", "India"), "dubai": ("Dubai", "", "UAE"), "united arab emirates": ("", "", "UAE"),
    "australia": ("", "", "Australia"), "sydney": ("Sydney", "", "Australia"),
    "melbourne": ("Melbourne", "", "Australia"), "singapore": ("Singapore", "", "Singapore"),
    "paris": ("Paris", "", "France"), "france": ("", "", "France"), "germany": ("", "", "Germany"),
}
_NICHE_TAGS = {
    "Pet Creator": ("pet creator", "pet influencer", "pet lifestyle", "pets"),
    "Dog Parent": ("dog mom", "dog dad", "dog parent", "dogs", "puppy"),
    "Dog Trainer": ("dog trainer", "canine trainer", "dog training"),
    "Pet Safety": ("pet safety", "animal safety", "dog safety"),
    "UGC Creator": ("ugc", "user generated content", "content creator"),
    "Product Reviewer": ("product review", "reviewer", "product testing"),
    "Car Gadgets": ("car gadget", "car accessories", "car essentials", "automotive"),
    "Amazon Finds": ("amazon finds", "amazon must haves"),
    "Outdoor Lifestyle": ("outdoor", "adventure", "hiking", "camping"),
    "Road Trip / Camping": ("road trip", "roadtrip", "camping", "van life"),
    "PNW Lifestyle": ("pnw", "pacific northwest", "seattle lifestyle", "portland lifestyle"),
    "Local Lifestyle": ("local lifestyle", "lifestyle vlogger", "daily life"),
}
_BRAND_SAFETY_CRITICAL = (
    "animal abuse",
    "hate speech",
    "violent extremism",
    "illegal animal fighting",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_follower_count(value: Any) -> Optional[int]:
    """Parse common follower formats without turning 12.5K into 12."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return max(0, int(value))
    text = str(value).strip().lower().replace(",", "").replace("followers", "").strip()
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*([kmb]?)", text)
    if not match:
        return None
    number = float(match.group(1))
    multiplier = {"": 1, "k": 1_000, "m": 1_000_000, "b": 1_000_000_000}[match.group(2)]
    return max(0, int(number * multiplier))


def follower_tier(count: Optional[int]) -> str:
    if count is None:
        return "unknown"
    if count < 5_000:
        return "below_nano"
    if count < 20_000:
        return "nano"
    if count < 100_000:
        return "mid"
    if count <= 500_000:
        return "upper_mid"
    return "above_target"


def _extract_location(text: str) -> Tuple[str, str, str, str]:
    original = str(text or "")
    lower = original.lower()
    for marker, (city, state) in _US_MAJOR_CITIES.items():
        if re.search(rf"\b{re.escape(marker)}\b", lower):
            return city, state, "US", "PNW" if state in {"WA", "OR"} else "US"
    if re.search(r"\b(?:socal|southern california)\b", lower):
        return "", "CA", "US", "US"
    if re.search(r"\b(?:norcal|northern california)\b", lower):
        return "", "CA", "US", "US"
    if re.search(r"\b(?:pnw|pacific northwest)\b", lower):
        return "", "", "US", "PNW"
    for state_name, code in _US_STATE_CODES.items():
        if re.search(rf"\b{re.escape(state_name)}\b", lower):
            return "", code, "US", "PNW" if code in {"WA", "OR"} else "US"
    # Two-letter state codes are accepted only in explicit location context and
    # remain case-sensitive. This prevents IN from "income", ME from "me", and
    # ID from "animal ID" from becoming creator locations.
    code_match = re.search(
        r"(?:\b(?:based|located|living|creator|from)\s+in\s+|[,|]\s*)([A-Z]{2})(?=\s*(?:[,|]|\b|$))",
        original,
    )
    if code_match and code_match.group(1) in _US_CODES:
        code = code_match.group(1).upper()
        return "", code, "US", "PNW" if code in {"WA", "OR"} else "US"
    if any(marker in lower for marker in _CANADA_MARKERS):
        return "", "", "Canada", "Canada"
    for marker, (city, state, country) in _NON_US_LOCATIONS.items():
        if re.search(rf"\b{re.escape(marker)}\b", lower):
            return city, state, country, country
    if re.search(r"\b(united states|u\.s\.|usa)\b", lower):
        return "", "", "US", "US"
    return "", "", "", ""


def verify_creator_location(lead: Dict[str, Any], *, target: str = "", strict_us_only: bool = True) -> Dict[str, Any]:
    explicit = str(lead.get("location") or "").strip()
    raw_content = " ".join(
        str(lead.get(field) or "") for field in ("raw_content", "raw_title", "bio_text", "location_evidence")
    ).strip()
    explicit_city, explicit_state, explicit_country, explicit_region = _extract_location(explicit)
    inferred_city, inferred_state, inferred_country, inferred_region = _extract_location(raw_content)

    source = str(lead.get("source") or "").lower()
    if explicit and (explicit_city or explicit_state or explicit_country):
        confidence = "confirmed" if source in {"bulk_ingest", "social_data_api", "apify", "rapidapi"} else "likely"
        city, state, country, region = explicit_city, explicit_state, explicit_country, explicit_region
        evidence = explicit
    elif inferred_city or inferred_state or inferred_country:
        city, state, country, region = inferred_city, inferred_state, inferred_country, inferred_region
        confidence = "likely" if city and country else "uncertain"
        evidence = raw_content[:500]
    else:
        city = state = country = region = evidence = ""
        confidence = "unknown"

    if (
        explicit_country
        and inferred_country
        and (explicit_country != inferred_country or (explicit_state and inferred_state and explicit_state != inferred_state))
    ):
        confidence = "conflicting"
        evidence = " | ".join(value for value in (explicit, raw_content[:500]) if value)
    if strict_us_only and country and country != "US":
        confidence = "outside_target"

    normalized = ", ".join(value for value in (city, state, country) if value)
    return {
        "location": normalized or explicit,
        "location_city": city,
        "location_state": state,
        "location_country": country,
        "location_region": region,
        "location_confidence": confidence,
        "location_evidence": evidence,
        "location_source_url": str(lead.get("location_source_url") or lead.get("source_url") or ""),
        "target_location": target,
    }


def verify_creator_niche(lead: Dict[str, Any]) -> Dict[str, Any]:
    haystack = " ".join(
        " ".join(str(item) for item in lead.get(field, [])) if isinstance(lead.get(field), list) else str(lead.get(field) or "")
        for field in ("niche", "raw_title", "raw_content", "bio_text", "value_proposition")
    ).lower()
    tags = [tag for tag, markers in _NICHE_TAGS.items() if any(marker in haystack for marker in markers)]
    pet_or_car = any(tag in tags for tag in {"Pet Creator", "Dog Parent", "Dog Trainer", "Pet Safety", "Car Gadgets"})
    adjacent = any(tag in tags for tag in {"UGC Creator", "Product Reviewer", "Amazon Finds", "Outdoor Lifestyle", "Road Trip / Camping", "PNW Lifestyle", "Local Lifestyle"})
    status = "relevant" if pet_or_car else "possibly_relevant" if adjacent else "unknown"
    return {
        "niche_tags": tags,
        "niche_verification_status": status,
        "niche_evidence": ", ".join(tags),
    }


def verify_follower_tier(lead: Dict[str, Any]) -> Dict[str, Any]:
    count = parse_follower_count(lead.get("follower_count"))
    raw_status = str(lead.get("follower_count_status") or "").strip().lower()
    if raw_status not in FOLLOWER_STATUSES:
        source = str(lead.get("source") or "").lower()
        if count is None:
            raw_status = "unknown"
        elif source in {"social_data_api", "apify", "rapidapi"}:
            raw_status = "provider_reported"
        elif source == "bulk_ingest":
            raw_status = "imported_unverified"
        else:
            raw_status = "estimated"
    return {
        "follower_count": count,
        "follower_count_status": raw_status,
        "follower_status": raw_status,
        "estimated_follower_tier": follower_tier(count),
    }


class VentoVerifier:
    """Evidence-preserving verifier with bounded, optional network checks."""

    def __init__(
        self,
        *,
        profile_fetcher: Optional[Callable[[str], Awaitable[Tuple[int, str]]]] = None,
        email_verifier: Optional[Callable[[str], Dict[str, Any]]] = None,
        enable_profile_http: Optional[bool] = None,
        enable_email_verification: Optional[bool] = None,
    ) -> None:
        self.profile_fetcher = profile_fetcher or self._fetch_profile
        self.email_verifier = email_verifier or verify_email_legitimacy
        self.enable_profile_http = (
            enable_profile_http
            if enable_profile_http is not None
            else os.getenv("VENTO_ENABLE_PROFILE_HTTP_VERIFICATION", "1").strip().lower() in {"1", "true", "yes", "on"}
        )
        self.enable_email_verification = (
            enable_email_verification
            if enable_email_verification is not None
            else os.getenv("VENTO_ENABLE_EMAIL_VERIFICATION", "1").strip().lower() in {"1", "true", "yes", "on"}
        )
        self.profile_timeout = max(2.0, min(float(os.getenv("VENTO_PROFILE_TIMEOUT_SECONDS", "8")), 20.0))
        self.profile_limit = max(0, int(os.getenv("VENTO_PROFILE_VERIFY_LIMIT", "100")))
        self.email_limit = max(0, int(os.getenv("VENTO_EMAIL_VERIFY_LIMIT", "60")))
        self.concurrency = max(1, min(int(os.getenv("VENTO_VERIFICATION_CONCURRENCY", "8")), 20))

    async def _fetch_profile(self, url: str) -> Tuple[int, str]:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; VentoCreatorVerifier/1.0)"}
        async with httpx.AsyncClient(timeout=self.profile_timeout, follow_redirects=True, headers=headers) as client:
            response = await client.get(url)
            return response.status_code, response.text[:20_000]

    async def _verify_profile_url(self, url: str) -> str:
        if not url:
            return "unknown"
        if not self.enable_profile_http:
            return "unknown"
        try:
            status_code, content = await self.profile_fetcher(url)
        except httpx.TimeoutException:
            return "rate_limited"
        except Exception:
            return "unknown"
        lower = str(content or "").lower()
        if status_code == 404:
            return "not_found"
        if status_code in {401, 403, 429}:
            return "rate_limited"
        if 200 <= status_code < 400:
            if any(marker in lower for marker in ("this account is private", "private account")):
                return "private"
            return "verified_active"
        return "unknown"

    async def _verify_email(self, email: str) -> Dict[str, Any]:
        email = str(email or "").strip().lower()
        if not email:
            return {"email": "", "email_verification_status": "not_available", "email_verification_reason": "No public email found."}
        if not _EMAIL_RE.fullmatch(email):
            return {"email": email, "email_verification_status": "invalid", "email_verification_reason": "Email syntax is invalid."}
        if not self.enable_email_verification:
            return {"email": email, "email_verification_status": "unknown", "email_verification_reason": "Email verification is disabled."}
        try:
            result = await asyncio.to_thread(self.email_verifier, email)
        except Exception as exc:
            return {"email": email, "email_verification_status": "unknown", "email_verification_reason": f"Verifier error: {type(exc).__name__}"}
        status = str(result.get("status") or "unknown").lower()
        if status not in EMAIL_STATUSES:
            status = "unknown"
        return {
            "email": email,
            "email_verification_status": status,
            "email_verification_reason": str(result.get("message") or result.get("reason") or ""),
            "email_verification": result,
        }

    async def verify_leads(
        self,
        leads: Iterable[Dict[str, Any]],
        *,
        target_location: str = "",
        strict_us_only: bool = True,
    ) -> List[Dict[str, Any]]:
        items = [dict(lead) for lead in leads if isinstance(lead, dict)]
        semaphore = asyncio.Semaphore(self.concurrency)

        async def verify_one(index: int, lead: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                profile_fields = (
                    ("instagram", "instagram_url"),
                    ("tiktok", "tiktok_url"),
                    ("youtube", "youtube_url"),
                )
                if index < self.profile_limit:
                    for platform, field in profile_fields:
                        lead[f"{platform}_profile_status"] = await self._verify_profile_url(str(lead.get(field) or ""))
                else:
                    for platform, _ in profile_fields:
                        lead.setdefault(f"{platform}_profile_status", "unknown")

                if index < self.email_limit:
                    lead.update(await self._verify_email(str(lead.get("email") or "")))
                elif lead.get("email"):
                    lead["email_verification_status"] = "unknown"
                    lead["email_verification_reason"] = "Daily verification limit reached."
                else:
                    lead["email_verification_status"] = "not_available"

                lead.update(verify_creator_location(lead, target=target_location, strict_us_only=strict_us_only))
                lead.update(verify_creator_niche(lead))
                lead.update(verify_follower_tier(lead))
                statuses = [str(lead.get(f"{platform}_profile_status") or "unknown") for platform, _ in profile_fields]
                lead["profile_verified"] = "verified_active" in statuses
                lead["profile_search_indexed"] = bool(
                    lead.get("profile_search_indexed")
                    and any(lead.get(field) for field in ("instagram_handle", "tiktok_handle", "youtube_channel"))
                )
                lead["profile_verification_status"] = (
                    "verified_active"
                    if lead["profile_verified"]
                    else "private"
                    if "private" in statuses
                    else "search_indexed"
                    if lead["profile_search_indexed"]
                    else "unknown"
                )
                lead["brand_safety_flags"] = [
                    marker for marker in _BRAND_SAFETY_CRITICAL
                    if marker in " ".join(str(lead.get(key) or "") for key in ("raw_content", "bio_text", "notes")).lower()
                ]
                lead["last_verified_at"] = utc_now()
                lead.setdefault("profile_checked_at", lead["last_verified_at"])
                return lead

        return await asyncio.gather(*(verify_one(index, lead) for index, lead in enumerate(items)))


def source_domain(url: str) -> str:
    return (urlparse(str(url or "")).hostname or "").lower()
