from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urlparse

from core.services.vento.discovery import normalize_social_handle


SCORING_VERSION = "vento-1.0"
_SHARED_CONTACT_PREFIXES = {"agency", "bookings", "collab", "collabs", "contact", "hello", "info", "management", "partnerships", "press", "team"}
_SHARED_DOMAINS = {
    "instagram.com", "tiktok.com", "youtube.com", "youtu.be", "linktr.ee", "beacons.ai",
    "linkin.bio", "bio.link", "stan.store", "koji.to",
}


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _email_identity(email: str, lead: Optional[Dict[str, Any]] = None) -> str:
    email = _clean(email).lower()
    if "@" not in email:
        return ""
    local, domain = email.split("@", 1)
    phone_type = _clean((lead or {}).get("phone_type")).lower()
    preferred = _clean((lead or {}).get("preferred_contact_method")).lower()
    if local in _SHARED_CONTACT_PREFIXES or phone_type == "management" or preferred == "management":
        return ""
    return f"email:{local}@{domain}"


def creator_identity_keys(lead: Dict[str, Any]) -> List[str]:
    keys: List[str] = []
    creator_id = _clean(lead.get("creator_id"))
    if creator_id:
        keys.append(f"creator:{creator_id.lower()}")
    for platform, field in (
        ("instagram", "instagram_handle"),
        ("tiktok", "tiktok_handle"),
        ("youtube", "youtube_channel"),
    ):
        handle = normalize_social_handle(str(lead.get(field) or ""), platform)
        if handle:
            keys.append(f"{platform}:{handle}")

    provider_id = _clean(lead.get("source_record_id"))
    if provider_id:
        keys.append(f"provider:{provider_id.lower()}")

    email_key = _email_identity(str(lead.get("email") or ""), lead)
    if email_key:
        keys.append(email_key)

    website = _clean(lead.get("website"))
    creator_name = _clean(lead.get("creator_name")).lower()
    domain = (urlparse(website).hostname or "").lower()
    if domain.startswith("www."):
        domain = domain[4:]
    if domain and domain not in _SHARED_DOMAINS and creator_name:
        keys.append(f"site:{domain}|{creator_name}")
    return list(dict.fromkeys(keys))


def build_creator_id(lead: Dict[str, Any]) -> str:
    existing = _clean(lead.get("creator_id"))
    if existing:
        return existing
    keys = creator_identity_keys(lead)
    if not keys:
        fallback = "|".join(
            _clean(lead.get(field)).lower()
            for field in ("creator_name", "source_url", "website")
            if _clean(lead.get(field))
        )
        if not fallback:
            return ""
        keys = [f"fallback:{fallback}"]
    return hashlib.sha256(keys[0].encode("utf-8")).hexdigest()


def merge_creator_records(existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(existing)
    for key, value in incoming.items():
        if value in (None, "", [], {}):
            continue
        if merged.get(key) in (None, "", [], {}):
            merged[key] = value
        elif key in {"niche_tags", "tags", "brand_safety_flags", "rejection_reasons", "source_urls"}:
            old_values = merged.get(key) if isinstance(merged.get(key), list) else [merged.get(key)]
            new_values = value if isinstance(value, list) else [value]
            merged[key] = list(dict.fromkeys(item for item in old_values + new_values if item))
    merged_sources = []
    for value in (existing.get("source"), incoming.get("source")):
        values = value if isinstance(value, list) else [value]
        merged_sources.extend(_clean(item) for item in values if _clean(item))
    if merged_sources:
        merged["source"] = list(dict.fromkeys(merged_sources))
    return merged


def deduplicate_creators(
    leads: Iterable[Dict[str, Any]],
    *,
    historical_identity_keys: Optional[Set[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    registry: Dict[str, int] = {}
    historical = {str(key).lower() for key in (historical_identity_keys or set()) if key}
    unique: List[Dict[str, Any]] = []
    stats = {"same_run_duplicates": 0, "historical_duplicates": 0, "merged_records": 0}

    for raw in leads:
        lead = dict(raw)
        keys = creator_identity_keys(lead)
        if keys and any(key.lower() in historical for key in keys):
            stats["historical_duplicates"] += 1
            continue
        duplicate_index = next((registry[key] for key in keys if key in registry), None)
        if duplicate_index is not None:
            unique[duplicate_index] = merge_creator_records(unique[duplicate_index], lead)
            stats["same_run_duplicates"] += 1
            stats["merged_records"] += 1
            for key in creator_identity_keys(unique[duplicate_index]):
                registry[key] = duplicate_index
            continue
        lead["creator_id"] = build_creator_id(lead)
        position = len(unique)
        unique.append(lead)
        for key in keys:
            registry[key] = position
    return unique, stats


def build_vento_tags(lead: Dict[str, Any]) -> List[str]:
    tags: List[str] = []
    country = _clean(lead.get("location_country"))
    state = _clean(lead.get("location_state"))
    city = _clean(lead.get("location_city"))
    region = _clean(lead.get("location_region"))
    tags.extend(value for value in (country, state, city, region) if value)
    tags.extend(str(value) for value in lead.get("niche_tags", []) if value)
    if lead.get("instagram_handle"):
        tags.append("Instagram")
    if lead.get("tiktok_handle"):
        tags.append("TikTok")
    if lead.get("youtube_channel"):
        tags.append("YouTube")
    tier_label = {
        "nano": "Nano",
        "mid": "Mid-Tier",
        "upper_mid": "Upper Mid-Tier",
        "below_nano": "Below 5K",
        "above_target": "Above 500K",
    }.get(_clean(lead.get("estimated_follower_tier")))
    if tier_label:
        tags.append(tier_label)
    if lead.get("profile_verified"):
        tags.append("Profile Verified")
    elif lead.get("profile_search_indexed"):
        tags.append("Profile Indexed")
    location_confidence = _clean(lead.get("location_confidence"))
    if location_confidence in {"confirmed", "likely"}:
        tags.append(f"Location {location_confidence.title()}")
    elif location_confidence == "unknown" and lead.get("search_target_location"):
        tags.append("Location Review Required")
    if _clean(lead.get("email_verification_status")) == "valid":
        tags.append("Email Verified")
    if any(lead.get(field) for field in ("instagram_handle", "tiktok_handle", "youtube_channel")):
        tags.append("DM Ready")
    product_level = _clean(lead.get("product_fit_level"))
    if product_level:
        tags.append(f"Vento Fit {product_level}")
    tags.extend(str(value) for value in lead.get("product_fit_tags", []) if value)
    creator_type = _clean(lead.get("creator_type"))
    if creator_type:
        tags.append(creator_type.replace("_", " ").title())
    return list(dict.fromkeys(tag for tag in tags if tag))


def _clamp(value: float) -> float:
    return round(max(0.0, min(100.0, value)), 1)


def calculate_vento_scores(lead: Dict[str, Any]) -> Dict[str, Any]:
    niche_status = _clean(lead.get("niche_verification_status"))
    niche_score = {"relevant": 30, "possibly_relevant": 18, "unknown": 5, "irrelevant": 0}.get(niche_status, 5)

    email_status = _clean(lead.get("email_verification_status"))
    has_dm = any(lead.get(field) for field in ("instagram_handle", "tiktok_handle", "youtube_channel"))
    contact_score = {"valid": 25, "risky": 14, "unknown": 8, "not_available": 0, "invalid": 0}.get(email_status, 0)
    if has_dm:
        contact_score = max(contact_score, 15)
    if lead.get("management_contact") or lead.get("contact_page_url"):
        contact_score = min(25, contact_score + 5)

    profile_statuses = [
        _clean(lead.get("instagram_profile_status")),
        _clean(lead.get("tiktok_profile_status")),
        _clean(lead.get("youtube_profile_status")),
    ]
    profile_score = (
        15
        if "verified_active" in profile_statuses
        else 8
        if "private" in profile_statuses or lead.get("profile_search_indexed")
        else 0
    )

    location_confidence = _clean(lead.get("location_confidence"))
    location_score = {"confirmed": 15, "likely": 12, "uncertain": 5, "unknown": 0, "outside_target": 0}.get(location_confidence, 0)
    if _clean(lead.get("location_region")) == "PNW" and location_confidence in {"confirmed", "likely"}:
        location_score = 15

    follower_status = _clean(lead.get("follower_count_status"))
    tier = _clean(lead.get("estimated_follower_tier"))
    follower_score = 0
    if follower_status in {"verified", "provider_verified", "provider_reported", "search_observed", "imported_unverified"}:
        follower_score = 10 if tier in {"nano", "mid", "upper_mid"} else 3
    elif follower_status == "estimated":
        follower_score = 4

    content_score = 0 if lead.get("brand_safety_flags") else 5
    total = niche_score + contact_score + profile_score + location_score + follower_score + content_score
    verification_score = _clamp((profile_score / 15 * 40) + (location_score / 15 * 30) + (25 if email_status == "valid" else 10 if email_status == "risky" else 0) + (5 if follower_score >= 8 else 0))
    contactability_score = _clamp(contact_score / 25 * 100)
    relevance_score = _clamp((niche_score / 30 * 70) + (location_score / 15 * 20) + (follower_score / 10 * 10))
    return {
        "relevance_score": round(relevance_score / 10, 1),
        "contactability_score": contactability_score,
        "verification_score": verification_score,
        "overall_score": _clamp(total),
        "quality_score": int(_clamp(total)),
        "score_breakdown": {
            "niche_fit": niche_score,
            "contactability": contact_score,
            "profile_activity": profile_score,
            "location_fit": location_score,
            "audience_fit": follower_score,
            "content_safety": content_score,
        },
        "scoring_version": SCORING_VERSION,
    }


def matches_follower_tier(lead: Dict[str, Any], requested_tier: str) -> bool:
    requested = _clean(requested_tier).lower() or "all"
    if requested == "all":
        return True
    status = _clean(lead.get("follower_count_status"))
    if status not in {"verified", "provider_verified", "provider_reported", "search_observed", "imported_unverified"}:
        return False
    return _clean(lead.get("estimated_follower_tier")) == requested


def classify_quality_level(lead: Dict[str, Any], *, strict_us_only: bool = True) -> Tuple[str, List[str]]:
    reasons: List[str] = []
    profile_active = bool(lead.get("profile_verified"))
    profile_indexed = bool(lead.get("profile_search_indexed"))
    niche_status = _clean(lead.get("niche_verification_status"))
    location = _clean(lead.get("location_confidence"))
    email_status = _clean(lead.get("email_verification_status"))
    has_dm = any(lead.get(field) for field in ("instagram_handle", "tiktok_handle", "youtube_channel"))
    if lead.get("brand_safety_flags"):
        reasons.append("critical_brand_safety_flag")
    if not has_dm and not lead.get("email") and not lead.get("management_contact"):
        reasons.append("no_contact_path")
    if all(_clean(lead.get(field)) in {"", "not_found", "unknown"} for field in (
        "instagram_profile_status", "tiktok_profile_status", "youtube_profile_status"
    )):
        reasons.append("profile_not_verified")
    if niche_status in {"irrelevant", "unknown"}:
        reasons.append("niche_not_verified")
    if strict_us_only and location == "outside_target":
        reasons.append("outside_target_location")
    if email_status == "invalid" and not has_dm:
        reasons.append("invalid_only_contact")

    critical = {"critical_brand_safety_flag", "no_contact_path", "outside_target_location", "invalid_only_contact"}
    if critical.intersection(reasons):
        return "Rejected", reasons
    if profile_active and niche_status == "relevant" and location in {"confirmed", "likely"} and email_status == "valid":
        return "A", reasons
    if profile_active and niche_status in {"relevant", "possibly_relevant"} and location in {"confirmed", "likely", "uncertain"} and has_dm:
        return "B", reasons
    # Social platforms commonly return 403/429 to server-side checks even when
    # their public profile URL is present in Tavily's current search index. Such
    # a profile is useful for DM outreach, but it must never be labeled verified.
    # A target-specific search may therefore produce a Level B lead with an
    # explicit location-review tag when no contradictory location was found.
    if (
        profile_indexed
        and niche_status in {"relevant", "possibly_relevant"}
        and location != "outside_target"
        and has_dm
        and bool(_clean(lead.get("search_target_location")))
    ):
        if location == "unknown":
            reasons.append("location_requires_manual_review")
        if not profile_active:
            reasons.append("profile_indexed_not_http_verified")
        return "B", reasons
    return "C", reasons


def matches_requested_influencer_type(lead: Dict[str, Any], requested_type: str) -> bool:
    requested = _clean(requested_type).lower() or "all"
    if requested == "all":
        return True
    tags = {str(tag) for tag in (lead.get("niche_tags") or []) if tag}
    required_tags = {
        "pet": {"Pet Creator", "Dog Parent", "Dog Trainer", "Pet Safety"},
        "gadget": {"Car Gadgets", "Product Reviewer", "Amazon Finds"},
        "lifestyle": {"PNW Lifestyle", "Outdoor Lifestyle", "Road Trip / Camping", "Local Lifestyle"},
    }
    return bool(tags.intersection(required_tags.get(requested, set())))


def classify_quality_v2(
    lead: Dict[str, Any],
    *,
    strict_us_only: bool,
    quality_mode: str,
    minimum_product_fit: str,
    require_email: bool,
    require_followers: bool,
    requested_follower_tier: str,
) -> Tuple[str, List[str], List[str]]:
    reasons: List[str] = []
    review_reasons: List[str] = []
    creator_type = _clean(lead.get("creator_type"))
    excluded = {
        "directory_or_listicle", "corporate_brand", "store_or_shop", "media_or_publisher", "unknown_entity",
    }
    if creator_type in excluded or not bool(lead.get("individual_creator")):
        reasons.append(f"excluded_creator_type:{creator_type or 'unknown'}")
        return "Rejected", reasons, review_reasons
    if not bool(lead.get("canonical_social_present")):
        reasons.append("canonical_instagram_or_tiktok_required")
        return "Rejected", reasons, review_reasons
    if lead.get("brand_safety_flags"):
        reasons.append("critical_brand_safety_flag")
        return "Rejected", reasons, review_reasons

    minimum_scores = {"P1": 85, "P2": 70, "P3": 55, "P4": 25, "P5": 0}
    minimum = minimum_scores.get(_clean(minimum_product_fit).upper(), 55)
    product_score = int(lead.get("product_fit_score") or 0)
    if product_score < minimum:
        review_reasons.append("product_fit_below_requested_minimum")
    if lead.get("product_fit_requires_profile_confirmation"):
        review_reasons.append("product_fit_requires_profile_confirmation")

    location = _clean(lead.get("location_confidence"))
    if location == "outside_target":
        reasons.append("outside_target_location")
        return "Rejected", reasons, review_reasons
    if location == "conflicting":
        review_reasons.append("location_evidence_conflicting")
    if strict_us_only and location == "unknown":
        review_reasons.append("location_unknown_in_strict_us_mode")

    has_dm = bool(lead.get("instagram_handle") or lead.get("tiktok_handle"))
    profile_active = bool(lead.get("profile_verified"))
    profile_indexed = bool(lead.get("profile_search_indexed"))
    email_valid = _clean(lead.get("email_verification_status")) == "valid"
    follower_known = _clean(lead.get("follower_count_status")) in {
        "verified", "provider_verified", "provider_reported", "search_observed", "imported_unverified",
    }
    if require_email and not email_valid:
        review_reasons.append("valid_email_required")
    if require_followers and not follower_known:
        review_reasons.append("known_follower_count_required")
    if requested_follower_tier != "all" and not matches_follower_tier(lead, requested_follower_tier):
        review_reasons.append("follower_tier_unverified_or_mismatch")
    if not has_dm:
        reasons.append("instagram_or_tiktok_dm_required")
        return "Rejected", reasons, review_reasons
    if not (profile_active or profile_indexed):
        review_reasons.append("profile_not_verified_or_indexed")

    if lead.get("lead_category") == "paid_ugc":
        review_reasons.append("paid_ugc_separate_from_influencer")

    if review_reasons:
        if quality_mode == "balanced":
            balanced_blockers = {
                "product_fit_below_requested_minimum", "location_evidence_conflicting", "valid_email_required",
                "known_follower_count_required", "follower_tier_unverified_or_mismatch",
                "profile_not_verified_or_indexed", "paid_ugc_separate_from_influencer",
            }
            if balanced_blockers.intersection(review_reasons):
                return "C", reasons, review_reasons
            if product_score >= minimum and (profile_active or profile_indexed):
                reasons.extend(review_reasons)
                return "B", reasons, review_reasons
        return "C", reasons, review_reasons

    location_ready = location in {"confirmed", "likely", "uncertain"} or not strict_us_only
    if profile_active and email_valid and location_ready and product_score >= minimum:
        reasons.extend(("canonical_social_profile", "vento_product_fit", "valid_public_email"))
        return "A", reasons, review_reasons
    if (profile_active or profile_indexed) and has_dm and location_ready and product_score >= minimum:
        reasons.extend(("canonical_social_profile", "vento_product_fit", "dm_available"))
        if profile_indexed and not profile_active:
            reasons.append("profile_indexed_not_http_verified")
        return "B", reasons, review_reasons
    review_reasons.append("minimum_usable_evidence_not_met")
    return "C", reasons, review_reasons


def qualify_leads(
    leads: Iterable[Dict[str, Any]],
    *,
    requested_follower_tier: str = "all",
    requested_influencer_type: str = "all",
    strict_us_only: bool = True,
    quality_v2_enabled: bool = False,
    quality_mode: str = "strict",
    minimum_product_fit: str = "P3",
    include_paid_ugc: bool = True,
    require_email: bool = False,
    require_followers: bool = False,
) -> List[Dict[str, Any]]:
    qualified: List[Dict[str, Any]] = []
    for raw in leads:
        lead = dict(raw)
        lead.update(calculate_vento_scores(lead))
        if quality_v2_enabled:
            level, reasons, review_reasons = classify_quality_v2(
                lead,
                strict_us_only=strict_us_only,
                quality_mode=quality_mode,
                minimum_product_fit=minimum_product_fit,
                require_email=require_email,
                require_followers=require_followers,
                requested_follower_tier=requested_follower_tier,
            )
        else:
            level, reasons = classify_quality_level(lead, strict_us_only=strict_us_only)
            review_reasons = []
        if not matches_requested_influencer_type(lead, requested_influencer_type):
            if level in {"A", "B"}:
                level = "C"
            reasons.append("selected_influencer_type_mismatch")
        if not quality_v2_enabled and requested_follower_tier != "all" and not matches_follower_tier(lead, requested_follower_tier):
            level = "C" if lead.get("follower_count") is None else "Rejected"
            reasons.append("follower_tier_unverified_or_mismatch")
        if quality_v2_enabled and lead.get("lead_category") == "paid_ugc" and not include_paid_ugc:
            level = "Rejected"
            reasons.append("paid_ugc_excluded_by_request")
        lead["quality_level"] = level
        lead["rejection_reasons"] = list(dict.fromkeys(reasons))
        lead["quality_reasons"] = list(dict.fromkeys(reasons))
        lead["review_reasons"] = list(dict.fromkeys(review_reasons))
        lead["usable"] = level in {"A", "B"}
        lead["rejected"] = level == "Rejected"
        lead["tags"] = build_vento_tags(lead)
        if level == "A":
            lead["tags"].extend(tag for tag in ("Email Ready", "High Priority") if tag not in lead["tags"])
            lead["preferred_contact_method"] = "email"
        elif level == "B":
            lead["preferred_contact_method"] = "instagram_dm" if lead.get("instagram_handle") else "tiktok_dm" if lead.get("tiktok_handle") else "youtube"
        qualified.append(lead)
    qualified.sort(key=lambda item: ({"A": 3, "B": 2, "C": 1, "Rejected": 0}.get(item.get("quality_level"), 0), item.get("overall_score", 0)), reverse=True)
    return qualified
