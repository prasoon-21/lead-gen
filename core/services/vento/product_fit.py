from __future__ import annotations

from typing import Any, Dict, Iterable


_PET = (
    "pet creator", "pet influencer", "dog mom", "dog dad", "dog parent", "cat mom", "cat parent",
    "puppy", "golden retriever", "cavapoo", "pitbull", "dog owner", "pet owner", "pets", "dogs",
)
_VEHICLE = ("car", "vehicle", "automotive", "auto accessory", "car accessory", "car gadget", "parked car")
_HEAT_SAFETY = (
    "heat", "hot car", "cooling", "keep cool", "keeping cool", "summer", "pet safety", "dog safety",
    "temperature", "overheat", "comfort fan",
)
_TRAVEL_OUTDOOR = (
    "road trip", "roadtrip", "camping", "outdoor", "hiking", "adventure", "travel with", "van life",
)
_REVIEW = ("product review", "reviewer", "product testing", "amazon finds", "car essentials", "gadget")
_UGC = ("ugc", "user generated content", "content creator", "brand content", "testimonial")


def _text(lead: Dict[str, Any]) -> str:
    values: list[str] = []
    for field in (
        "creator_name", "niche", "raw_title", "raw_content", "bio_text", "niche_evidence", "value_proposition",
    ):
        values.append(str(lead.get(field) or ""))
    for field in ("niche_tags", "tags"):
        value = lead.get(field)
        if isinstance(value, (list, tuple, set)):
            values.extend(str(item) for item in value)
        else:
            values.append(str(value or ""))
    return " ".join(values).lower()


def _profile_and_post_text(lead: Dict[str, Any]) -> tuple[str, str]:
    profile_values = [
        str(lead.get(field) or "")
        for field in ("creator_name", "niche", "bio_text", "profile_enrichment_evidence", "website")
    ]
    post_values = [str(lead.get(field) or "") for field in ("raw_title", "raw_content")]
    source_scope = str(lead.get("identity_source_scope") or "")
    if source_scope == "profile":
        profile_values.extend(post_values)
    elif source_scope != "post":
        value = lead.get("niche_tags")
        if isinstance(value, (list, tuple, set)):
            profile_values.extend(str(item) for item in value)
    return " ".join(profile_values).lower(), " ".join(post_values).lower()


def _present(text: str, markers: Iterable[str]) -> bool:
    return any(marker in text for marker in markers)


def _evidence(text: str, markers: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(marker for marker in markers if marker in text))[:8]


def evaluate_product_fit(raw: Dict[str, Any]) -> Dict[str, Any]:
    lead = dict(raw)
    text = _text(lead)
    profile_text, post_text = _profile_and_post_text(lead)
    creator_type = str(lead.get("creator_type") or "")
    excluded_entity = creator_type in {
        "directory_or_listicle", "corporate_brand", "store_or_shop", "media_or_publisher", "unknown_entity",
    }
    pet_profile = _present(profile_text, _PET)
    pet_post = _present(post_text, _PET)
    pet = pet_profile or pet_post or any(
        tag in set(lead.get("niche_tags") or [])
        for tag in ("Pet Creator", "Dog Parent", "Dog Trainer", "Pet Safety")
    )
    vehicle = _present(text, _VEHICLE) or "Car Gadgets" in set(lead.get("niche_tags") or [])
    vehicle_profile = _present(profile_text, _VEHICLE)
    vehicle_post = _present(post_text, _VEHICLE)
    heat = _present(text, _HEAT_SAFETY)
    travel = _present(text, _TRAVEL_OUTDOOR) or any(
        tag in set(lead.get("niche_tags") or []) for tag in ("Road Trip / Camping", "Outdoor Lifestyle")
    )
    review = _present(text, _REVIEW) or "Product Reviewer" in set(lead.get("niche_tags") or [])
    ugc = _present(text, _UGC) or "UGC Creator" in set(lead.get("niche_tags") or [])

    tags: list[str] = []
    reasons: list[str] = []
    score = 0
    level = "P5"
    category = "influencer"
    requires_profile_confirmation = False

    if excluded_entity:
        reasons.append(f"excluded_entity:{creator_type}")
    elif str(lead.get("identity_source_scope") or "") == "post" and not pet_profile and pet_post:
        # One relevant post is useful discovery evidence, but it does not prove
        # that the account has a recurring pet audience.
        score, level = 56, "P3"
        requires_profile_confirmation = True
        tags.append("Single-Post Pet Relevance")
        reasons.append("single_post_requires_profile_niche_confirmation")
    elif str(lead.get("identity_source_scope") or "") == "post" and not vehicle_profile and vehicle_post:
        score, level = 48, "P4"
        requires_profile_confirmation = True
        tags.append("Single-Post Automotive Relevance")
        reasons.append("single_post_requires_profile_niche_confirmation")
    elif pet and heat and (vehicle or travel):
        score, level = 94, "P1"
        tags.append("Pet + Vehicle Heat/Safety")
        reasons.append("pet_vehicle_heat_safety_evidence")
    elif pet and vehicle:
        score, level = 88, "P1"
        tags.append("Pet + Vehicle")
        reasons.append("pet_vehicle_evidence")
    elif pet and (travel or ugc or review):
        score, level = 78, "P2"
        tags.append("Pet + Travel/UGC/Review")
        reasons.append("pet_adjacent_campaign_evidence")
    elif vehicle and pet:
        score, level = 76, "P2"
        tags.append("Automotive + Pet Ownership")
        reasons.append("automotive_pet_owner_evidence")
    elif pet:
        score, level = 64, "P3"
        tags.append("Established Pet Relevance")
        reasons.append("pet_creator_evidence")
    elif vehicle and (review or ugc):
        score, level = 60, "P3"
        tags.append("Automotive Product Demonstration")
        reasons.append("automotive_review_evidence")
    elif ugc:
        score, level, category = 38, "P4", "paid_ugc"
        tags.append("Generic Paid UGC")
        reasons.append("generic_ugc_without_vento_intersection")
    elif vehicle or travel or review:
        score, level = 34, "P4"
        tags.append("Adjacent Lifestyle/Automotive")
        reasons.append("adjacent_without_pet_evidence")
    else:
        reasons.append("vento_product_evidence_missing")

    if creator_type == "agency_or_manager":
        category = "paid_ugc"
        score = min(score or 30, 45)
        level = "P4"
        reasons.append("agency_or_manager_not_influencer")
    elif creator_type == "ugc_creator" and level in {"P4", "P5"}:
        category = "paid_ugc"

    evidence_markers = _evidence(text, (*_PET, *_VEHICLE, *_HEAT_SAFETY, *_TRAVEL_OUTDOOR, *_REVIEW, *_UGC))
    return {
        "product_fit_level": level,
        "product_fit_score": score,
        "product_fit_tags": tags,
        "product_fit_evidence": evidence_markers,
        "product_fit_reasons": list(dict.fromkeys(reasons)),
        "lead_category": category,
        "product_fit_requires_profile_confirmation": requires_profile_confirmation,
    }


def apply_product_fit(leads: Iterable[Dict[str, Any]]) -> list[Dict[str, Any]]:
    results: list[Dict[str, Any]] = []
    for raw in leads:
        lead = dict(raw)
        lead.update(evaluate_product_fit(lead))
        results.append(lead)
    return results
