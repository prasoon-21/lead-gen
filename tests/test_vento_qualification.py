from core.services.vento.qualification import (
    classify_quality_level,
    deduplicate_creators,
    qualify_leads,
)


def base_lead(**updates):
    lead = {
        "creator_name": "Jane Dog",
        "instagram_handle": "@janedog",
        "instagram_profile_status": "verified_active",
        "tiktok_profile_status": "unknown",
        "youtube_profile_status": "unknown",
        "profile_verified": True,
        "email": "jane@example.com",
        "email_verification_status": "valid",
        "location": "Seattle, WA, US",
        "location_confidence": "confirmed",
        "location_region": "PNW",
        "niche_verification_status": "relevant",
        "niche_tags": ["Pet Creator", "Dog Parent"],
        "follower_count": 12000,
        "follower_count_status": "provider_reported",
        "estimated_follower_tier": "nano",
        "brand_safety_flags": [],
    }
    lead.update(updates)
    return lead


def test_level_a_and_b_classification():
    level, _ = classify_quality_level(base_lead())
    assert level == "A"
    level, _ = classify_quality_level(base_lead(email="", email_verification_status="not_available"))
    assert level == "B"


def test_shared_management_email_does_not_merge_creators():
    first = base_lead(creator_name="One", instagram_handle="@one", email="management@agency.com")
    second = base_lead(creator_name="Two", instagram_handle="@two", email="management@agency.com")
    unique, stats = deduplicate_creators([first, second])
    assert len(unique) == 2
    assert stats["same_run_duplicates"] == 0


def test_same_instagram_identity_merges_records():
    first = base_lead(email="")
    second = base_lead(email="jane@example.com", tiktok_handle="@janetok")
    unique, stats = deduplicate_creators([first, second])
    assert len(unique) == 1
    assert unique[0]["email"] == "jane@example.com"
    assert unique[0]["tiktok_handle"] == "@janetok"
    assert stats["same_run_duplicates"] == 1


def test_missing_follower_evidence_is_not_forced_into_requested_tier():
    lead = base_lead(follower_count=None, follower_count_status="unknown", estimated_follower_tier="unknown")
    result = qualify_leads([lead], requested_follower_tier="mid")[0]
    assert result["quality_level"] == "C"
    assert "follower_tier_unverified_or_mismatch" in result["rejection_reasons"]


def test_pet_mode_moves_general_road_trip_creator_to_review():
    travel = base_lead(
        email="",
        email_verification_status="not_available",
        niche_verification_status="possibly_relevant",
        niche_tags=["Road Trip / Camping", "PNW Lifestyle"],
    )
    result = qualify_leads([travel], requested_influencer_type="pet")[0]
    assert result["quality_level"] == "C"
    assert "selected_influencer_type_mismatch" in result["rejection_reasons"]


def test_tiktok_only_exact_pet_creator_can_be_dm_ready():
    tiktok_creator = base_lead(
        instagram_handle="",
        instagram_profile_status="unknown",
        tiktok_handle="@dogtok",
        tiktok_profile_status="verified_active",
        email="",
        email_verification_status="not_available",
        niche_tags=["Pet Creator", "Dog Parent"],
    )
    result = qualify_leads([tiktok_creator], requested_influencer_type="pet")[0]
    assert result["quality_level"] == "B"
    assert result["preferred_contact_method"] == "tiktok_dm"


def test_road_trip_creator_with_real_dog_parent_evidence_can_qualify_pet():
    creator = base_lead(
        email="",
        email_verification_status="not_available",
        niche_tags=["Road Trip / Camping", "Dog Parent"],
    )
    result = qualify_leads([creator], requested_influencer_type="pet")[0]
    assert result["quality_level"] == "B"


def test_tavily_indexed_pet_profile_is_usable_without_being_marked_verified():
    creator = base_lead(
        email="",
        email_verification_status="not_available",
        instagram_profile_status="rate_limited",
        profile_verified=False,
        profile_search_indexed=True,
        search_target_location="Los Angeles",
        location="",
        location_confidence="unknown",
    )
    result = qualify_leads([creator], requested_influencer_type="pet")[0]
    assert result["quality_level"] == "B"
    assert result["profile_verified"] is False
    assert "Profile Indexed" in result["tags"]
    assert "Profile Verified" not in result["tags"]
    assert "Location Review Required" in result["tags"]
    assert "location_requires_manual_review" in result["rejection_reasons"]
