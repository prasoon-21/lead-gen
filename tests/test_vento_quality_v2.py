import asyncio

from core.services.vento.creator_enrichment import AdaptiveCreatorEnricher
from core.services.vento.discovery import build_quality_vento_queries
from core.services.vento.identity_quality import canonicalize_creator_identity
from core.services.vento.product_fit import evaluate_product_fit
from core.services.vento.qualification import qualify_leads
from core.services.vento.verification import verify_creator_location


def test_post_url_becomes_canonical_tiktok_profile_and_title_name_is_replaced():
    result = canonicalize_creator_identity(
        {
            "creator_name": "Keeping Cool with My Dog in SoCal Heat",
            "source": "tavily_fallback",
            "source_url": "https://www.tiktok.com/@luvnthecru/video/7550392705004604703",
            "raw_title": "Keeping Cool with My Dog in SoCal Heat | TikTok",
            "raw_content": "Dog mom keeping the dogs comfortable during hot California car travel.",
        }
    )
    assert result["tiktok_handle"] == "@luvnthecru"
    assert result["tiktok_url"] == "https://www.tiktok.com/@luvnthecru"
    assert result["discovery_source_url"].endswith("7550392705004604703")
    assert result["creator_name"] == "Luvnthecru"
    assert result["identity_confidence"] == "high"


def test_directory_corporate_store_and_media_entities_are_not_individual_creators():
    fixtures = [
        ({"creator_name": "Top 40 Auto TikTok Influencers in 2026", "source_url": "https://creators.feedspot.com/auto_tiktok_influencers"}, "directory_or_listicle"),
        ({"creator_name": "Roverdotcom", "instagram_handle": "@roverdotcom", "raw_content": "Official Rover pet care company"}, "corporate_brand"),
        ({"creator_name": "Auto Car Accessories", "instagram_handle": "@cargadget.ae", "raw_content": "Shop car accessories UAE worldwide shipping"}, "store_or_shop"),
        ({"creator_name": "De Los Angeles Times", "tiktok_handle": "@delosangelestimes", "raw_content": "news correspondent publication"}, "media_or_publisher"),
    ]
    for lead, expected in fixtures:
        result = canonicalize_creator_identity(lead)
        assert result["creator_type"] == expected
        assert result["individual_creator"] is False


def test_location_parser_does_not_create_states_from_income_me_or_animal_id():
    result = verify_creator_location(
        {
            "raw_content": "Helping you turn content into income. Join me as I foster dog Los Angeles, animal ID A2257886.",
            "source": "tavily_fallback",
        },
        target="Los Angeles",
    )
    assert result["location_state"] == "CA"
    assert result["location_city"] == "Los Angeles"
    assert "IN" not in result["location"]
    assert "ME" not in result["location"]
    assert "ID" not in result["location"]


def test_query_location_is_not_creator_location_evidence():
    result = verify_creator_location(
        {
            "raw_content": "Pet creator and dog product reviewer",
            "search_target_location": "Los Angeles, CA",
            "source": "tavily_fallback",
        },
        target="Los Angeles, CA",
    )
    assert result["location_confidence"] == "unknown"


def test_pet_plus_heat_vehicle_evidence_is_p1_product_fit():
    result = evaluate_product_fit(
        {
            "raw_title": "Keeping Cool with My Dog in SoCal Heat",
            "raw_content": "Dog mom shows cooling and pet safety during car travel in summer heat.",
            "niche_tags": ["Pet Creator", "Dog Parent"],
        }
    )
    assert result["product_fit_level"] == "P1"
    assert result["product_fit_score"] >= 85
    assert "Pet + Vehicle Heat/Safety" in result["product_fit_tags"]


def test_generic_ugc_is_paid_ugc_review_not_influencer_fit():
    result = evaluate_product_fit(
        {
            "creator_name": "Generic UGC Creator",
            "raw_content": "UGC creator helping brands make short form content",
            "niche_tags": ["UGC Creator"],
        }
    )
    assert result["product_fit_level"] == "P4"
    assert result["lead_category"] == "paid_ugc"


def test_masked_email_is_never_preserved():
    result = canonicalize_creator_identity(
        {
            "creator_name": "Creator",
            "instagram_handle": "@creator",
            "email": "****@web.detiktok",
        }
    )
    assert result["email"] == ""


class EnrichmentTavily:
    def __init__(self):
        self.calls = []

    async def search(self, payload):
        self.calls.append(payload["query"])
        return {
            "results": [
                {
                    "url": "https://www.instagram.com/strongdog/",
                    "title": "Strong Dog Creator",
                    "content": "@strongdog TikTok @strongdogtok business strongdog@example.com 25K followers Seattle, WA",
                }
            ]
        }


def test_one_adaptive_enrichment_search_can_recover_platform_email_follower_and_location(tmp_path):
    tavily = EnrichmentTavily()
    enricher = AdaptiveCreatorEnricher(tavily_client=tavily, cache_path=tmp_path / "cache.json")
    lead = {
        "creator_name": "Strong Dog",
        "instagram_handle": "@strongdog",
        "instagram_url": "https://www.instagram.com/strongdog/",
        "product_fit_score": 80,
        "raw_content": "Dog parent and car travel creator",
    }
    enriched = asyncio.run(enricher.enrich([lead], target_location="Seattle, WA"))[0]
    assert len(tavily.calls) == 1
    assert enriched["tiktok_handle"] == "@strongdogtok"
    assert enriched["email"] == "strongdog@example.com"
    assert enriched["follower_count"] == 25000
    assert enriched["follower_status"] == "search_observed"
    assert "Seattle" in enriched["location_evidence"]


def test_quality_queries_start_with_product_intersections_not_generic_ugc():
    queries = build_quality_vento_queries(location="Los Angeles", influencer_type="pet")
    assert "dog mom" in queries[0]
    assert "car" in queries[0]
    assert all("generic UGC" not in query for query in queries)


def _quality_lead(**updates):
    lead = {
        "creator_name": "Strong Dog",
        "creator_type": "pet_creator",
        "individual_creator": True,
        "canonical_social_present": True,
        "instagram_handle": "@strongdog",
        "instagram_profile_status": "verified_active",
        "profile_verified": True,
        "profile_search_indexed": True,
        "location": "Seattle, WA, US",
        "location_confidence": "likely",
        "location_country": "US",
        "niche_tags": ["Pet Creator", "Dog Parent"],
        "niche_verification_status": "relevant",
        "product_fit_level": "P2",
        "product_fit_score": 78,
        "product_fit_tags": ["Pet + Travel/UGC/Review"],
        "lead_category": "influencer",
        "email": "",
        "email_verification_status": "not_available",
        "follower_count": None,
        "follower_count_status": "unknown",
        "brand_safety_flags": [],
    }
    lead.update(updates)
    return lead


def test_quality_v2_strict_unknown_location_is_review_but_balanced_can_be_dm_ready():
    lead = _quality_lead(location="", location_confidence="unknown")
    strict = qualify_leads([lead], quality_v2_enabled=True, quality_mode="strict")[0]
    balanced = qualify_leads([lead], quality_v2_enabled=True, quality_mode="balanced")[0]
    assert strict["quality_level"] == "C"
    assert "location_unknown_in_strict_us_mode" in strict["review_reasons"]
    assert balanced["quality_level"] == "B"


def test_quality_v2_require_email_and_followers_moves_incomplete_creator_to_review():
    result = qualify_leads(
        [_quality_lead()],
        quality_v2_enabled=True,
        require_email=True,
        require_followers=True,
    )[0]
    assert result["quality_level"] == "C"
    assert "valid_email_required" in result["review_reasons"]
    assert "known_follower_count_required" in result["review_reasons"]


def test_quality_v2_balanced_keeps_profile_confirmation_as_dm_ready():
    lead = _quality_lead(
        identity_source_scope="post",
        product_fit_level="P3",
        product_fit_score=56,
        product_fit_requires_profile_confirmation=True,
    )
    strict = qualify_leads([lead], quality_v2_enabled=True, quality_mode="strict")[0]
    balanced = qualify_leads([lead], quality_v2_enabled=True, quality_mode="balanced")[0]
    assert strict["quality_level"] == "C"
    assert balanced["quality_level"] == "B"
    assert "product_fit_requires_profile_confirmation" in balanced["review_reasons"]


def test_paid_ugc_is_separate_from_influencer_output():
    result = qualify_leads(
        [
            _quality_lead(
                creator_type="ugc_creator",
                lead_category="paid_ugc",
                product_fit_level="P4",
                product_fit_score=38,
            )
        ],
        quality_v2_enabled=True,
        minimum_product_fit="P4",
    )[0]
    assert result["quality_level"] == "C"
    assert "paid_ugc_separate_from_influencer" in result["review_reasons"]
