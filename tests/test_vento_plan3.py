import pytest

from api.routes.vento_leads import VentoSearchRequest, router
from core.services.lead_service import LeadService
from core.services.todo_sheet_store import TodoSheetStore
from core.services.vento.discovery import DISCOVERY_QUERIES, VALID_CATEGORIES, build_discovery_queries
from core.services.vento.vento_lead_pipeline import VentoLeadPipeline


class FakeSearch:
    def __init__(self):
        self.calls = []

    async def search(self, payload):
        self.calls.append(dict(payload))
        call = len(self.calls)
        if call <= 6:
            return {
                "results": [
                    {
                        "title": f"Happy Paws {call} | Pet Shop",
                        "url": f"https://happypaws{call}.example.org",
                        "content": f"Local dog and pet supplies in Seattle, WA. (206) 555-12{call:02d}",
                    },
                    {
                        "title": f"Dog House {call} | Pet Store",
                        "url": f"https://doghouse{call}.example.net",
                        "content": f"Pet store in Seattle, WA. hello{call}@doghouse{call}.example.net",
                    },
                ]
            }
        return {"results": []}


class NoNetworkScrapePipeline(VentoLeadPipeline):
    async def _scrape_candidate_website(self, website):
        return {"emails": [], "phones": [], "socials": {"instagram": [], "tiktok": []}, "urls": [], "text": ""}


class ProfileNamePipeline(NoNetworkScrapePipeline):
    async def _fetch_social_profile_name(self, url, platform, handle):
        assert platform == "instagram"
        assert handle == "the.dog.duo"
        return "Maya & Milo"

    async def _fetch_social_profile_follower_count(self, url, platform, handle):
        assert platform == "instagram"
        assert handle == "the.dog.duo"
        return 42000


class NoSocialProfilePipeline(NoNetworkScrapePipeline):
    async def _enrich_creator_name_from_social_profile(self, lead):
        return None


class FakeApifyEnricher:
    configured = True

    def __init__(self):
        self.calls = 0
        self.last_leads = []

    async def enrich_profiles(self, leads):
        self.calls += 1
        self.last_leads = list(leads)
        enriched = []
        for lead in leads:
            item = dict(lead)
            item["follower_count"] = 65000
            item["instagram_follower_count"] = 65000
            item["follower_count_status"] = "apify_verified"
            enriched.append(item)
        return enriched, {"apify_enabled": 1, "apify_profiles_enriched": len(enriched), "apify_followers_found": len(enriched)}


class FakeV3Store:
    def __init__(self):
        self.saved = []

    def ensure_vento_store(self):
        return {}

    def list_vento_leads(self):
        return [{"Name": "Old Pet Shop", "Email": "old@example.net"}]

    def save_vento_leads(self, leads):
        self.saved.extend(leads)
        return leads


def test_plan3_has_exact_four_categories_and_six_queries_each():
    assert set(DISCOVERY_QUERIES) == set(VALID_CATEGORIES)
    for category in VALID_CATEGORIES:
        queries = build_discovery_queries(category, "Seattle, WA")
        assert len(queries) == 6
        assert all("Seattle, WA" in query for query in queries)


@pytest.mark.asyncio
async def test_two_pass_pipeline_returns_usable_business_leads_and_honors_old_csv():
    search = FakeSearch()
    pipeline = NoNetworkScrapePipeline(tavily_client=search, gemini_api_key="")
    old_csv = "Name,Email,Website\nHappy Paws 1,,https://happypaws1.example.org\n"
    result = await pipeline.run(
        category="pet_shops",
        location="Seattle, WA",
        target_count=5,
        old_leads_csv=old_csv,
    )

    assert result["returned_count"] == 5
    assert result["shortfall"] == 0
    assert result["metrics"]["old_rows"] == 1
    assert result["metrics"]["duplicates_skipped"] >= 1
    assert len(search.calls) == 18  # six discovery + twelve enrichment calls; old-history dedup follows enrichment
    assert all(call["search_depth"] == "basic" for call in search.calls)
    assert all(lead["usable"] and not lead["rejected"] for lead in result["leads"])
    assert all(lead["business_name"] for lead in result["leads"])
    assert all(lead["email"] or lead["phone"] or lead["website"] for lead in result["leads"])


def test_quality_gate_uses_category_specific_contact_rules():
    pipeline = VentoLeadPipeline(tavily_client=FakeSearch(), gemini_api_key="")
    influencer = {"creator_name": "Dog Mom", "website": "https://example.org", "evidence": "dog parent"}
    business = {"business_name": "Pet Store", "website": "https://petstore.example", "evidence": "pet shop"}
    influencer_usable, influencer_rejected = pipeline._quality_gate([influencer], "dog_parent_influencers")
    business_usable, business_rejected = pipeline._quality_gate([business], "pet_shops")
    assert not influencer_usable and influencer_rejected[0]["rejection_reason"] == "missing_contact"
    assert business_usable and not business_rejected


def test_quality_gate_rejects_placeholder_names():
    pipeline = VentoLeadPipeline(tavily_client=FakeSearch(), gemini_api_key="")
    lead = {"creator_name": "Username", "instagram_handle": "@realdog", "evidence": "dog mom pet creator"}
    usable, rejected = pipeline._quality_gate([lead], "dog_parent_influencers")
    assert not usable
    assert rejected[0]["rejection_reason"] == "placeholder_name"


def test_quality_gate_requires_50k_plus_followers_for_creators():
    pipeline = VentoLeadPipeline(tavily_client=FakeSearch(), gemini_api_key="")
    below = {
        "creator_name": "Small Dog Creator",
        "instagram_handle": "@smalldog",
        "follower_count": 49999,
        "evidence": "dog mom pet creator",
    }
    qualified = {
        "creator_name": "Big Dog Creator",
        "instagram_handle": "@bigdog",
        "follower_count": 50000,
        "evidence": "dog mom pet creator",
    }
    usable, rejected = pipeline._quality_gate([below, qualified], "dog_parent_influencers")
    assert [lead["creator_name"] for lead in usable] == ["Big Dog Creator"]
    assert rejected[0]["rejection_reason"] == "social_followers_below_50000"


def test_quality_gate_can_allow_unknown_followers_when_env_disables_requirement(monkeypatch):
    monkeypatch.setenv("VENTO_REQUIRE_SOCIAL_FOLLOWERS", "false")
    pipeline = VentoLeadPipeline(tavily_client=FakeSearch(), gemini_api_key="")
    lead = {"creator_name": "Dog Creator", "instagram_handle": "@dogcreator", "evidence": "dog mom pet creator"}
    usable, rejected = pipeline._quality_gate([lead], "dog_parent_influencers")
    assert usable and not rejected


def test_creator_with_large_second_platform_passes_with_max_follower_count():
    pipeline = VentoLeadPipeline(tavily_client=FakeSearch(), gemini_api_key="")
    lead = {
        "creator_name": "Road Pup",
        "instagram_handle": "@roadpup",
        "tiktok_handle": "@roadpup",
        "instagram_follower_count": 20000,
        "tiktok_follower_count": 80000,
        "follower_count": 80000,
        "evidence": "dog travel pet creator",
    }
    usable, rejected = pipeline._quality_gate([lead], "dog_parent_influencers")
    assert usable and not rejected


def test_follower_range_prefers_50k_to_500k_but_allows_larger_as_fallback():
    pipeline = VentoLeadPipeline(tavily_client=FakeSearch(), gemini_api_key="")
    target_range = {"creator_name": "Target Creator", "instagram_handle": "@target", "follower_count": 120000}
    above_range = {"creator_name": "Mega Creator", "instagram_handle": "@mega", "follower_count": 900000}
    unknown = {"creator_name": "Unknown Creator", "instagram_handle": "@unknown"}
    assert pipeline._follower_sort_bucket(target_range, "dog_parent_influencers") == 0
    assert pipeline._follower_sort_bucket(above_range, "dog_parent_influencers") == 2
    assert pipeline._follower_sort_bucket(unknown, "dog_parent_influencers") == 3
    usable, rejected = pipeline._quality_gate([target_range, above_range, unknown], "dog_parent_influencers")
    assert len(usable) == 3
    assert not rejected


def test_strict_max_follower_range_can_reject_above_500k(monkeypatch):
    monkeypatch.setenv("VENTO_ALLOW_ABOVE_MAX_SOCIAL_FOLLOWERS", "false")
    pipeline = VentoLeadPipeline(tavily_client=FakeSearch(), gemini_api_key="")
    lead = {"creator_name": "Mega Creator", "instagram_handle": "@mega", "follower_count": 900000}
    usable, rejected = pipeline._quality_gate([lead], "dog_parent_influencers")
    assert not usable
    assert rejected[0]["rejection_reason"] == "social_followers_above_500000"


@pytest.mark.asyncio
async def test_apify_enrichment_runs_only_for_influencers():
    apify = FakeApifyEnricher()
    pipeline = VentoLeadPipeline(tavily_client=FakeSearch(), gemini_api_key="", apify_enricher=apify)
    leads = [{"creator_name": "Dog Creator", "instagram_handle": "@dogcreator"}]
    enriched = await pipeline._enrich_with_apify_if_enabled(leads, "dog_parent_influencers")
    skipped = await pipeline._enrich_with_apify_if_enabled(leads, "pet_shops")
    assert enriched[0]["follower_count"] == 65000
    assert "follower_count" not in skipped[0]
    assert apify.calls == 1
    assert pipeline._stats["apify_profiles_enriched"] == 1


@pytest.mark.asyncio
async def test_apify_enrichment_shortlists_only_best_prefiltered_candidates(monkeypatch):
    monkeypatch.setenv("VENTO_APIFY_CANDIDATE_POOL", "2")
    apify = FakeApifyEnricher()
    pipeline = VentoLeadPipeline(tavily_client=FakeSearch(), gemini_api_key="", apify_enricher=apify)
    leads = [
        {
            "creator_name": "Best Creator",
            "instagram_handle": "@best",
            "tiktok_handle": "@best",
            "email": "best@example.com",
            "follower_count": 80000,
            "evidence": "dog mom pet creator Seattle",
            "location": "Seattle, WA",
        },
        {
            "creator_name": "Good Creator",
            "instagram_handle": "@good",
            "follower_count": 120000,
            "evidence": "dog pet creator",
        },
        {
            "creator_name": "Below Creator",
            "instagram_handle": "@below",
            "follower_count": 12000,
            "evidence": "dog pet creator",
        },
        {
            "creator_name": "No Social",
            "evidence": "dog pet creator",
        },
    ]
    enriched = await pipeline._enrich_with_apify_if_enabled(leads, "dog_parent_influencers")
    assert apify.calls == 1
    assert [lead["creator_name"] for lead in apify.last_leads] == ["Best Creator", "Good Creator"]
    assert pipeline._stats["apify_prequalified_candidates"] == 2
    assert pipeline._stats["apify_candidates_skipped_prefilter"] == 2
    assert enriched[0]["follower_count"] == 65000
    assert enriched[1]["follower_count"] == 65000
    assert enriched[2]["follower_count"] == 12000
    assert "follower_count" not in enriched[3]


def test_follower_count_parser_reads_snippet_formats():
    pipeline = VentoLeadPipeline(tavily_client=FakeSearch(), gemini_api_key="")
    assert pipeline._extract_follower_count("Dog creator with 31.5K followers on Instagram") == 31500
    assert pipeline._extract_follower_count("Followers: 1.2M") == 1200000


def test_creator_email_filter_removes_media_emails_and_prefers_creator_contacts():
    pipeline = VentoLeadPipeline(tavily_client=FakeSearch(), gemini_api_key="")
    emails = pipeline._creator_emails(
        ["orange@patch.com", "news@cbs.com", "hello@dogcreator.com", "realdog@gmail.com"],
        "Real Dog creator. For collabs email realdog@gmail.com. Publisher tips: news@cbs.com",
        name="Real Dog",
        socials={"instagram": ["realdog"]},
    )
    assert "orange@patch.com" not in emails
    assert "news@cbs.com" not in emails
    assert emails[0] == "realdog@gmail.com"


@pytest.mark.asyncio
async def test_enrichment_replaces_bad_publisher_email_with_creator_email():
    class CreatorEmailSearch:
        async def search(self, payload):
            return {
                "results": [
                    {
                        "title": "Real Dog (@realdog) | Instagram",
                        "url": "https://www.instagram.com/realdog/",
                        "content": "Dog parent pet creator. Business contact realdog@gmail.com. 85K followers.",
                    }
                ]
            }

    pipeline = NoSocialProfilePipeline(tavily_client=CreatorEmailSearch(), gemini_api_key="")
    lead = await pipeline._enrich_single_candidate(
        {
            "creator_name": "Real Dog",
            "instagram_handle": "@realdog",
            "email": "news@patch.com",
            "evidence": "dog parent pet creator",
        },
        "dog_parent_influencers",
        "Seattle, WA",
    )
    assert lead["email"] == "realdog@gmail.com"


@pytest.mark.asyncio
async def test_social_profile_display_name_replaces_handle_fallback(monkeypatch):
    monkeypatch.setenv("VENTO_SOCIAL_NAME_FETCH_ENABLED", "true")
    pipeline = ProfileNamePipeline(tavily_client=FakeSearch(), gemini_api_key="")
    lead = {
        "creator_name": "The Dog Duo",
        "creator_name_source": "handle_fallback",
        "instagram_handle": "@the.dog.duo",
        "instagram_url": "https://www.instagram.com/the.dog.duo/",
    }
    await pipeline._enrich_creator_name_from_social_profile(lead)
    assert lead["creator_name"] == "Maya & Milo"
    assert lead["creator_name_source"] == "instagram_profile_metadata"
    assert lead["creator_name_confidence"] == "high"
    assert lead["follower_count"] == 42000
    assert lead["follower_count_status"] == "instagram_profile_metadata"


def test_enrichment_evidence_is_bound_to_candidate_domain():
    pipeline = VentoLeadPipeline(tavily_client=FakeSearch(), gemini_api_key="")
    candidate = {"business_name": "Dirty Dog Austin", "website": "https://dirtydog.example/contact"}
    results = [
        {"title": "Eastside Pet Ranch", "url": "https://eastside.example", "content": "east@example.com"},
        {"title": "Dirty Dog Contact", "url": "https://dirtydog.example/about", "content": "hello@dirtydog.example"},
    ]
    matched = pipeline._identity_matched_results(candidate, results)
    assert [item["url"] for item in matched] == ["https://dirtydog.example/about"]


def test_api_and_sheet_schema_are_plan3_only():
    assert {route.path for route in router.routes} == {"/search", "/ingest"}
    request = VentoSearchRequest(category="pet_grooming", location="Austin, TX", target_count=200)
    assert request.target_count == 200
    assert TodoSheetStore.VENTO_LEAD_HEADERS == [
        "Name", "Category", "Email", "Phone", "Website", "Instagram", "TikTok",
        "Location", "Niche", "Relevance Score", "Source", "Status", "Notes", "Follower Count",
    ]


@pytest.mark.asyncio
async def test_v3_sheet_export_skips_old_duplicates_without_abc_filtering():
    store = FakeV3Store()
    leads = [
        {"business_name": "Old Pet Shop", "email": "new@example.net", "website": "https://oldpet.example", "usable": True},
        {"business_name": "New Groomer", "phone": "(512) 555-0111", "website": "https://newgroomer.example", "usable": True},
    ]
    result = await LeadService(store).export_leads(leads, mode="vento")
    assert result["added"] == 1
    assert result["duplicates_skipped"] == 1
    assert store.saved[0]["business_name"] == "New Groomer"
