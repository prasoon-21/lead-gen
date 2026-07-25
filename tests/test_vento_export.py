import asyncio

from core.services.lead_service import LeadService


class FakeVentoStore:
    def __init__(self, existing=None):
        self.existing = existing or []
        self.saved = []

    def ensure_vento_store(self):
        return {}

    def list_vento_leads(self):
        return self.existing

    def save_vento_leads(self, leads):
        self.saved.extend(leads)


def lead(handle, level="A"):
    return {
        "creator_name": handle.lstrip("@").title(),
        "instagram_handle": handle,
        "instagram_url": f"https://instagram.com/{handle.lstrip('@')}",
        "quality_level": level,
        "email": f"{handle.lstrip('@')}@example.com",
        "email_verification_status": "valid",
    }


def test_vento_export_uses_isolated_store_and_skips_review_records():
    store = FakeVentoStore()
    result = asyncio.run(LeadService(store).export_leads([lead("@one"), lead("@review", "C")], mode="vento"))
    assert result["added"] == 1
    assert result["rejected_skipped"] == 1
    assert store.saved[0]["instagram_handle"] == "@one"


def test_vento_export_deduplicates_historical_handle():
    store = FakeVentoStore(existing=[{"Creator Name": "One", "Instagram Handle": "@one"}])
    result = asyncio.run(LeadService(store).export_leads([lead("@one"), lead("@two")], mode="vento"))
    assert result["added"] == 1
    assert result["duplicates_skipped"] == 1
    assert store.saved[0]["instagram_handle"] == "@two"


def test_vento_export_skips_creator_rows_without_required_followers(monkeypatch):
    monkeypatch.setenv("VENTO_MIN_SOCIAL_FOLLOWERS", "50000")
    monkeypatch.setenv("VENTO_REQUIRE_SOCIAL_FOLLOWERS", "true")
    store = FakeVentoStore()
    unknown = {**lead("@unknown"), "category": "dog_parent_influencers"}
    below = {**lead("@below"), "category": "dog_parent_influencers", "follower_count": 49999}
    good = {**lead("@good"), "category": "dog_parent_influencers", "follower_count": 50000}
    result = asyncio.run(LeadService(store).export_leads([unknown, below, good], mode="vento"))
    assert result["added"] == 1
    assert result["rejected_skipped"] == 2
    assert store.saved[0]["instagram_handle"] == "@good"


def test_vento_export_allows_unknown_followers_when_requirement_is_relaxed(monkeypatch):
    monkeypatch.setenv("VENTO_MIN_SOCIAL_FOLLOWERS", "50000")
    monkeypatch.setenv("VENTO_REQUIRE_SOCIAL_FOLLOWERS", "false")
    store = FakeVentoStore()
    unknown = {**lead("@unknown"), "category": "dog_parent_influencers"}
    below = {**lead("@below"), "category": "dog_parent_influencers", "follower_count": 49999}
    good = {**lead("@good"), "category": "dog_parent_influencers", "follower_count": 120000}
    result = asyncio.run(LeadService(store).export_leads([unknown, below, good], mode="vento"))
    assert result["added"] == 2
    assert result["rejected_skipped"] == 1
