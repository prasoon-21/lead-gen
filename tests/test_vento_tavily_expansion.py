import asyncio

from core.services.vento.vento_lead_pipeline import VentoLeadPipeline


class VolumeTavily:
    def __init__(self):
        self.calls = 0

    async def search(self, payload):
        self.calls += 1
        is_tiktok = "tiktok.com" in payload["query"]
        results = []
        for index in range(10):
            unique = self.calls * 100 + index
            if is_tiktok:
                url = f"https://www.tiktok.com/@dogcreator{unique}"
            else:
                url = f"https://www.instagram.com/dogcreator{unique}/"
            results.append(
                {
                    "url": url,
                    "title": f"Dog Creator {unique}",
                    "content": "Dog mom pet creator based in Seattle WA 75K followers",
                }
            )
        return {"results": results}


class NoNetworkPipeline(VentoLeadPipeline):
    async def _scrape_candidate_website(self, website):
        return {"emails": [], "phones": [], "socials": {"instagram": [], "tiktok": []}, "urls": [], "text": ""}

    async def _enrich_creator_name_from_social_profile(self, lead):
        return None


class DisabledApify:
    configured = False


def test_discovery_collects_more_than_twenty_and_reports_platforms():
    tavily = VolumeTavily()
    pipeline = NoNetworkPipeline(tavily_client=tavily, gemini_api_key="", apify_enricher=DisabledApify())
    leads = asyncio.run(
        pipeline._discover_candidates(
            category="dog_parent_influencers",
            location="Seattle",
            target_count=50,
        )
    )
    assert len(leads) == 120
    assert len(leads) > 20
    assert pipeline._stats["discovery_expansion_queries"] == 6
    assert sum(bool(lead.get("instagram_handle")) for lead in leads) > 0
    assert sum(bool(lead.get("tiktok_handle")) for lead in leads) > 0


def test_target_ten_can_return_ten_indexed_social_dm_leads():
    pipeline = NoNetworkPipeline(tavily_client=VolumeTavily(), gemini_api_key="", apify_enricher=DisabledApify())
    result = asyncio.run(
        pipeline.run(
            category="dog_parent_influencers",
            location="Seattle, WA",
            target_count=10,
        )
    )

    assert len(result["leads"]) == 10
    assert all(lead["usable"] for lead in result["leads"])
    assert all(lead["follower_count"] >= 50000 for lead in result["leads"])
