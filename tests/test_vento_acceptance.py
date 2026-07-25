import asyncio

from core.services.vento.vento_lead_pipeline import VentoLeadPipeline


class VolumeCreatorSearch:
    def __init__(self, total=60):
        self.calls = 0
        self.total = total
        self.generated = 0

    async def search(self, payload):
        self.calls += 1
        if self.calls > 6:
            return {"results": []}
        results = []
        while self.generated < self.total and len(results) < 10:
            index = self.generated
            self.generated += 1
            if index % 2:
                url = f"https://www.tiktok.com/@dogcreator{index}"
            else:
                url = f"https://www.instagram.com/dogcreator{index}/"
            results.append(
                {
                    "title": f"Dog Creator {index}",
                    "url": url,
                    "content": f"Seattle WA dog mom pet creator creator{index}@example.com 60K followers",
                }
            )
        return {"results": results}


class NoNetworkCreatorPipeline(VentoLeadPipeline):
    async def _scrape_candidate_website(self, website):
        return {"emails": [], "phones": [], "socials": {"instagram": [], "tiktok": []}, "urls": [], "text": ""}

    async def _enrich_creator_name_from_social_profile(self, lead):
        return None


class DisabledApify:
    configured = False


def test_end_to_end_creator_search_returns_50_usable_50k_plus_leads():
    pipeline = NoNetworkCreatorPipeline(tavily_client=VolumeCreatorSearch(total=60), gemini_api_key="", apify_enricher=DisabledApify())
    result = asyncio.run(
        pipeline.run(
            category="dog_parent_influencers",
            location="Seattle, WA",
            target_count=50,
        )
    )

    assert len(result["leads"]) == 50
    assert result["returned_count"] == 50
    assert result["metrics"]["with_followers"] == 50
    assert all(lead["usable"] for lead in result["leads"])
    assert all(lead["follower_count"] >= 50000 for lead in result["leads"])


def test_target_ten_returns_all_seven_when_only_seven_are_usable():
    pipeline = NoNetworkCreatorPipeline(tavily_client=VolumeCreatorSearch(total=7), gemini_api_key="", apify_enricher=DisabledApify())
    result = asyncio.run(
        pipeline.run(
            category="dog_parent_influencers",
            location="Seattle, WA",
            target_count=10,
        )
    )
    assert len(result["leads"]) == 7
    assert result["returned_count"] == 7
    assert result["shortfall"] == 3
