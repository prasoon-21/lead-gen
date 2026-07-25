import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from api.routes.vento_leads import VentoSearchRequest, router, search


def test_vento_routes_are_plan3_only():
    assert {route.path for route in router.routes} == {"/search", "/ingest"}


def test_search_request_validates_current_plan3_contract():
    request = VentoSearchRequest(
        category="dog_parent_influencers",
        location=" Seattle, WA ",
        target_count=50,
    )
    assert request.category == "dog_parent_influencers"
    assert request.location == "Seattle, WA"
    assert request.target_count == 50


def test_search_endpoint_passes_validated_request_to_pipeline():
    expected = {"leads": [], "category": "dog_parent_influencers", "location": "Seattle, WA"}
    instance = MagicMock()
    instance.run = AsyncMock(return_value=expected)
    with patch("api.routes.vento_leads.VentoLeadPipeline", return_value=instance), patch(
        "api.routes.vento_leads._save_to_sheets",
        new=AsyncMock(),
    ):
        result = asyncio.run(
            search(
                VentoSearchRequest(
                    category="dog_parent_influencers",
                    location="Seattle, WA",
                    target_count=50,
                )
            )
        )
    assert result == expected
    instance.run.assert_awaited_once_with(
        category="dog_parent_influencers",
        location="Seattle, WA",
        target_count=50,
        old_leads_csv=None,
    )
