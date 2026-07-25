import asyncio

import httpx

from core.services.vento.providers import VentoSocialProvider


def test_provider_is_inert_without_url_or_key(monkeypatch):
    monkeypatch.delenv("SOCIAL_DATA_API_URL", raising=False)
    monkeypatch.delenv("SOCIAL_DATA_API_KEY", raising=False)
    provider = VentoSocialProvider()
    assert provider.configured is False
    result = asyncio.run(provider.fetch(location="Seattle", influencer_type="pet", follower_min=5000, follower_max=20000, limit=10))
    assert result.status == "not_configured"
    assert result.records == []


def test_generic_provider_maps_results_and_uses_environment_auth(monkeypatch):
    monkeypatch.setenv("VENTO_SOCIAL_PROVIDER", "generic")
    monkeypatch.setenv("SOCIAL_DATA_API_URL", "https://provider.example/search")
    monkeypatch.setenv("SOCIAL_DATA_API_KEY", "secret")

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer secret"
        return httpx.Response(200, json={"results": [{"name": "Jane", "instagram": "@jane"}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        provider = VentoSocialProvider(client=client)
        result = asyncio.run(provider.fetch(location="Seattle", influencer_type="pet", follower_min=5000, follower_max=20000, limit=10))
    finally:
        asyncio.run(client.aclose())
    assert result.status == "completed"
    assert result.requests == 1
    assert result.records[0]["name"] == "Jane"
