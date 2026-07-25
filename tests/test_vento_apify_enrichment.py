import asyncio
import json

import httpx

from core.services.vento.apify_enrichment import ApifyCreatorEnricher


def run(coro):
    return asyncio.run(coro)


def test_apify_enricher_is_inert_without_token(monkeypatch, tmp_path):
    monkeypatch.delenv("APIFY_API_TOKEN", raising=False)
    enricher = ApifyCreatorEnricher(cache_path=tmp_path / "cache.json")
    assert enricher.configured is False
    leads, stats = run(enricher.enrich_profiles([{"creator_name": "Doggo", "instagram_handle": "@doggo"}]))
    assert leads[0]["instagram_handle"] == "@doggo"
    assert stats["apify_enabled"] == 0


def test_instagram_actor_response_maps_followers_email_and_cache(tmp_path):
    calls = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        payload = json.loads(request.content.decode())
        assert payload["directUrls"] == ["https://www.instagram.com/doggo/"]
        return httpx.Response(
            200,
            json=[
                {
                    "username": "doggo",
                    "fullName": "Doggo Pup",
                    "biography": "Dog mom contact doggo@gmail.com",
                    "followersCount": 75000,
                    "businessEmail": "doggo@gmail.com",
                    "externalUrl": "doggo.example",
                    "url": "https://www.instagram.com/doggo/",
                }
            ],
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        enricher = ApifyCreatorEnricher(token="secret", client=client, cache_path=tmp_path / "cache.json")
        leads, stats = run(enricher.enrich_profiles([{"creator_name": "@doggo", "instagram_handle": "@doggo"}]))
        assert leads[0]["creator_name"] == "Doggo Pup"
        assert leads[0]["email"] == "doggo@gmail.com"
        assert leads[0]["instagram_follower_count"] == 75000
        assert leads[0]["follower_count"] == 75000
        assert leads[0]["follower_count_status"] == "apify_verified"
        assert stats["apify_profile_requests"] == 1

        second, second_stats = run(enricher.enrich_profiles([{"creator_name": "@doggo", "instagram_handle": "@doggo"}]))
        assert second[0]["follower_count"] == 75000
        assert second_stats["apify_cache_hits"] == 1
        assert len(calls) == 1
    finally:
        run(client.aclose())


def test_tiktok_actor_response_maps_author_meta_fans(tmp_path):
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode())
        assert payload["profiles"] == ["roadpup"]
        return httpx.Response(
            200,
            json=[
                {
                    "authorMeta": {
                        "name": "roadpup",
                        "nickName": "Road Pup Adventures",
                        "signature": "Dog travel creator",
                        "fans": 88000,
                        "verified": True,
                    },
                    "profileUrl": "https://www.tiktok.com/@roadpup",
                }
            ],
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        enricher = ApifyCreatorEnricher(token="secret", client=client, cache_path=tmp_path / "cache.json")
        leads, stats = run(enricher.enrich_profiles([{"creator_name": "@roadpup", "tiktok_handle": "@roadpup"}]))
    finally:
        run(client.aclose())
    assert leads[0]["creator_name"] == "Road Pup Adventures"
    assert leads[0]["tiktok_follower_count"] == 88000
    assert leads[0]["follower_count"] == 88000
    assert stats["apify_followers_found"] == 1


def test_bad_media_email_is_ignored(tmp_path):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[{"username": "dognews", "fullName": "Dog News", "followersCount": 90000, "email": "news@example.com"}],
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        enricher = ApifyCreatorEnricher(token="secret", client=client, cache_path=tmp_path / "cache.json")
        leads, _ = run(enricher.enrich_profiles([{"creator_name": "@dognews", "instagram_handle": "@dognews"}]))
    finally:
        run(client.aclose())
    assert leads[0].get("email", "") == ""
    assert leads[0]["follower_count"] == 90000


def test_apify_creator_email_replaces_existing_publisher_email(tmp_path):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[
                {
                    "username": "doggo",
                    "fullName": "Doggo Pup",
                    "followersCount": 90000,
                    "businessEmail": "doggo@gmail.com",
                    "url": "https://www.instagram.com/doggo/",
                }
            ],
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        enricher = ApifyCreatorEnricher(token="secret", client=client, cache_path=tmp_path / "cache.json")
        leads, stats = run(
            enricher.enrich_profiles(
                [{"creator_name": "Doggo Pup", "instagram_handle": "@doggo", "email": "news@cbs.com"}]
            )
        )
    finally:
        run(client.aclose())
    assert leads[0]["email"] == "doggo@gmail.com"
    assert leads[0]["email_source_type"] == "apify_profile"
    assert stats["apify_emails_found"] == 1


def test_instagram_tiktok_mismatch_drops_wrong_second_platform(tmp_path):
    async def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "instagram-scraper" in url:
            return httpx.Response(
                200,
                json=[{"username": "lilothehusky", "fullName": "Lilo The Husky", "followersCount": 26000}],
            )
        return httpx.Response(
            200,
            json=[{"authorMeta": {"name": "sfgate", "nickName": "SFGATE", "fans": 900000}}],
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        enricher = ApifyCreatorEnricher(token="secret", client=client, cache_path=tmp_path / "cache.json")
        leads, stats = run(
            enricher.enrich_profiles(
                [
                    {
                        "creator_name": "Lilo The Husky",
                        "instagram_handle": "@lilothehusky",
                        "tiktok_handle": "@sfgate",
                    }
                ]
            )
        )
    finally:
        run(client.aclose())
    assert leads[0]["instagram_handle"] == "@lilothehusky"
    assert leads[0].get("tiktok_handle", "") == ""
    assert leads[0]["follower_count"] == 26000
    assert stats["apify_platform_mismatches_dropped"] == 1


def test_actor_429_does_not_crash_pipeline(tmp_path):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "rate limited"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        enricher = ApifyCreatorEnricher(token="secret", client=client, cache_path=tmp_path / "cache.json")
        leads, stats = run(enricher.enrich_profiles([{"creator_name": "Doggo", "instagram_handle": "@doggo"}]))
    finally:
        run(client.aclose())
    assert leads[0]["instagram_handle"] == "@doggo"
    assert "follower_count" not in leads[0]
    assert stats["apify_failures"] == 1
