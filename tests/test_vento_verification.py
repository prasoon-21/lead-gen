import asyncio

from core.services.vento.verification import (
    VentoVerifier,
    parse_follower_count,
    verify_creator_location,
)


def test_follower_parser_handles_decimal_suffixes():
    assert parse_follower_count("5,000") == 5000
    assert parse_follower_count("12.5K") == 12500
    assert parse_follower_count("100K") == 100000
    assert parse_follower_count("1.2M") == 1200000


def test_requested_location_is_not_used_as_creator_evidence():
    verified = verify_creator_location(
        {"location": "", "raw_content": "Pet creator and product reviewer", "source": "tavily_fallback"},
        target="Seattle, WA",
    )
    assert verified["location_confidence"] == "unknown"
    assert verified["location"] == ""


def test_explicit_bulk_location_is_confirmed():
    verified = verify_creator_location(
        {"location": "Seattle, WA", "source": "bulk_ingest"},
        target="Seattle, WA",
    )
    assert verified["location_confidence"] == "confirmed"
    assert verified["location_city"] == "Seattle"
    assert verified["location_region"] == "PNW"


def test_verifier_preserves_per_field_statuses():
    async def profile_fetcher(url):
        return 200, "active public creator profile"

    def email_verifier(email):
        return {"status": "valid", "message": "Mailbox accepted"}

    verifier = VentoVerifier(profile_fetcher=profile_fetcher, email_verifier=email_verifier)
    result = asyncio.run(
        verifier.verify_leads(
            [
                {
                    "creator_name": "Jane",
                    "email": "jane@example.com",
                    "instagram_url": "https://instagram.com/jane",
                    "tiktok_url": "",
                    "youtube_url": "",
                    "location": "Portland, OR",
                    "source": "bulk_ingest",
                    "niche": "dog mom pet creator",
                    "follower_count": "20K",
                }
            ],
            target_location="Portland, OR",
        )
    )[0]
    assert result["instagram_profile_status"] == "verified_active"
    assert result["email_verification_status"] == "valid"
    assert result["location_confidence"] == "confirmed"
    assert result["profile_verified"] is True
