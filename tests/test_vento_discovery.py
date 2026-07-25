from core.services.vento.discovery import (
    VENTO_MAX_QUERY_LOOPS,
    build_vento_queries,
    extract_social_profile_name,
    extract_social_handles,
    looks_like_influencer_text,
    normalize_social_handle,
)


def test_queries_are_bounded_and_location_specific():
    queries = build_vento_queries(location="Seattle", influencer_type="all")
    assert 1 <= len(queries) <= VENTO_MAX_QUERY_LOOPS
    assert all('"' in query for query in queries)
    assert any("Seattle" in query for query in queries)


def test_query_pool_deliberately_covers_instagram_and_tiktok():
    queries = build_vento_queries(location="Indiana, US", influencer_type="pet", max_queries=12)
    instagram = [query for query in queries if "instagram.com" in query]
    tiktok = [query for query in queries if "tiktok.com" in query]
    assert len(instagram) >= 2
    assert len(tiktok) >= 2
    assert "instagram.com" in queries[0]
    assert "tiktok.com" in queries[1]


def test_extracts_handles_from_direct_result_urls():
    handles = extract_social_handles(
        "Creator profile",
        "https://www.instagram.com/Jane.Dog/",
        "https://www.tiktok.com/@JaneDog",
        "https://www.youtube.com/@JaneDogTV",
    )
    assert handles["instagram"] == ["jane.dog"]
    assert handles["tiktok"] == ["janedog"]
    assert handles["youtube"] == ["janedogtv"]


def test_rejects_non_profile_instagram_paths():
    assert normalize_social_handle("https://instagram.com/reel/abc", "instagram") == ""
    assert normalize_social_handle("@Real_Creator", "instagram") == "real_creator"


def test_rejects_placeholder_social_handles():
    assert normalize_social_handle("@username", "instagram") == ""
    assert normalize_social_handle("https://www.tiktok.com/@your_handle", "tiktok") == ""


def test_influencer_keyword_detection():
    assert looks_like_influencer_text("Seattle dog mom and pet content creator")
    assert not looks_like_influencer_text("general company directory result")


def test_extracts_instagram_display_name_from_profile_metadata():
    html = '<meta property="og:title" content="Jane &amp; Scout (@jane.scout) • Instagram photos and videos">'
    assert extract_social_profile_name(html, "instagram", "@jane.scout") == "Jane & Scout"
    assert extract_social_profile_name(html, "instagram", "@different.creator") == ""


def test_extracts_tiktok_display_name_from_hydration_json():
    html = '<script>{"uniqueId":"roadpup","nickname":"Road Pup Adventures"}</script>'
    assert extract_social_profile_name(html, "tiktok", "@roadpup") == "Road Pup Adventures"


def test_profile_json_name_can_contain_an_apostrophe():
    html = '<script>{"uniqueId":"dogs.life","nickname":"A Dog\\u0027s Life"}</script>'
    assert extract_social_profile_name(html, "tiktok", "@dogs.life") == "A Dog's Life"
