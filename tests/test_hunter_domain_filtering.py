from core.services._test_features.hunter.hunter_search_pipeline import HunterSearchPipeline


def test_hunter_blocks_marketplace_and_social_domains_before_credit_use():
    assert HunterSearchPipeline._is_excluded_domain("instagram.com")
    assert HunterSearchPipeline._is_excluded_domain("www.carfax.com")
    assert HunterSearchPipeline._is_excluded_domain("hibid.com")


def test_hunter_velit_intent_and_candidate_detection():
    assert HunterSearchPipeline._is_velit_intent(
        industry="Van Upfitters / RV Builders",
        seed_query="velit camping vans",
    )
    assert HunterSearchPipeline._looks_like_velit_candidate(
        domain="northtexasupfitters.com",
        title="Commercial van upfitter in Dallas",
        content="Fleet upfit, van shelving, truck body equipment",
        url="https://northtexasupfitters.com",
    )
    assert not HunterSearchPipeline._looks_like_velit_candidate(
        domain="instagram.com",
        title="Instagram",
        content="Social media photos and reels",
        url="https://instagram.com",
    )
