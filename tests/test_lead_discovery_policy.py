from core.services.lead_discovery_policy import EXCLUDED_LEAD_SOURCE_DOMAINS, is_excluded_lead_source_url


def test_excludes_social_forum_and_review_sources_from_lead_discovery():
    assert is_excluded_lead_source_url("https://www.reddit.com/r/dallas/comments/example")
    assert is_excluded_lead_source_url("https://quora.com/example")
    assert is_excluded_lead_source_url("https://www.yelp.com/biz/example")


def test_allows_normal_company_domains_for_lead_discovery():
    assert not is_excluded_lead_source_url("https://example-upfitters.com/contact")


def test_exclusion_list_is_broad_enough_for_hunter_credit_protection():
    assert 100 <= len(set(EXCLUDED_LEAD_SOURCE_DOMAINS)) <= 150
    for domain in ("instagram.com", "carfax.com", "hibid.com", "reddit.com", "linkedin.com"):
        assert domain in set(EXCLUDED_LEAD_SOURCE_DOMAINS)
