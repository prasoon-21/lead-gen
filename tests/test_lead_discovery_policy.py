from core.services.lead_discovery_policy import is_excluded_lead_source_url


def test_excludes_social_forum_and_review_sources_from_lead_discovery():
    assert is_excluded_lead_source_url("https://www.reddit.com/r/dallas/comments/example")
    assert is_excluded_lead_source_url("https://quora.com/example")
    assert is_excluded_lead_source_url("https://www.yelp.com/biz/example")


def test_allows_normal_company_domains_for_lead_discovery():
    assert not is_excluded_lead_source_url("https://example-upfitters.com/contact")
