from api.auth import is_valid_api_key


def test_is_valid_api_key_allows_requests_when_no_key_configured():
    assert is_valid_api_key(configured_key="", header_value=None, authorization=None) is True


def test_is_valid_api_key_rejects_missing_key_when_required():
    assert is_valid_api_key(configured_key="", header_value=None, authorization=None, require_key=True) is False


def test_is_valid_api_key_accepts_configured_header_value():
    assert is_valid_api_key(configured_key="secret", header_value="secret", authorization=None) is True


def test_is_valid_api_key_accepts_bearer_token():
    assert is_valid_api_key(configured_key="secret", header_value=None, authorization="Bearer secret") is True


def test_is_valid_api_key_rejects_missing_or_wrong_key_when_configured():
    assert is_valid_api_key(configured_key="secret", header_value=None, authorization=None) is False
    assert is_valid_api_key(configured_key="secret", header_value="wrong", authorization=None) is False
