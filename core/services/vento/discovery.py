"""Query templates and deterministic extraction helpers for Vento Plan 3."""
from __future__ import annotations

import html
import json
import re
from html.parser import HTMLParser
from typing import Dict, List
from urllib.parse import urlparse

VENTO_SEARCH_DEPTH = "basic"
VENTO_MAX_DISCOVERY_QUERIES = 6
VENTO_MAX_ENRICHMENT_QUERIES = 50
VENTO_RESULTS_PER_DISCOVERY_QUERY = 10
VENTO_RESULTS_PER_ENRICHMENT_QUERY = 5
# Compatibility ceiling for the retained-but-unused Plan 1/2 modules.
VENTO_MAX_QUERY_LOOPS = 12

VALID_CATEGORIES = frozenset(
    {
        "dog_parent_influencers",
        "online_pet_businesses",
        "pet_shops",
        "pet_grooming",
    }
)

DISCOVERY_QUERIES: Dict[str, List[str]] = {
    "dog_parent_influencers": [
        'site:instagram.com "{location}" "dog mom"',
        'site:tiktok.com/@ "{location}" "dog mom"',
        'site:instagram.com "{location}" "dog dad"',
        'site:tiktok.com/@ "{location}" "dog dad"',
        'site:instagram.com "{location}" "pet influencer"',
        'site:tiktok.com/@ "{location}" "pet influencer"',
    ],
    "online_pet_businesses": [
        '"online pet store" "{location}"',
        '"pet supplies" "{location}" shop',
        '"pet products" "{location}" ecommerce',
        '"pet subscription box" "{location}"',
        '"dog food delivery" "{location}"',
        '"pet accessories" "{location}" online',
    ],
    "pet_shops": [
        '"pet shop" "{location}"',
        '"pet store" "{location}"',
        '"pet supply store" "{location}"',
        '"dog store" "{location}"',
        '"pet supermarket" "{location}"',
        '"local pet shop" "{location}"',
    ],
    "pet_grooming": [
        '"dog grooming" "{location}"',
        '"pet grooming" "{location}"',
        '"dog spa" "{location}"',
        '"mobile dog grooming" "{location}"',
        '"pet salon" "{location}"',
        '"dog wash" "{location}"',
    ],
}

ENRICHMENT_QUERY_TEMPLATES: Dict[str, str] = {
    "dog_parent_influencers": '"{name}" instagram tiktok followers email contact',
    "online_pet_businesses": '"{name}" "{location}" phone email website contact',
    "pet_shops": '"{name}" "{location}" phone email website contact',
    "pet_grooming": '"{name}" "{location}" phone email website contact',
}

EMAIL_RE = re.compile(r"[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?1[\s.()-]?)?\(?[2-9]\d{2}\)?[\s.-]?[2-9]\d{2}[\s.-]?\d{4}(?!\d)"
)
URL_RE = re.compile(r"https?://[^\s\"'<>]+", re.I)
IG_URL_RE = re.compile(r"(?:https?://)?(?:www\.)?instagram\.com/([A-Z0-9_.]+)", re.I)
IG_MENTION_RE = re.compile(r"(?:instagram|insta|\big\b)[\s:=-]*@([A-Z0-9_.]+)", re.I)
TT_URL_RE = re.compile(r"(?:https?://)?(?:www\.)?tiktok\.com/@([A-Z0-9_.]+)", re.I)
TT_MENTION_RE = re.compile(r"(?:tiktok|\btt\b)[\s:=-]*@([A-Z0-9_.]+)", re.I)
YT_URL_RE = re.compile(r"(?:https?://)?(?:www\.)?youtube\.com/(?:@|c/|channel/|user/)([A-Z0-9_.-]+)", re.I)

_NOISE_EMAIL_DOMAINS = frozenset(
    {"example.com", "example.org", "domain.com", "sentry.io", "wixpress.com", "cloudflare.com", "googleapis.com", "schema.org", "w3.org"}
)
_NOISE_EMAIL_LOCALS = frozenset({"user", "name", "email", "yourname", "youremail", "test", "admin@example"})
_NON_WEBSITE_DOMAINS = frozenset(
    {
        "instagram.com", "tiktok.com", "youtube.com", "youtu.be", "facebook.com",
        "twitter.com", "x.com", "linkedin.com", "pinterest.com", "reddit.com",
        "google.com", "googleusercontent.com", "bing.com", "yahoo.com", "yelp.com",
        "mapquest.com", "yellowpages.com", "bbb.org", "tripadvisor.com",
    }
)
_GENERIC_DEDUP_DOMAINS = _NON_WEBSITE_DOMAINS | frozenset(
    {"linktr.ee", "linkin.bio", "beacons.ai", "bio.link", "stan.store", "koji.to"}
)
_PLACEHOLDER_HANDLES = frozenset(
    {
        "username",
        "user_name",
        "user.name",
        "yourusername",
        "your_username",
        "your.name",
        "yourhandle",
        "your_handle",
        "myusername",
        "my_username",
        "example",
        "exampleuser",
        "testuser",
    }
)
_INVALID_IG_HANDLES = frozenset(
    {
        "p", "reel", "reels", "stories", "explore", "accounts", "about", "privacy", "directory", "tags", "tv",
        *_PLACEHOLDER_HANDLES,
    }
)
_INVALID_TT_HANDLES = frozenset(
    {"tag", "music", "video", "discover", "login", "signup", *_PLACEHOLDER_HANDLES}
)


class _ProfileMetadataParser(HTMLParser):
    """Collect the small set of public metadata that carries profile names."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.titles: List[str] = []
        self.scripts: List[str] = []
        self._in_title = False
        self._in_script = False

    def handle_starttag(self, tag: str, attrs: List[tuple[str, str | None]]) -> None:
        attributes = {str(key).lower(): str(value or "") for key, value in attrs}
        if tag.lower() == "title":
            self._in_title = True
        elif tag.lower() == "script":
            self._in_script = True
        elif tag.lower() == "meta":
            key = (attributes.get("property") or attributes.get("name") or "").lower()
            if key in {"og:title", "twitter:title", "title"} and attributes.get("content"):
                self.titles.append(attributes["content"])

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self._in_title = False
        elif tag.lower() == "script":
            self._in_script = False

    def handle_data(self, data: str) -> None:
        if self._in_title and data.strip():
            self.titles.append(data)
        elif self._in_script and data.strip():
            self.scripts.append(data)


def _decode_json_string(value: str) -> str:
    try:
        return str(json.loads(f'"{value}"'))
    except (ValueError, TypeError):
        return value


def _clean_profile_display_name(value: str, expected_handle: str) -> str:
    name = html.unescape(_decode_json_string(str(value or "")))
    name = re.sub(r"\s+", " ", name).strip(" \t\r\n|\u2022-")
    name = re.sub(r"\s*\(@[A-Z0-9_.-]+\)\s*$", "", name, flags=re.I).strip()
    name = re.sub(r"\s*(?:\||\u2022|-)\s*(?:Instagram|TikTok).*?$", "", name, flags=re.I).strip()
    if not name or len(name) > 120 or name.lower() in {"instagram", "tiktok", "log in", "sign up"}:
        return ""
    normalized_name = name.lower().lstrip("@").strip()
    normalized_handle = str(expected_handle or "").lower().lstrip("@").strip()
    if normalized_handle and normalized_name == normalized_handle:
        return ""
    return name


def extract_social_profile_name(content: str, platform: str, expected_handle: str = "") -> str:
    """Extract a public display name from Instagram/TikTok HTML or indexed metadata.

    A handle embedded in a title must match the requested profile. This prevents a
    post or recommendation for another account from overwriting the lead identity.
    """
    platform = str(platform or "").strip().lower()
    if platform not in {"instagram", "tiktok"}:
        return ""
    raw = str(content or "")
    if not raw:
        return ""
    expected = normalize_social_handle(expected_handle, platform)
    parser = _ProfileMetadataParser()
    try:
        parser.feed(raw[:1_000_000])
    except Exception:
        pass

    sources = list(dict.fromkeys(parser.titles + [raw[:20_000]]))
    title_pattern = re.compile(
        r"(?P<name>[^<>\r\n]{1,120}?)\s*\(@(?P<handle>[A-Z0-9_.-]+)\)"
        r"\s*(?:\||\u2022|-)\s*" + re.escape(platform),
        re.I,
    )
    for source in sources:
        for match in title_pattern.finditer(html.unescape(source)):
            found_handle = normalize_social_handle(match.group("handle"), platform)
            if expected and found_handle != expected:
                continue
            name = _clean_profile_display_name(match.group("name"), expected or found_handle)
            if name:
                return name

    # Public profile pages commonly expose these fields in their hydration JSON.
    # Only consider them when the expected handle is also present in the response.
    searchable = " ".join(parser.scripts) or raw[:500_000]
    if expected and expected.lower() not in searchable.lower():
        return ""
    json_fields = ("full_name", "fullName") if platform == "instagram" else ("nickname", "displayName")
    for field in json_fields:
        match = re.search(rf'"{field}"\s*:\s*"(?P<name>(?:\\.|[^"\\]){{1,240}})"', searchable, re.I)
        if match:
            name = _clean_profile_display_name(match.group("name"), expected)
            if name:
                return name
    return ""


def build_discovery_queries(category: str, location: str) -> List[str]:
    if category not in VALID_CATEGORIES:
        raise ValueError(f"Unsupported Vento category: {category}")
    return [template.format(location=location) for template in DISCOVERY_QUERIES[category]]


def build_enrichment_query(category: str, name: str, location: str) -> str:
    if category not in VALID_CATEGORIES:
        raise ValueError(f"Unsupported Vento category: {category}")
    return ENRICHMENT_QUERY_TEMPLATES[category].format(name=name, location=location)


def extract_emails(text: str) -> List[str]:
    emails: List[str] = []
    for match in EMAIL_RE.findall(html.unescape(str(text or ""))):
        email = match.lower().strip(".,;:()[]{}")
        local, domain = email.rsplit("@", 1)
        if domain in _NOISE_EMAIL_DOMAINS or local in _NOISE_EMAIL_LOCALS or email in emails:
            continue
        emails.append(email)
    return emails


def extract_phones(text: str) -> List[str]:
    phones: List[str] = []
    seen = set()
    for match in PHONE_RE.findall(html.unescape(str(text or ""))):
        digits = re.sub(r"\D", "", match)
        digits = digits[1:] if len(digits) == 11 and digits.startswith("1") else digits
        if len(digits) != 10 or digits in seen:
            continue
        seen.add(digits)
        phones.append(f"({digits[:3]}) {digits[3:6]}-{digits[6:]}")
    return phones


def _normalize_handle(value: str) -> str:
    value = str(value or "").strip().lstrip("@").rstrip("/").split("?", 1)[0].split("#", 1)[0]
    return value.lower() if re.fullmatch(r"[A-Z0-9_.]{2,100}", value, re.I) else ""


def extract_social_handles(text: str, *source_urls: str) -> Dict[str, List[str]]:
    haystack = " ".join(str(value or "") for value in (text, *source_urls))
    output: Dict[str, List[str]] = {"instagram": [], "tiktok": [], "youtube": []}
    for raw in IG_URL_RE.findall(haystack) + IG_MENTION_RE.findall(haystack):
        handle = _normalize_handle(raw)
        if handle and handle not in _INVALID_IG_HANDLES and handle not in output["instagram"]:
            output["instagram"].append(handle)
    for raw in TT_URL_RE.findall(haystack) + TT_MENTION_RE.findall(haystack):
        handle = _normalize_handle(raw)
        if handle and handle not in _INVALID_TT_HANDLES and handle not in output["tiktok"]:
            output["tiktok"].append(handle)
    for raw in YT_URL_RE.findall(haystack):
        handle = _normalize_handle(raw)
        if handle and handle not in output["youtube"]:
            output["youtube"].append(handle)
    return output


def normalize_social_handle(value: str, platform: str) -> str:
    """Compatibility helper for the old modules retained but unused by Plan 3."""
    raw = str(value or "").strip()
    if platform == "instagram":
        match = IG_URL_RE.search(raw)
        if match:
            handle = _normalize_handle(match.group(1))
            return "" if handle in _INVALID_IG_HANDLES else handle
    elif platform == "tiktok":
        match = TT_URL_RE.search(raw)
        if match:
            handle = _normalize_handle(match.group(1))
            return "" if handle in _INVALID_TT_HANDLES else handle
    elif platform == "youtube":
        match = YT_URL_RE.search(raw)
        if match:
            return _normalize_handle(match.group(1))
    handle = _normalize_handle(raw)
    if platform == "instagram" and handle in _INVALID_IG_HANDLES:
        return ""
    if platform == "tiktok" and handle in _INVALID_TT_HANDLES:
        return ""
    return handle


def looks_like_influencer_text(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(term in lowered for term in ("creator", "influencer", "dog mom", "dog dad", "vlogger", "content creator"))


def build_vento_queries(location: str, influencer_type: str = "all", max_queries: int = VENTO_MAX_QUERY_LOOPS, **_: object) -> List[str]:
    """Legacy import compatibility; the Plan 3 pipeline does not call this."""
    templates = [
        'site:instagram.com "dog mom" "{location}"',
        'site:tiktok.com "dog mom" "{location}"',
        'site:instagram.com "pet influencer" "{location}"',
        'site:tiktok.com "pet influencer" "{location}"',
        'site:instagram.com "dog trainer" "{location}"',
        'site:tiktok.com "pet vlogger" "{location}"',
    ]
    return [item.format(location=location) for item in templates[: max(1, min(max_queries, VENTO_MAX_QUERY_LOOPS))]]


def build_quality_vento_queries(location: str, influencer_type: str = "all", **kwargs: object) -> List[str]:
    templates = [
        '"dog mom" "car" "{location}" instagram',
        '"dog dad" "car" "{location}" tiktok',
        '"pet creator" "road trip" "{location}" instagram',
        '"dog parent" "product review" "{location}" tiktok',
        '"pet influencer" "car safety" "{location}" instagram',
        '"dog trainer" "travel" "{location}" social media',
        '"pet content creator" "car accessories" "{location}"',
        '"dog mom" "amazon finds" "{location}" instagram',
        '"pet vlogger" "camping" "{location}" tiktok',
        '"dog parent" "cooling" "{location}" creator',
        '"pet influencer" "vehicle" "{location}" email',
        '"dog creator" "outdoor" "{location}" instagram',
    ]
    max_queries = int(kwargs.get("max_queries") or VENTO_MAX_QUERY_LOOPS)
    return [item.format(location=location) for item in templates[: max(1, min(max_queries, VENTO_MAX_QUERY_LOOPS))]]


def normalized_domain(url: str, *, allow_generic: bool = False) -> str:
    raw = str(url or "").strip()
    if raw and not raw.lower().startswith(("http://", "https://")):
        raw = f"https://{raw}"
    host = (urlparse(raw).hostname or "").lower().removeprefix("www.")
    if not allow_generic and any(host == item or host.endswith(f".{item}") for item in _GENERIC_DEDUP_DOMAINS):
        return ""
    return host


def is_social_or_directory_url(url: str) -> bool:
    host = normalized_domain(url, allow_generic=True)
    return any(host == item or host.endswith(f".{item}") for item in _NON_WEBSITE_DOMAINS)


def extract_websites(text: str, *source_urls: str) -> List[str]:
    websites: List[str] = []
    seen = set()
    haystack = " ".join(str(value or "") for value in (text, *source_urls))
    for raw in URL_RE.findall(html.unescape(haystack)):
        url = raw.rstrip(".,;:)]}")
        domain = normalized_domain(url)
        if not domain or domain in seen:
            continue
        seen.add(domain)
        websites.append(url)
    return websites


def infer_name_from_title(title: str, url: str) -> str:
    clean = html.unescape(re.sub(r"\s+", " ", str(title or ""))).strip()
    clean = re.sub(r"\s*\([^)]*@[^)]*\)\s*", " ", clean).strip()
    for separator in (" | ", " - ", " — ", " – ", " · ", " : "):
        if separator in clean:
            candidate = clean.split(separator, 1)[0].strip()
            if 1 <= len(candidate.split()) <= 8 and len(candidate) <= 100:
                return candidate
    if 1 <= len(clean.split()) <= 8 and len(clean) <= 100:
        return clean
    host = normalized_domain(url, allow_generic=True).split(".", 1)[0]
    return host.replace("-", " ").replace("_", " ").title() if host else ""
