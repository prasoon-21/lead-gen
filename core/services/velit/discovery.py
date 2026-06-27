from __future__ import annotations

import re
from typing import List
from urllib.parse import urlparse


VELIT_SEARCH_DEPTH = "basic"
VELIT_EXTRACT_DEPTH = "basic"

_NOISE_DOMAINS = (
    "linkedin.com",
    "facebook.com",
    "instagram.com",
    "youtube.com",
    "youtu.be",
    "x.com",
    "twitter.com",
    "clutch.co",
    "bbb.org",
    "chamberofcommerce.com",
    "yelp.com",
    "mapquest.com",
    "yellowpages.com",
    "angi.com",
    "thumbtack.com",
    "rvtrader.com",
    "rvbusiness.com",
    "upwork.com",
    "glassdoor.com",
    "glassdoor.co.in",
    "indeed.com",
    "zoominfo.com",
    "apollo.io",
)

_NOISE_PATH_MARKERS = (
    "/blog",
    "/blogs",
    "/news",
    "/article",
    "/articles",
    "/privacy",
    "/terms",
    "/category",
    "/tag/",
    "/author/",
    "/feed",
    ".pdf",
)

_VELIT_KEYWORDS = (
    "van upfit",
    "upfitter",
    "camper van",
    "van conversion",
    "sprinter van",
    "rv builder",
    "rv conversion",
    "mobile workspace",
    "adventure van",
    "overland van",
)


_STATE_CITY_EXPANSIONS = {
    "alaska": ["Anchorage", "Fairbanks", "Wasilla", "Palmer", "Juneau", "Kenai"],
    "ak": ["Anchorage", "Fairbanks", "Wasilla", "Palmer", "Juneau", "Kenai"],
    "new york": ["New York", "Brooklyn", "Long Island", "Albany", "Buffalo", "Rochester"],
    "ny": ["New York", "Brooklyn", "Long Island", "Albany", "Buffalo", "Rochester"],
    "new jersey": ["Newark", "Jersey City", "Trenton", "Edison", "Paterson", "Cherry Hill"],
    "nj": ["Newark", "Jersey City", "Trenton", "Edison", "Paterson", "Cherry Hill"],
    "connecticut": ["Hartford", "New Haven", "Stamford", "Bridgeport", "Waterbury", "Norwalk"],
    "ct": ["Hartford", "New Haven", "Stamford", "Bridgeport", "Waterbury", "Norwalk"],
}


def build_velit_queries(*, location: str, seed_query: str = "", max_queries: int = 16) -> List[str]:
    cleaned_location = " ".join((location or "").split()).strip()
    cleaned_seed = " ".join((seed_query or "").split()).strip()
    if not cleaned_location:
        return []

    base_queries = [
        f'"van upfitter" "{cleaned_location}" official website',
        f'"van conversion company" "{cleaned_location}" official website',
        f'"camper van builder" "{cleaned_location}" official website',
        f'"rv upfitter" "{cleaned_location}" official website',
        f'"sprinter van conversion" "{cleaned_location}" official website',
        f'"overland van builder" "{cleaned_location}" official website',
        f'"custom van interiors" "{cleaned_location}" official website',
        f'"van outfitter" "{cleaned_location}" contact',
        f'"mobile upfitter" "{cleaned_location}" website',
        f'"rv conversion shop" "{cleaned_location}" website',
        f'"van builder" near "{cleaned_location}"',
        f'"campervan conversion" near "{cleaned_location}"',
    ]
    expanded_locations = _STATE_CITY_EXPANSIONS.get(cleaned_location.lower(), [])
    for city in expanded_locations:
        base_queries.extend(
            [
                f'"van upfitter" "{city}" official website',
                f'"van conversion" "{city}" contact',
                f'"camper van builder" "{city}" website',
            ]
        )
    if cleaned_location.lower() in {"alaska", "ak"}:
        base_queries.extend(
            [
                '"camper van conversion" "ships to Alaska"',
                '"van conversion" "serves Alaska"',
                '"sprinter van conversion" "Alaska delivery"',
                '"adventure van" "Alaska" "contact"',
                '"van upfitter" "Anchorage" phone email',
                '"camper van rental" "Alaska" "conversion"',
            ]
        )
    if cleaned_seed:
        base_queries.insert(0, f'"{cleaned_seed}" "{cleaned_location}" official website')

    queries: List[str] = []
    for query in base_queries:
        normalized = " ".join(query.split()).strip()
        if normalized and normalized not in queries:
            queries.append(normalized)
        if len(queries) >= max_queries:
            break
    return queries


def is_noise_url(url: str) -> bool:
    parsed = urlparse(url or "")
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path = (parsed.path or "").lower()
    if not host:
        return True
    if any(host == domain or host.endswith(f".{domain}") for domain in _NOISE_DOMAINS):
        return True
    return any(marker in path for marker in _NOISE_PATH_MARKERS)


def normalize_company_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())


def normalize_domain(url: str) -> str:
    parsed = urlparse(url or "")
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def looks_like_velit_text(text: str) -> bool:
    lowered = (text or "").lower()
    return any(keyword in lowered for keyword in _VELIT_KEYWORDS)
