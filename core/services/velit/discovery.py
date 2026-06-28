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
    "pennsylvania": ["Philadelphia", "Pittsburgh", "Harrisburg", "Allentown", "Lancaster", "Erie"],
    "pa": ["Philadelphia", "Pittsburgh", "Harrisburg", "Allentown", "Lancaster", "Erie"],
    "new york": ["New York", "Brooklyn", "Long Island", "Albany", "Buffalo", "Rochester"],
    "ny": ["New York", "Brooklyn", "Long Island", "Albany", "Buffalo", "Rochester"],
    "new jersey": ["Newark", "Jersey City", "Trenton", "Edison", "Paterson", "Cherry Hill"],
    "nj": ["Newark", "Jersey City", "Trenton", "Edison", "Paterson", "Cherry Hill"],
    "connecticut": ["Hartford", "New Haven", "Stamford", "Bridgeport", "Waterbury", "Norwalk"],
    "ct": ["Hartford", "New Haven", "Stamford", "Bridgeport", "Waterbury", "Norwalk"],
}

_LOCATION_ALIASES = {
    "pensyvania": "Pennsylvania",
    "pennysylvania": "Pennsylvania",
    "pennsylvannia": "Pennsylvania",
    "pennslyvania": "Pennsylvania",
    "pennsylvania": "Pennsylvania",
    "pa": "PA",
}


def normalize_velit_location(value: str) -> str:
    cleaned = " ".join((value or "").split()).strip()
    if not cleaned:
        return ""
    return _LOCATION_ALIASES.get(cleaned.lower(), cleaned)


def build_velit_queries(*, location: str, seed_query: str = "", max_queries: int = 8) -> List[str]:
    cleaned_location = normalize_velit_location(location)
    cleaned_seed = " ".join((seed_query or "").split()).strip()
    if not cleaned_location:
        return []

    base_queries = [
        f'"van upfitter" "{cleaned_location}" official website',
        f'"van conversion company" "{cleaned_location}" official website',
        f'"camper van builder" "{cleaned_location}" "contact" OR "email" OR "phone"',
        f'"sprinter van conversion" "{cleaned_location}" website',
        f'"rv upfitter" OR "rv conversion" "{cleaned_location}" website',
        f'"van builder" near "{cleaned_location}"',
    ]

    # Merge city expansion into a single OR-joined query instead of 3 per city
    expanded_locations = _STATE_CITY_EXPANSIONS.get(cleaned_location.lower(), [])
    if expanded_locations:
        cities_or = " OR ".join(f'"{city}"' for city in expanded_locations[:4])
        base_queries.append(
            f'"van upfitter" OR "van conversion" {cities_or} official website'
        )

    if cleaned_location.lower() in {"alaska", "ak"}:
        base_queries.append('"camper van conversion" "ships to Alaska" OR "serves Alaska"')
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
