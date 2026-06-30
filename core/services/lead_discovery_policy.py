from __future__ import annotations

import re
from typing import Dict, Iterable, List
from urllib.parse import urlparse


DEFAULT_WEB_SEARCH_MAX_RESULTS = 10
MAX_WEB_SEARCH_MAX_RESULTS = 12
DEFAULT_TAVILY_SEARCH_DEPTH = "basic"
DEFAULT_TAVILY_EXTRACT_DEPTH = "basic"

TRUSTED_DIRECTORY_DOMAINS = (
    "clutch.co",
    "chamberofcommerce.com",
    "bbb.org",
)

EXCLUDED_LEAD_SOURCE_DOMAINS = (
    "alibaba.com",
    "aliexpress.com",
    "amazon.com",
    "ambitionbox.com",
    "angel.co",
    "angi.com",
    "apollo.io",
    "apps.apple.com",
    "blogger.com",
    "carfax.com",
    "cbinsights.com",
    "craigslist.org",
    "crunchbase.com",
    "ebay.com",
    "eventbrite.com",
    "reddit.com",
    "old.reddit.com",
    "new.reddit.com",
    "quora.com",
    "medium.com",
    "substack.com",
    "facebook.com",
    "fb.com",
    "flickr.com",
    "forbes.com",
    "foursquare.com",
    "github.com",
    "gitlab.com",
    "instagram.com",
    "justdial.com",
    "linktr.ee",
    "mapquest.com",
    "twitter.com",
    "x.com",
    "maps.apple.com",
    "maps.google.com",
    "manta.com",
    "neusourcestartup.com",
    "opencorporates.com",
    "pitchbook.com",
    "snapchat.com",
    "threads.net",
    "trustpilot.com",
    "upwork.com",
    "wikipedia.org",
    "youtube.com",
    "youtu.be",
    "tiktok.com",
    "pinterest.com",
    "linkedin.com",
    "glassdoor.com",
    "glassdoor.co.in",
    "indeed.com",
    "ziprecruiter.com",
    "careerbuilder.com",
    "dice.com",
    "greenhouse.io",
    "lever.co",
    "monster.com",
    "simplyhired.com",
    "wellfound.com",
    "yelp.com",
    "tripadvisor.com",
    "yellowpages.com",
    "bbb.org",
    "chamberofcommerce.com",
    "clutch.co",
    "goodfirms.co",
    "g2.com",
    "capterra.com",
    "softwareadvice.com",
    "zoominfo.com",
    "rocketreach.co",
    "lusha.com",
    "signalhire.com",
    "adapt.io",
    "seamless.ai",
    "theorg.com",
    "dnb.com",
    "hoovers.com",
    "datanyze.com",
    "tracxn.com",
    "tofler.in",
    "zaubacorp.com",
    "thecompanycheck.com",
    "falconebiz.com",
    "opengovus.com",
    "bizapedia.com",
    "corporationwiki.com",
    "ycombinator.com",
    "indiamart.com",
    "tradeindia.com",
    "exportersindia.com",
    "rvtrader.com",
    "rvbusiness.com",
    "hibid.com",
    "cars.com",
    "autotrader.com",
    "carsforsale.com",
    "cargurus.com",
    "carvana.com",
    "truecar.com",
    "edmunds.com",
    "kbb.com",
    "nada.org",
    "bringatrailer.com",
    "auctionzip.com",
    "proxibid.com",
    "govplanet.com",
    "ironplanet.com",
    "equipmenttrader.com",
    "commercialtrucktrader.com",
    "truckpaper.com",
    "machinerytrader.com",
    "nextdoor.com",
    "patch.com",
    "local.yahoo.com",
    "muckrack.com",
    "prnewswire.com",
    "businesswire.com",
    "globenewswire.com",
    "newswire.com",
)

_QUERY_TEMPLATES = (
    '"{industry}" companies in "{location}" -directory -blog -jobs -yelp -glassdoor -reddit -quora -forum',
    '"{industry}" "{location}" "contact us" "email" OR "phone" -reddit -quora -forum',
    '"{industry}" "{location}" official website -directory -blog -news -jobs -reddit -quora -forum',
)

_GOOD_DIRECTORY_PATH_MARKERS = (
    "/directory",
    "/directories",
    "/members",
    "/member-directory",
    "/member-list",
    "/find-a-business",
    "/find-a-member",
    "/business-directory",
    "/companies",
    "/vendors",
    "/partners",
    "/search",
)

_BAD_DIRECTORY_PATH_MARKERS = (
    "/article",
    "/articles",
    "/news",
    "/blog",
    "/press",
    "/event",
    "/events",
    "/sitemap",
    "/resource",
    "/resources",
    "/insight",
    "/insights",
    "/trust",
    "/privacy",
    "/terms",
    "/contact",
    "/about",
)


def clamp_web_search_max_results(value: int | None) -> int:
    if value is None:
        return DEFAULT_WEB_SEARCH_MAX_RESULTS
    try:
        numeric = int(value)
    except Exception:
        numeric = DEFAULT_WEB_SEARCH_MAX_RESULTS
    return max(1, min(numeric, MAX_WEB_SEARCH_MAX_RESULTS))


def _slugify_location(value: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", (value or "").lower())
    return "".join(tokens)


def _dynamic_chamber_queries(industry: str, location: str) -> List[str]:
    slug = _slugify_location(location)
    if not slug:
        return []

    candidate_sites = [
        f"{slug}.org.uk",
        f"{slug}chamber.com",
        f"{slug}chamber.org",
        f"{slug}chamber.net",
        f"member.{slug}chamber.com",
    ]
    queries: List[str] = []
    for site in candidate_sites:
        queries.append(f'site:{site} "{industry}" "{location}" "member directory"')
        queries.append(f'site:{site} "{industry}" "{location}" "business directory"')
    return queries


def build_targeted_directory_queries(
    *,
    industry: str,
    location: str,
    seed_query: str = "",
    max_queries: int = 3,
) -> List[str]:
    cleaned_industry = " ".join((industry or "").split()).strip()
    cleaned_location = " ".join((location or "").split()).strip()
    cleaned_seed = " ".join((seed_query or "").split()).strip()
    queries: List[str] = []

    def add(query: str) -> None:
        normalized = " ".join((query or "").split()).strip()
        if normalized and normalized not in queries:
            queries.append(normalized)

    if cleaned_industry and cleaned_location:
        for template in _QUERY_TEMPLATES:
            add(template.format(industry=cleaned_industry, location=cleaned_location))

    if cleaned_seed and cleaned_location:
        add(f'"{cleaned_seed}" "{cleaned_location}" official website -directory')

    return queries[:max_queries]


def is_whitelisted_directory_url(url: str) -> bool:
    parsed = urlparse(url or "")
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path = (parsed.path or "").lower()

    if not host:
        return False
    if any(marker in path for marker in _BAD_DIRECTORY_PATH_MARKERS):
        return False

    if host == "clutch.co" or host.endswith(".clutch.co"):
        return "/directory" in path or "/agencies" in path or "/it-services" in path or "/developers" in path

    if host == "chamberofcommerce.com" or host.endswith(".chamberofcommerce.com"):
        return any(marker in path for marker in _GOOD_DIRECTORY_PATH_MARKERS)

    if host == "bbb.org" or host.endswith(".bbb.org"):
        return "/us/" in path or "/search" in path or "/business-reviews" in path

    if "chamber" in host and host.endswith((".org", ".com")):
        return any(marker in path for marker in _GOOD_DIRECTORY_PATH_MARKERS)

    if host.startswith("member.") or ".member." in host:
        return any(marker in path for marker in _GOOD_DIRECTORY_PATH_MARKERS)

    return any(marker in path for marker in _GOOD_DIRECTORY_PATH_MARKERS)


def is_excluded_lead_source_url(url: str) -> bool:
    parsed = urlparse(url or "")
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if not host:
        return True
    return any(host == domain or host.endswith(f".{domain}") for domain in EXCLUDED_LEAD_SOURCE_DOMAINS)


def looks_like_bad_directory_path(url: str) -> bool:
    path = (urlparse(url or "").path or "").lower()
    return any(marker in path for marker in _BAD_DIRECTORY_PATH_MARKERS)


def filter_whitelisted_directory_results(results: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    filtered: List[Dict[str, object]] = []
    for item in results or []:
        url = str(item.get("url") or "").strip()
        if url and is_whitelisted_directory_url(url):
            filtered.append(item)
    return filtered
