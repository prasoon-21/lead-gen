from __future__ import annotations

import html
import logging
import re
from typing import Any, Dict, Iterable, List
from urllib.parse import urljoin, urlparse

import httpx

from core.tools.base import ToolExecutionError
from core.tools.external.tavily_client import TavilyClient
from core.utils.phone_quality import extract_phone_candidates, normalize_phone_candidate, rank_phone_candidates


CONTACT_PATHS = (
    "",
    "/contact",
    "/contact-us",
    "/about",
    "/about-us",
    "/locations",
    "/location",
    "/support",
    "/sales",
)
CONTACT_LINK_HINTS = ("contact", "about", "location", "support", "sales", "dealer", "visit")
MAX_PAGES_PER_DOMAIN = 7
MAX_BODY_CHARS = 1_250_000


class PhoneCompanyMatcher:
    """Match a phone number against official phones scraped from the company's website."""

    def __init__(self, *, use_tavily_fallback: bool = True):
        self.use_tavily_fallback = use_tavily_fallback
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._locks: Dict[str, Any] = {}
        self._logger = logging.getLogger("agent.runtime")

    async def verify(
        self,
        phone: Any,
        website: Any,
        *,
        region: str = "US",
    ) -> Dict[str, Any]:
        phone_candidate = normalize_phone_candidate(phone, source_type="existing", region=region)
        if not phone_candidate:
            return {
                "status": "no_valid_phone_to_match",
                "confidence": "low",
                "source_url": "",
                "matched_phone": "",
                "scraped_phones": [],
                "method": "none",
                "message": "Phone could not be normalized, so company website matching was skipped.",
            }

        normalized_site = self._normalize_website(website)
        if not normalized_site:
            return {
                "status": "no_website",
                "confidence": "low",
                "source_url": "",
                "matched_phone": "",
                "scraped_phones": [],
                "method": "none",
                "message": "No company website URL was available in this row.",
            }

        domain_key = self._domain_key(normalized_site)
        evidence = await self._get_domain_evidence(normalized_site, domain_key, region=region)
        candidates = evidence.get("phone_candidates") or []
        scraped_phones = [item.get("display") for item in candidates if item.get("display")]

        if evidence.get("status") == "scrape_failed":
            return {
                "status": "website_scrape_failed",
                "confidence": "low",
                "source_url": "",
                "matched_phone": "",
                "scraped_phones": scraped_phones[:5],
                "method": evidence.get("method", "normal_scrape"),
                "message": evidence.get("message") or "Could not scrape the company website.",
            }

        match = self._find_match(phone_candidate, candidates)
        if match:
            method = str(match.get("source") or evidence.get("method") or "normal_scrape")
            confidence = "high" if method != "tavily_extract" else "medium"
            return {
                "status": "matched_on_company_website",
                "confidence": confidence,
                "source_url": match.get("source_url", ""),
                "matched_phone": match.get("display", ""),
                "scraped_phones": scraped_phones[:5],
                "method": method,
                "message": "Phone number was found on the official company website/contact pages.",
            }

        if candidates:
            return {
                "status": "different_phone_found_on_website",
                "confidence": "medium",
                "source_url": candidates[0].get("source_url", ""),
                "matched_phone": "",
                "scraped_phones": scraped_phones[:5],
                "method": evidence.get("method", "normal_scrape"),
                "message": "Website has phone numbers, but they did not match the uploaded phone.",
            }

        return {
            "status": "not_found_on_website",
            "confidence": "low",
            "source_url": "",
            "matched_phone": "",
            "scraped_phones": [],
            "method": evidence.get("method", "normal_scrape"),
            "message": "No phone number was found on the scraped company website pages.",
        }

    async def _get_domain_evidence(self, website: str, domain_key: str, *, region: str) -> Dict[str, Any]:
        if domain_key in self._cache:
            return self._cache[domain_key]

        # Lightweight per-domain lock avoids spending duplicate Tavily credits when the same site appears repeatedly.
        import asyncio

        lock = self._locks.setdefault(domain_key, asyncio.Lock())
        async with lock:
            if domain_key in self._cache:
                return self._cache[domain_key]
            evidence = await self._collect_domain_evidence(website, region=region)
            self._cache[domain_key] = evidence
            return evidence

    async def _collect_domain_evidence(self, website: str, *, region: str) -> Dict[str, Any]:
        normal = await self._normal_scrape(website, region=region)
        if normal.get("phone_candidates"):
            return normal
        if not self.use_tavily_fallback:
            return normal

        tavily = await self._tavily_extract(website, region=region)
        if tavily.get("phone_candidates"):
            return tavily
        if normal.get("status") == "scrape_failed" and tavily.get("status") == "scrape_failed":
            return {
                "status": "scrape_failed",
                "method": "normal_scrape+tavily_extract",
                "phone_candidates": [],
                "message": f"{normal.get('message', '')} {tavily.get('message', '')}".strip(),
            }
        return tavily if tavily.get("status") != "not_found" else normal

    async def _normal_scrape(self, website: str, *, region: str) -> Dict[str, Any]:
        urls = self._candidate_urls(website)
        candidates: List[Dict[str, Any]] = []
        fetched = 0
        errors: List[str] = []
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        try:
            async with httpx.AsyncClient(timeout=12, follow_redirects=True, headers=headers) as client:
                for index, url in enumerate(urls):
                    if index >= MAX_PAGES_PER_DOMAIN:
                        break
                    try:
                        response = await client.get(url)
                        if response.status_code >= 400:
                            errors.append(f"{url} HTTP {response.status_code}")
                            continue
                        body = (response.text or "")[:MAX_BODY_CHARS]
                        fetched += 1
                        candidates.extend(
                            extract_phone_candidates(
                                self._html_to_searchable_text(body),
                                source_type=self._source_type_for_url(url),
                                source_url=str(response.url),
                                region=region,
                            )
                        )
                        if index == 0:
                            urls.extend(self._discover_contact_links(str(response.url), body, urls))
                    except Exception as exc:
                        errors.append(f"{url} {type(exc).__name__}")
        except Exception as exc:
            return {
                "status": "scrape_failed",
                "method": "normal_scrape",
                "phone_candidates": [],
                "message": f"Normal website scrape failed: {type(exc).__name__}: {exc}",
            }

        ranked = rank_phone_candidates(candidates)
        if ranked:
            return {"status": "phones_found", "method": "normal_scrape", "phone_candidates": ranked}
        if fetched:
            return {
                "status": "not_found",
                "method": "normal_scrape",
                "phone_candidates": [],
                "message": "Website pages loaded, but no phone numbers were found.",
            }
        return {
            "status": "scrape_failed",
            "method": "normal_scrape",
            "phone_candidates": [],
            "message": "; ".join(errors[:3]) or "No website pages could be loaded.",
        }

    async def _tavily_extract(self, website: str, *, region: str) -> Dict[str, Any]:
        urls = self._candidate_urls(website)[:5]
        try:
            data = await TavilyClient().extract({"urls": urls, "include_images": False})
        except ToolExecutionError as exc:
            return {
                "status": "scrape_failed",
                "method": "tavily_extract",
                "phone_candidates": [],
                "message": f"Tavily fallback failed: {exc}",
            }
        except Exception as exc:
            return {
                "status": "scrape_failed",
                "method": "tavily_extract",
                "phone_candidates": [],
                "message": f"Tavily fallback failed: {type(exc).__name__}: {exc}",
            }

        candidates: List[Dict[str, Any]] = []
        for item in data.get("results") or []:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or website)
            text = "\n".join(
                str(item.get(key) or "")
                for key in ("raw_content", "content", "text", "description")
                if item.get(key)
            )
            candidates.extend(
                extract_phone_candidates(text, source_type="tavily_extract", source_url=url, region=region)
            )
        ranked = rank_phone_candidates(candidates)
        if ranked:
            return {"status": "phones_found", "method": "tavily_extract", "phone_candidates": ranked}
        return {
            "status": "not_found",
            "method": "tavily_extract",
            "phone_candidates": [],
            "message": "Tavily fallback did not find phone numbers on the company website.",
        }

    @staticmethod
    def _normalize_website(value: Any) -> str:
        text = str(value or "").strip()
        if not text or text.lower() in {"-", "--", "n/a", "na", "none", "null", "not found"}:
            return ""
        if "://" not in text:
            text = f"https://{text}"
        parsed = urlparse(text)
        if not parsed.netloc:
            return ""
        return f"{parsed.scheme or 'https'}://{parsed.netloc}{parsed.path}".rstrip("/")

    @staticmethod
    def _domain_key(url: str) -> str:
        parsed = urlparse(url)
        return parsed.netloc.lower().removeprefix("www.")

    def _candidate_urls(self, website: str) -> List[str]:
        parsed = urlparse(website)
        base = f"{parsed.scheme}://{parsed.netloc}"
        urls: List[str] = []
        for path in CONTACT_PATHS:
            candidate = urljoin(base + "/", path.lstrip("/"))
            if candidate not in urls:
                urls.append(candidate)
        if website not in urls:
            urls.insert(0, website)
        return urls[:MAX_PAGES_PER_DOMAIN]

    def _discover_contact_links(self, base_url: str, body: str, existing_urls: Iterable[str]) -> List[str]:
        existing = set(existing_urls)
        links: List[str] = []
        base_domain = self._domain_key(base_url)
        for match in re.finditer(r"href=[\"']([^\"'#]+)[\"']", body or "", flags=re.IGNORECASE):
            href = html.unescape(match.group(1)).strip()
            if not href or href.startswith(("mailto:", "tel:", "javascript:")):
                continue
            absolute = urljoin(base_url, href).split("#", 1)[0].rstrip("/")
            if absolute in existing or absolute in links:
                continue
            parsed = urlparse(absolute)
            if self._domain_key(absolute) != base_domain:
                continue
            lowered = f"{parsed.path} {href}".lower()
            if any(hint in lowered for hint in CONTACT_LINK_HINTS):
                links.append(absolute)
            if len(links) >= 4:
                break
        return links

    @staticmethod
    def _source_type_for_url(url: str) -> str:
        path = urlparse(url).path.lower()
        if "contact" in path:
            return "contact_page"
        if "support" in path:
            return "support_page"
        if "sales" in path:
            return "sales_page"
        if "about" in path:
            return "about_page"
        if "team" in path:
            return "team_page"
        return "homepage"

    @staticmethod
    def _html_to_searchable_text(body: str) -> str:
        body = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", " ", body or "", flags=re.IGNORECASE | re.DOTALL)
        body = re.sub(r"<br\s*/?>", "\n", body, flags=re.IGNORECASE)
        body = re.sub(r"</(p|div|li|tr|td|section|footer|header)>", "\n", body, flags=re.IGNORECASE)
        body = re.sub(r"<[^>]+>", " ", body)
        return html.unescape(re.sub(r"\s+", " ", body))

    @staticmethod
    def _find_match(phone_candidate: Dict[str, Any], candidates: Iterable[Dict[str, Any]]) -> Dict[str, Any] | None:
        target_digits = str(phone_candidate.get("digits") or "")
        target_last10 = target_digits[-10:]
        for candidate in candidates or []:
            candidate_digits = str(candidate.get("digits") or "")
            if not candidate_digits:
                continue
            if candidate_digits == target_digits:
                return candidate
            if target_last10 and candidate_digits[-10:] == target_last10:
                return candidate
        return None
