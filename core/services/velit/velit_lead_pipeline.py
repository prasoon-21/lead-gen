from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urljoin, urlparse

import httpx

from core.services.lead_quality_service import score_lead
from core.services.production_lead_pipeline import (
    BROWSER_VERIFICATION_HEADERS,
    ProductionLeadPipeline,
)
from core.services.velit.discovery import (
    VELIT_EXTRACT_DEPTH,
    VELIT_SEARCH_DEPTH,
    build_velit_queries,
    is_noise_url,
    looks_like_velit_text,
    normalize_company_key,
    normalize_domain,
)
from core.utils.lead_summary import build_plain_lead_summary
from core.utils.phone_quality import (
    build_phone_summary,
    extract_phone_candidates,
    phone_candidates_from_values,
)


class VelitLeadPipeline(ProductionLeadPipeline):
    """Velit-specific lead pipeline using Tavily search, Gemini scoring, and direct site scraping."""

    CONTACT_PATHS = (
        "",
        "/contact",
        "/about",
        "/team",
        "/services",
        "/support",
        "/careers",
        "/contact-us",
        "/our-team",
        "/sales",
        "/locations",
        "/builds",
        "/gallery",
        "/faq",
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.gemini_max_calls_per_run = max(
            self.gemini_max_calls_per_run,
            self._bounded_int_env("VELIT_GEMINI_MAX_CALLS_PER_RUN", default=8, minimum=2, maximum=20),
        )
        self.linkedin_enrichment_limit = max(
            self.linkedin_enrichment_limit,
            self._bounded_int_env("VELIT_LINKEDIN_ENRICHMENT_LIMIT", default=20, minimum=5, maximum=60),
        )
        self.official_site_resolution_limit = max(
            self.official_site_resolution_limit,
            self._bounded_int_env("VELIT_OFFICIAL_SITE_RESOLUTION_LIMIT", default=40, minimum=10, maximum=100),
        )
        self.max_search_results_per_query = self._bounded_int_env(
            "VELIT_SEARCH_RESULTS_PER_QUERY",
            default=8,
            minimum=4,
            maximum=12,
        )
        self.max_contact_scrapes = self._bounded_int_env(
            "VELIT_CONTACT_SCRAPE_LIMIT",
            default=40,
            minimum=8,
            maximum=80,
        )
        self.contact_search_limit = self._bounded_int_env(
            "VELIT_CONTACT_SEARCH_LIMIT",
            default=60,
            minimum=10,
            maximum=100,
        )
        self.max_extra_site_links = self._bounded_int_env(
            "VELIT_EXTRA_SITE_LINKS",
            default=8,
            minimum=2,
            maximum=20,
        )
        self.playwright_scrape_limit = self._bounded_int_env(
            "VELIT_PLAYWRIGHT_SCRAPE_LIMIT",
            default=8,
            minimum=0,
            maximum=30,
        )

    async def run(
        self,
        *,
        industry: str,
        location: str,
        seed_query: str = "",
        target_count: int = 15,
    ) -> Dict[str, Any]:
        target_count = max(1, min(int(target_count or 15), 100))
        phase_steps: List[Dict[str, Any]] = []
        try:
            search_results = await self._discover_targets(industry=industry, location=location, seed_query=seed_query)
            phase_steps.append(
                {
                    "step": 1,
                    "stage": "velit_direct_discovery",
                    "type": "tavily_search",
                    "search_depth": VELIT_SEARCH_DEPTH,
                    "query_count": len(build_velit_queries(location=location, seed_query=seed_query)),
                    "result_count": len(search_results),
                }
            )

            corporate_targets = await self._resolve_corporate_targets(
                search_results,
                industry=industry,
                location=location,
                limit=max(20, target_count * 2),
            )
            phase_steps.append(
                {
                    "step": 2,
                    "stage": "velit_site_resolution",
                    "type": "direct_site_resolution",
                    "input_count": len(search_results),
                    "result_count": len(corporate_targets),
                }
            )

            verified = await self._verify_targets(corporate_targets, limit=max(20, target_count * 2))
            phase_steps.append(
                {
                    "step": 3,
                    "stage": "velit_site_verification",
                    "type": "async_http_verification",
                    "input_count": len(corporate_targets),
                    "result_count": len(verified),
                }
            )

            extracted = await self._extract_targets(verified)
            leads = await self._score_extracted_targets(
                extracted,
                industry=industry or "Upfitter",
                location=location,
                target_count=target_count,
            )
            leads = await self._enrich_leads_with_linkedin(
                leads,
                industry=industry or "Upfitter",
                location=location,
            )
            leads = await self._recover_missing_contacts(
                leads,
                industry=industry or "Upfitter",
                location=location,
            )
            phase_steps.append(
                {
                    "step": 4,
                    "stage": "velit_enrichment_scoring",
                    "type": "site_scraping_gemini_linkedin",
                    "extract_depth": VELIT_EXTRACT_DEPTH,
                    "input_count": len(verified),
                    "lead_count": len(leads),
                    "model": self.gemini_model,
                }
            )

            return {
                "leads": leads,
                "steps": phase_steps,
                "metadata": {
                    "pipeline": "velit_specialized_pipeline",
                    "search_depth": VELIT_SEARCH_DEPTH,
                    "extract_depth": VELIT_EXTRACT_DEPTH,
                    "discovered_targets": len(search_results),
                    "corporate_targets": len(corporate_targets),
                    "verified_targets": len(verified),
                    "scored_targets": len(leads),
                    "linkedin_enriched_targets": sum(
                        1 for lead in leads if lead.get("contact_person_name") or lead.get("founder_name")
                    ),
                    "email_backed_targets": sum(1 for lead in leads if lead.get("contact_email")),
                    "contact_scrape_limit": self.max_contact_scrapes,
                    "contact_search_limit": self.contact_search_limit,
                    "gemini_max_calls_per_run": self.gemini_max_calls_per_run,
                },
            }
        except Exception as exc:
            self.logger.error("velit_pipeline_run_failed", extra={"payload": str(exc), "stacktrace": True})
            return {
                "leads": [],
                "steps": phase_steps,
                "error": f"Internal Server Error during Velit lead generation: {exc}",
                "metadata": {
                    "pipeline": "velit_specialized_pipeline",
                    "status": "failed",
                    "error_type": type(exc).__name__,
                },
            }

    async def _discover_targets(self, *, industry: str, location: str, seed_query: str) -> List[Dict[str, Any]]:
        queries = build_velit_queries(location=location, seed_query=seed_query, max_queries=12)
        seen_urls = set()
        results: List[Dict[str, Any]] = []
        for query in queries:
            try:
                data = await self.tavily.search(
                    {
                        "query": query,
                        "search_depth": VELIT_SEARCH_DEPTH,
                        "max_results": self.max_search_results_per_query,
                        "include_raw_content": False,
                    }
                )
            except Exception as exc:
                self.logger.warning("velit_tavily_search_failed", extra={"payload": str(exc)})
                continue
            for item in data.get("results") or []:
                url = str(item.get("url") or "").strip()
                title = str(item.get("title") or "").strip()
                content = str(item.get("content") or "").strip()
                if not url or url in seen_urls:
                    continue
                if is_noise_url(url) and not looks_like_velit_text(f"{title} {content}"):
                    continue
                seen_urls.add(url)
                results.append(
                    {
                        "url": url,
                        "title": title,
                        "content": content,
                        "query": query,
                    }
                )
        return results

    async def _resolve_corporate_targets(
        self,
        discovery_items: Iterable[Dict[str, Any]],
        *,
        industry: str,
        location: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        targets: List[Dict[str, Any]] = []
        seen_companies = set()
        seen_domains = set()
        items = list(discovery_items)

        for item in items:
            url = str(item.get("url") or "").strip()
            title = str(item.get("title") or "").strip()
            content = str(item.get("content") or "").strip()
            company_name = self._resolve_company_name(
                str(item.get("company_name") or "").strip(),
                title=title,
                url=url,
            )
            company_name = self._clean_company_candidate(company_name)
            if not self._looks_like_valid_company_name(company_name):
                continue

            company_key = normalize_company_key(company_name)
            domain = normalize_domain(url)
            if company_key in seen_companies or (domain and domain in seen_domains):
                continue

            official = None
            if not is_noise_url(url) and self._looks_like_operating_company_result(
                url=url,
                title=title,
                content=content,
                company_name=company_name,
                industry=industry or "Upfitter",
                location=location,
            ):
                official = {
                    "url": self._homepage_url(url),
                    "title": title or company_name,
                    "content": content,
                }
            else:
                official = await self._find_velit_company_site(company_name=company_name, location=location)

            if not official:
                continue
            official_url = str(official.get("url") or "").strip()
            official_domain = normalize_domain(official_url)
            if not official_url or (official_domain and official_domain in seen_domains):
                continue

            seen_companies.add(company_key)
            if official_domain:
                seen_domains.add(official_domain)
            targets.append(
                {
                    "url": official_url,
                    "title": str(official.get("title") or company_name).strip(),
                    "content": str(official.get("content") or content).strip(),
                    "company_name": company_name,
                    "directory_url": url,
                }
            )
            if len(targets) >= max(10, min(limit, 100)):
                break
        return targets

    async def _find_velit_company_site(self, *, company_name: str, location: str) -> Optional[Dict[str, Any]]:
        queries = [
            f'"{company_name}" "{location}" official website',
            f'"{company_name}" van conversion website',
            f'"{company_name}" rv upfitter website',
        ]
        best_item: Optional[Dict[str, Any]] = None
        best_score = -1
        for query in queries:
            try:
                data = await asyncio.wait_for(
                    self.tavily.search(
                        {
                            "query": query,
                            "search_depth": VELIT_SEARCH_DEPTH,
                            "max_results": 6,
                            "include_raw_content": False,
                        }
                    ),
                    timeout=self.official_site_timeout_seconds,
                )
            except Exception:
                continue
            for item in data.get("results") or []:
                url = str(item.get("url") or "").strip()
                if not url or is_noise_url(url):
                    continue
                title = str(item.get("title") or "").strip()
                content = str(item.get("content") or "").strip()
                if self._looks_like_operating_company_result(
                    url=url,
                    title=title,
                    content=content,
                    company_name=company_name,
                    industry="Upfitter",
                    location=location,
                ):
                    item = dict(item)
                    item["url"] = self._homepage_url(url)
                    return item
                score = self._official_site_candidate_score(
                    url=url,
                    title=title,
                    content=content,
                    company_name=company_name,
                    industry="Upfitter",
                    location=location,
                )
                if score > best_score:
                    best_item = dict(item)
                    best_item["url"] = self._homepage_url(url)
                    best_score = score
        return best_item if best_score >= 2 else None

    async def _extract_targets(self, targets: List[Any]) -> List[Dict[str, Any]]:
        extracted = await super()._extract_targets(targets)
        if not extracted:
            return []

        enriched_results = await asyncio.gather(
            *(
                self._augment_item_with_site_scrape(index, item)
                for index, item in enumerate(extracted)
            ),
            return_exceptions=True,
        )

        merged: List[Dict[str, Any]] = []
        for index, result in enumerate(enriched_results):
            if isinstance(result, Exception):
                merged.append(extracted[index])
            else:
                merged.append(result)
        return merged

    async def _augment_item_with_site_scrape(self, index: int, item: Dict[str, Any]) -> Dict[str, Any]:
        enriched = dict(item)
        if index >= self.max_contact_scrapes:
            return enriched
        scrape = await self._scrape_site_contact_signals(
            str(item.get("url") or ""),
            allow_playwright=index < self.playwright_scrape_limit,
        )
        if not scrape:
            return enriched

        signal_lines: List[str] = []
        for label, values in (
            ("emails", scrape.get("emails") or []),
            ("phones", scrape.get("phones") or []),
            ("linkedin", scrape.get("linkedin") or []),
            ("instagram", scrape.get("instagram") or []),
            ("facebook", scrape.get("facebook") or []),
            ("youtube", scrape.get("youtube") or []),
        ):
            if values:
                signal_lines.append(f"{label}: {' | '.join(values[:4])}")
        if signal_lines:
            enriched["content"] = f"{enriched.get('content') or ''}\n\nContact signals:\n" + "\n".join(signal_lines)
        enriched["contact_signals"] = scrape
        return enriched

    async def _scrape_site_contact_signals(self, base_url: str, *, allow_playwright: bool = True) -> Dict[str, List[str]]:
        if not base_url:
            return {}
        timeout = httpx.Timeout(connect=4.0, read=6.0, write=4.0, pool=4.0)
        results: Dict[str, List[str]] = {
            "emails": [],
            "phones": [],
            "phone_candidates": [],
            "linkedin": [],
            "instagram": [],
            "facebook": [],
            "youtube": [],
        }
        seen_pages = set()
        base_home = self._homepage_url(base_url)
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers=BROWSER_VERIFICATION_HEADERS,
        ) as client:
            extra_links: List[str] = []
            for path in self.CONTACT_PATHS:
                page_url = urljoin(base_home.rstrip("/") + "/", path.lstrip("/"))
                if page_url in seen_pages:
                    continue
                seen_pages.add(page_url)
                try:
                    response = await client.get(page_url)
                except Exception:
                    continue
                if response.status_code >= 400:
                    continue
                html = response.text or ""
                source_type = self._phone_source_type_for_path(path)
                self._collect_signal_values(results, html, source_type=source_type, source_url=page_url)
                if path in {"", "/contact", "/about"}:
                    extra_links.extend(self._extract_relevant_internal_links(base_home, html))
            for page_url in extra_links[: self.max_extra_site_links]:
                if page_url in seen_pages:
                    continue
                seen_pages.add(page_url)
                try:
                    response = await client.get(page_url)
                except Exception:
                    continue
                if response.status_code >= 400:
                    continue
                source_type = self._phone_source_type_for_path(urlparse(page_url).path)
                self._collect_signal_values(results, response.text or "", source_type=source_type, source_url=page_url)
        if allow_playwright and self._needs_rendered_scrape(results):
            rendered_results = await self._scrape_site_contact_signals_with_playwright(base_home)
            self._merge_signal_results(results, rendered_results)
        return {
            key: values[:12] if key == "phone_candidates" else values[:5]
            for key, values in results.items()
            if values
        }

    @staticmethod
    def _needs_rendered_scrape(results: Dict[str, List[Any]]) -> bool:
        signal_count = sum(len(results.get(key) or []) for key in ("emails", "phones", "linkedin", "instagram", "facebook", "youtube"))
        return signal_count < 2

    @classmethod
    def _merge_signal_results(cls, target: Dict[str, List[Any]], source: Dict[str, List[Any]]) -> None:
        for key, values in (source or {}).items():
            bucket = target.setdefault(key, [])
            if key == "phone_candidates":
                seen = {
                    str(item.get("e164") or item.get("display"))
                    for item in bucket
                    if isinstance(item, dict)
                }
                for item in values or []:
                    if not isinstance(item, dict):
                        continue
                    item_key = str(item.get("e164") or item.get("display"))
                    if item_key and item_key not in seen:
                        seen.add(item_key)
                        bucket.append(item)
                continue
            for value in values or []:
                if value and value not in bucket:
                    bucket.append(value)

    async def _scrape_site_contact_signals_with_playwright(self, base_home: str) -> Dict[str, List[Any]]:
        try:
            from playwright.async_api import async_playwright
        except Exception as exc:
            self.logger.info("velit_playwright_not_available", extra={"payload": str(exc)})
            return {}

        results: Dict[str, List[Any]] = {
            "emails": [],
            "phones": [],
            "phone_candidates": [],
            "linkedin": [],
            "instagram": [],
            "facebook": [],
            "youtube": [],
        }
        pages = [
            base_home,
            urljoin(base_home.rstrip("/") + "/", "contact"),
            urljoin(base_home.rstrip("/") + "/", "about"),
        ]
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page(
                    user_agent=BROWSER_VERIFICATION_HEADERS.get("User-Agent"),
                    viewport={"width": 1366, "height": 900},
                )
                for page_url in pages:
                    try:
                        await page.goto(page_url, wait_until="networkidle", timeout=12000)
                        html = await page.content()
                    except Exception:
                        continue
                    source_type = self._phone_source_type_for_path(urlparse(page_url).path)
                    self._collect_signal_values(results, html, source_type=source_type, source_url=page_url)
                await browser.close()
        except Exception as exc:
            self.logger.info("velit_playwright_scrape_failed", extra={"payload": str(exc)})
            return {}
        return results

    @classmethod
    def _collect_signal_values(
        cls,
        results: Dict[str, List[Any]],
        html: str,
        *,
        source_type: str = "text",
        source_url: str = "",
    ) -> None:
        text = html or ""
        emails = cls._valid_email_matches(text)
        phone_candidates = extract_phone_candidates(text, source_type=source_type, source_url=source_url)
        phones = [candidate.get("display", "") for candidate in phone_candidates if candidate.get("display")]
        linkedin = list(dict.fromkeys(re.findall(r"https?://(?:www\.)?linkedin\.com/[^\s\"'<>]+", text, re.IGNORECASE)))
        instagram = list(dict.fromkeys(re.findall(r"https?://(?:www\.)?instagram\.com/[^\s\"'<>]+", text, re.IGNORECASE)))
        facebook = list(dict.fromkeys(re.findall(r"https?://(?:www\.)?facebook\.com/[^\s\"'<>]+", text, re.IGNORECASE)))
        youtube = list(dict.fromkeys(re.findall(r"https?://(?:www\.)?(?:youtube\.com|youtu\.be)/[^\s\"'<>]+", text, re.IGNORECASE)))

        for value in emails:
            if value not in results["emails"]:
                results["emails"].append(value)
        for value in phones:
            if value not in results["phones"]:
                results["phones"].append(value)
        seen_candidate_keys = {
            str(candidate.get("e164") or candidate.get("display"))
            for candidate in results.get("phone_candidates", [])
            if isinstance(candidate, dict)
        }
        for candidate in phone_candidates:
            key = str(candidate.get("e164") or candidate.get("display"))
            if key and key not in seen_candidate_keys:
                seen_candidate_keys.add(key)
                results["phone_candidates"].append(candidate)
        for value in linkedin:
            if value not in results["linkedin"]:
                results["linkedin"].append(value)
        for value in instagram:
            if value not in results["instagram"]:
                results["instagram"].append(value)
        for value in facebook:
            if value not in results["facebook"]:
                results["facebook"].append(value)
        for value in youtube:
            if value not in results["youtube"]:
                results["youtube"].append(value)

    @staticmethod
    def _extract_relevant_internal_links(base_home: str, html: str) -> List[str]:
        matches = re.findall(r'href=["\']([^"\']+)["\']', html or "", re.IGNORECASE)
        collected: List[str] = []
        seen = set()
        for href in matches:
            absolute = urljoin(base_home, href.strip())
            parsed = urlparse(absolute)
            if not parsed.scheme.startswith("http"):
                continue
            path = (parsed.path or "").lower()
            if not any(marker in path for marker in ("contact", "about", "team", "service", "support", "career", "build", "gallery", "social")):
                continue
            cleaned = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            if cleaned not in seen:
                seen.add(cleaned)
                collected.append(cleaned)
        return collected

    @staticmethod
    def _phone_source_type_for_path(path: str) -> str:
        lowered = (path or "").lower()
        if "contact" in lowered:
            return "contact_page"
        if "support" in lowered:
            return "support_page"
        if "sales" in lowered:
            return "sales_page"
        if "team" in lowered:
            return "team_page"
        if "about" in lowered:
            return "about_page"
        if not lowered or lowered == "/":
            return "homepage"
        return "text"

    async def _score_one(
        self,
        item: Dict[str, Any],
        *,
        industry: str,
        location: str,
        use_gemini: bool = True,
    ) -> Optional[Dict[str, Any]]:
        lead = await super()._score_one(
            item,
            industry=industry or "Upfitter",
            location=location,
            use_gemini=use_gemini,
        )
        if not lead:
            return None
        signals = item.get("contact_signals") if isinstance(item.get("contact_signals"), dict) else {}
        if signals:
            lead["contact_email"] = self._first_non_empty(lead.get("contact_email"), *(signals.get("emails") or []))
            scrape_phone_summary = build_phone_summary(
                signals.get("phone_candidates")
                or phone_candidates_from_values(
                    signals.get("phones") or [],
                    source_type="contact_page",
                    source_url=lead.get("company_website") or "",
                )
            )
            self._apply_phone_summary(lead, scrape_phone_summary)
            lead["company_linkedin_url"] = self._first_non_empty(*(signals.get("linkedin") or []))
            lead["instagram_url"] = self._first_non_empty(*(signals.get("instagram") or []))
            lead["facebook_url"] = self._first_non_empty(*(signals.get("facebook") or []))
            lead["youtube_url"] = self._first_non_empty(*(signals.get("youtube") or []))
            if lead.get("company_linkedin_url") and not lead.get("linkedin_url"):
                lead["linkedin_url"] = lead["company_linkedin_url"]
            lead.setdefault("source_details", []).append(
                {
                    "stage": "velit_contact_scrape",
                    "type": "site_contact_signals",
                    "provider": "httpx_scrape",
                    "url": lead.get("company_website"),
                    "value": json.dumps(signals)[:500],
                }
            )
        quality = score_lead(lead)
        lead.update({key: value for key, value in quality.items() if key != "quality_score"})
        lead["quality_score"] = max(int(lead.get("quality_score") or 0), int(quality.get("quality_score") or 0))
        lead["quality_score"] = max(0, min(int(lead["quality_score"]), 100))
        lead["lead_summary"] = build_plain_lead_summary(lead)
        return lead

    async def _recover_missing_contacts(
        self,
        leads: List[Dict[str, Any]],
        *,
        industry: str,
        location: str,
    ) -> List[Dict[str, Any]]:
        if not leads:
            return leads

        async def recover_one(index: int, lead: Dict[str, Any]) -> Dict[str, Any]:
            item = dict(lead)
            if index >= self.contact_search_limit:
                return item
            if item.get("contact_email") and item.get("contact_phone") and (item.get("linkedin_url") or item.get("company_linkedin_url")):
                return item
            company_name = str(item.get("company_name") or "").strip()
            website = str(item.get("company_website") or "").strip()
            domain = normalize_domain(website)
            if not company_name:
                return item

            recovery = await self._search_company_contact_signals(
                company_name=company_name,
                domain=domain,
                location=location,
                person_name=self._first_non_empty(item.get("contact_person_name"), item.get("founder_name")),
            )
            if not recovery:
                return item

            item["contact_email"] = self._pick_best_email(
                existing=item.get("contact_email"),
                candidates=recovery.get("emails") or [],
                website_domain=domain,
            )
            recovery_phone_summary = build_phone_summary(
                recovery.get("phone_candidates")
                or phone_candidates_from_values(
                    recovery.get("phones") or [],
                    source_type="tavily_contact_search",
                    source_url=item.get("company_website") or "",
                )
            )
            self._apply_phone_summary(item, recovery_phone_summary)
            item["linkedin_url"] = self._first_non_empty(item.get("linkedin_url"), *(recovery.get("person_linkedin") or []))
            item["company_linkedin_url"] = self._first_non_empty(item.get("company_linkedin_url"), *(recovery.get("company_linkedin") or []))
            item["instagram_url"] = self._first_non_empty(item.get("instagram_url"), *(recovery.get("instagram") or []))
            item["facebook_url"] = self._first_non_empty(item.get("facebook_url"), *(recovery.get("facebook") or []))
            item["youtube_url"] = self._first_non_empty(item.get("youtube_url"), *(recovery.get("youtube") or []))
            item.setdefault("source_details", []).append(
                {
                    "stage": "velit_contact_recovery",
                    "type": "tavily_company_contact_search",
                    "provider": "tavily_search",
                    "url": item.get("company_website"),
                    "value": json.dumps(recovery)[:500],
                }
            )
            quality = score_lead(item)
            item.update({key: value for key, value in quality.items() if key != "quality_score"})
            item["quality_score"] = max(int(item.get("quality_score") or 0), int(quality.get("quality_score") or 0))
            item["quality_score"] = max(0, min(int(item["quality_score"]), 100))
            item["lead_summary"] = build_plain_lead_summary(item)
            return item

        recovered = await asyncio.gather(
            *(recover_one(index, lead) for index, lead in enumerate(leads)),
            return_exceptions=True,
        )
        normalized = [
            dict(leads[index]) if isinstance(item, Exception) else item
            for index, item in enumerate(recovered)
        ]
        normalized.sort(key=self._velit_rank_key, reverse=True)
        return normalized

    async def _search_company_contact_signals(
        self,
        *,
        company_name: str,
        domain: str,
        location: str,
        person_name: str = "",
    ) -> Dict[str, List[str]]:
        queries = [
            f'"{company_name}" "{location}" email',
            f'"{company_name}" "{location}" phone',
            f'site:{domain} contact' if domain else "",
            f'site:{domain} team' if domain else "",
            f'site:{domain} linkedin' if domain else "",
            f'site:{domain} instagram OR youtube OR facebook' if domain else "",
            f'"{person_name}" "{company_name}" email' if person_name else "",
            f'"{company_name}" founder owner ceo linkedin',
        ]
        signals: Dict[str, List[str]] = {
            "emails": [],
            "phones": [],
            "phone_candidates": [],
            "person_linkedin": [],
            "company_linkedin": [],
            "instagram": [],
            "facebook": [],
            "youtube": [],
        }
        seen_queries = set()
        for raw_query in queries:
            query = " ".join((raw_query or "").split()).strip()
            if not query or query in seen_queries:
                continue
            seen_queries.add(query)
            try:
                data = await self.tavily.search(
                    {
                        "query": query,
                        "search_depth": VELIT_SEARCH_DEPTH,
                        "max_results": 5,
                        "include_raw_content": False,
                    }
                )
            except Exception:
                continue
            for result in data.get("results") or []:
                url = str(result.get("url") or "").strip()
                blob = " ".join(
                    [
                        str(result.get("title") or ""),
                        str(result.get("content") or ""),
                        url,
                    ]
                )
                self._merge_contact_blob_into_signals(signals, blob, source_url=url)
                if "linkedin.com/in/" in url and url not in signals["person_linkedin"]:
                    signals["person_linkedin"].append(url)
                if "linkedin.com/company/" in url and url not in signals["company_linkedin"]:
                    signals["company_linkedin"].append(url)
                if "instagram.com/" in url and url not in signals["instagram"]:
                    signals["instagram"].append(url)
                if "facebook.com/" in url and url not in signals["facebook"]:
                    signals["facebook"].append(url)
                if ("youtube.com/" in url or "youtu.be/" in url) and url not in signals["youtube"]:
                    signals["youtube"].append(url)
        return {
            key: values[:12] if key == "phone_candidates" else values[:5]
            for key, values in signals.items()
            if values
        }

    @classmethod
    def _merge_contact_blob_into_signals(cls, signals: Dict[str, List[Any]], blob: str, *, source_url: str = "") -> None:
        emails = cls._valid_email_matches(blob or "")
        phone_candidates = extract_phone_candidates(blob or "", source_type="tavily_contact_search", source_url=source_url)
        phones = [candidate.get("display", "") for candidate in phone_candidates if candidate.get("display")]
        for value in emails:
            if value not in signals["emails"]:
                signals["emails"].append(value)
        for value in phones:
            if value not in signals["phones"]:
                signals["phones"].append(value)
        seen_candidate_keys = {
            str(candidate.get("e164") or candidate.get("display"))
            for candidate in signals.get("phone_candidates", [])
            if isinstance(candidate, dict)
        }
        for candidate in phone_candidates:
            key = str(candidate.get("e164") or candidate.get("display"))
            if key and key not in seen_candidate_keys:
                seen_candidate_keys.add(key)
                signals["phone_candidates"].append(candidate)

    @classmethod
    def _pick_best_email(cls, *, existing: Any, candidates: List[str], website_domain: str) -> str:
        current = cls._first_non_empty(existing)
        pool = [current] if current else []
        for candidate in candidates:
            cleaned = cls._first_non_empty(candidate)
            if cleaned and cleaned not in pool:
                pool.append(cleaned)
        if not pool:
            return ""

        def email_score(value: str) -> tuple[int, int]:
            local = value.split("@", 1)[0].lower().strip() if "@" in value else ""
            domain = value.split("@", 1)[1].lower().strip() if "@" in value else ""
            generic = local in {"info", "support", "contact", "hello", "sales", "admin", "office", "team", "help"}
            domain_match = 1 if website_domain and website_domain in domain else 0
            return (domain_match, 0 if generic else 1)

        return sorted(pool, key=email_score, reverse=True)[0]

    @staticmethod
    def _velit_rank_key(item: Dict[str, Any]) -> tuple[int, int, int, int, int, int]:
        return (
            1 if item.get("contact_email") else 0,
            1 if item.get("contact_phone") else 0,
            1 if item.get("contact_person_name") or item.get("founder_name") else 0,
            1 if item.get("linkedin_url") or item.get("company_linkedin_url") else 0,
            1 if item.get("instagram_url") or item.get("facebook_url") or item.get("youtube_url") else 0,
            int(item.get("quality_score") or 0),
        )

    @classmethod
    def _looks_like_operating_company_result(
        cls,
        *,
        url: str,
        title: str,
        content: str,
        company_name: str,
        industry: str,
        location: str,
    ) -> bool:
        if is_noise_url(url):
            return False
        combined = f"{title} {content} {url}".lower()
        if any(marker in combined for marker in ("privacy policy", "terms of service", "careers at indeed", "job opening")):
            return False
        company_tokens = [token for token in re.findall(r"[a-z0-9]+", company_name.lower()) if len(token) > 2]
        if company_tokens and not any(token in combined for token in company_tokens[:2]):
            return False
        velit_markers = cls._industry_markers(industry)
        return any(marker in combined for marker in velit_markers) or looks_like_velit_text(combined)

    @staticmethod
    def _industry_markers(industry: str) -> List[str]:
        return [
            "upfitter",
            "van conversion",
            "camper van",
            "sprinter van",
            "rv builder",
            "rv conversion",
            "mobile upfit",
            "custom van",
            "adventure van",
            "overland van",
        ] + [token for token in re.findall(r"[a-z0-9]+", (industry or "").lower()) if len(token) > 2]

    @classmethod
    def _official_site_candidate_score(
        cls,
        *,
        url: str,
        title: str,
        content: str,
        company_name: str,
        industry: str,
        location: str,
    ) -> int:
        score = super()._official_site_candidate_score(
            url=url,
            title=title,
            content=content,
            company_name=company_name,
            industry=industry,
            location=location,
        )
        combined = f"{title} {content} {url}".lower()
        if looks_like_velit_text(combined):
            score += 2
        if any(marker in combined for marker in ("contact", "about", "services", "gallery", "build", "conversion")):
            score += 1
        return score

    @staticmethod
    def _homepage_url(url: str) -> str:
        parsed = urlparse(url or "")
        if not parsed.scheme or not parsed.netloc:
            return url
        return f"{parsed.scheme}://{parsed.netloc}/"

    @staticmethod
    def _bounded_int_env(name: str, *, default: int, minimum: int, maximum: int) -> int:
        try:
            value = int(os.getenv(name, str(default)))
        except ValueError:
            value = default
        return max(minimum, min(value, maximum))
