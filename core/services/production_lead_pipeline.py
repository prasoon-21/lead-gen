from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import httpx
from google import genai
from google.genai import types as genai_types

from core.services.lead_discovery_policy import (
    DEFAULT_TAVILY_EXTRACT_DEPTH,
    DEFAULT_TAVILY_SEARCH_DEPTH,
    build_targeted_directory_queries,
    filter_whitelisted_directory_results,
)
from core.services.lead_quality_service import score_lead
from core.services.lead_quality_gate import apply_quality_gate
from core.tools.base import ToolContext
from core.tools.external.linkedin_research import LinkedInResearchTool
from core.tools.external.tavily_client import TavilyClient
from core.utils.lead_summary import build_plain_lead_summary
from core.utils.phone_quality import (
    build_phone_summary,
    phone_candidates_from_values,
    phone_summary_from_text,
)


GEMINI_LEAD_ANALYST_PROMPT = (
    "You are an automated B2B Lead Analyst. Analyze the supplied text layout "
    "against the user intent profile. Output ONLY a raw, un-nested JSON string "
    "containing a qualification score from 0-10 and a 1-sentence value proposition "
    "summary. Do not include introductory, explanatory, or conversational prose analysis."
)

BROWSER_VERIFICATION_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}


@dataclass
class VerifiedTarget:
    url: str
    title: str = ""
    content: str = ""
    company_name: str = ""
    directory_url: str = ""
    status_code: int = 0
    contact_signals: Dict[str, Any] = field(default_factory=dict)


class ProductionLeadPipeline:
    """Cost-controlled four-phase lead discovery pipeline for the lead dashboard."""

    CONTACT_PATHS = ("", "contact", "about", "team")

    def __init__(
        self,
        *,
        tavily_client: Optional[TavilyClient] = None,
        gemini_api_key: Optional[str] = None,
        gemini_model: str = "gemini-2.0-flash",
    ) -> None:
        self.logger = logging.getLogger("agent.runtime")
        self.tavily = tavily_client or TavilyClient()
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.gemini_model = gemini_model
        configured_model = os.getenv("GEMINI_MODEL", "").strip()
        self.gemini_fallback_models = [
            model
            for model in (self.gemini_model, configured_model, "gemini-2.5-flash")
            if model
        ]
        try:
            self.gemini_max_calls_per_run = int(os.getenv("LEAD_GEMINI_MAX_CALLS_PER_RUN", "2"))
        except ValueError:
            self.gemini_max_calls_per_run = 2
        try:
            self.gemini_directory_analysis_enabled = os.getenv("LEAD_GEMINI_DIRECTORY_ANALYSIS", "1").strip() != "0"
        except Exception:
            self.gemini_directory_analysis_enabled = True
        try:
            self.linkedin_enrichment_limit = max(0, min(int(os.getenv("LEAD_LINKEDIN_ENRICHMENT_LIMIT", "4")), 10))
        except ValueError:
            self.linkedin_enrichment_limit = 4
        try:
            self.official_site_resolution_limit = max(6, min(int(os.getenv("LEAD_OFFICIAL_SITE_RESOLUTION_LIMIT", "12")), 20))
        except ValueError:
            self.official_site_resolution_limit = 12
        try:
            self.official_site_timeout_seconds = max(3.0, min(float(os.getenv("LEAD_OFFICIAL_SITE_TIMEOUT_SECONDS", "8")), 15.0))
        except ValueError:
            self.official_site_timeout_seconds = 8.0
        try:
            self.linkedin_timeout_seconds = max(4.0, min(float(os.getenv("LEAD_LINKEDIN_TIMEOUT_SECONDS", "8")), 15.0))
        except ValueError:
            self.linkedin_timeout_seconds = 8.0
        self._gemini_client = genai.Client(api_key=self.gemini_api_key) if self.gemini_api_key else None
        self._linkedin_tool = LinkedInResearchTool()

    async def run(
        self,
        *,
        industry: str,
        location: str,
        seed_query: str = "",
        target_count: int = 15,
    ) -> Dict[str, Any]: # Removed `Optional` from return type, as we will always return a dict.
        try:
            target_count = max(1, min(int(target_count or 15), 20))
            phase_steps: List[Dict[str, Any]] = []

            search_results = await self._discover_targets(industry=industry, location=location, seed_query=seed_query)
            phase_steps.append(
                {
                    "step": 1,
                    "stage": "target_discovery",
                    "type": "tavily_search",
                    "search_depth": DEFAULT_TAVILY_SEARCH_DEPTH,
                    "max_results": 10,
                    "result_count": len(search_results),
                }
            )

            whitelisted = filter_whitelisted_directory_results(search_results)
            candidate_seed_results = self._prioritized_candidate_results(
                search_results,
                whitelisted,
                industry=industry,
                location=location,
            )
            phase_steps.append(
                {
                    "step": 2,
                    "stage": "candidate_filtering",
                    "type": "local_validator",
                    "input_count": len(search_results),
                    "result_count": len(candidate_seed_results),
                    "trusted_directory_count": len(whitelisted),
                }
            )

            corporate_targets = await self._resolve_corporate_targets(
                candidate_seed_results,
                industry=industry,
                location=location,
                limit=max(10, target_count),
            )
            verified = await self._verify_targets(corporate_targets, limit=max(10, target_count))
            verified = await self._scrape_contact_pages_batch(verified)
            phase_steps.append(
                {
                    "step": 3,
                    "stage": "firewall_bypass_verification",
                    "type": "async_http_verification",
                    "input_count": len(corporate_targets),
                    "result_count": len(verified),
                }
            )

            extracted = await self._extract_targets(verified)
            leads = await self._score_extracted_targets(
                extracted,
                industry=industry,
                location=location,
                target_count=target_count,
            )
            leads = await self._enrich_leads_with_linkedin(
                leads,
                industry=industry,
                location=location,
            )
            leads, quality_gate_stats = apply_quality_gate(leads, drop_rejected=True)
            phase_steps.append(
                {
                    "step": 4,
                    "stage": "gemini_content_extraction_scoring",
                    "type": "tavily_extract_gemini_score",
                    "extract_depth": DEFAULT_TAVILY_EXTRACT_DEPTH,
                    "input_count": len(verified),
                    "lead_count": len(leads),
                    "model": self.gemini_model,
                }
            )

            return {
                "leads": leads,
                "steps": phase_steps,
                "metadata": {
                    "pipeline": "production_directory_refactor",
                    "search_depth": DEFAULT_TAVILY_SEARCH_DEPTH,
                    "extract_depth": DEFAULT_TAVILY_EXTRACT_DEPTH,
                    "discovered_targets": len(search_results),
                    "whitelisted_targets": len(whitelisted),
                    "candidate_targets": len(candidate_seed_results),
                    "corporate_targets": len(corporate_targets),
                    "verified_targets": len(verified),
                    "scored_targets": len(leads),
                    "quality_gate": quality_gate_stats,
                    "linkedin_enriched_targets": sum(
                        1 for lead in leads if lead.get("contact_person_name") or lead.get("founder_name")
                    ),
                    "gemini_directory_analysis": self.gemini_directory_analysis_enabled,
                    "gemini_max_calls_per_run": self.gemini_max_calls_per_run,
                },
            }
        except Exception as e:
            self.logger.error("lead_pipeline_run_failed", extra={"payload": str(e), "stacktrace": True})
            return {
                "leads": [],
                "steps": phase_steps, # Include steps completed so far
                "error": f"Internal Server Error during lead generation: {e}",
                "metadata": {
                    "pipeline": "production_directory_refactor",
                    "status": "failed",
                    "error_type": type(e).__name__,
                },
            }

    async def _discover_targets(self, *, industry: str, location: str, seed_query: str) -> List[Dict[str, Any]]:
        queries = build_targeted_directory_queries(
            industry=industry,
            location=location,
            seed_query=seed_query,
            max_queries=3,
        )
        seen_urls = set()
        results: List[Dict[str, Any]] = []
        for query in queries:
            try:
                data = await self.tavily.search(
                    {
                        "query": query,
                        "search_depth": DEFAULT_TAVILY_SEARCH_DEPTH,
                        "max_results": 10,
                        "include_raw_content": False,
                    }
                )
            except Exception as exc:
                self.logger.warning("lead_pipeline_tavily_search_failed", extra={"payload": str(exc)})
                continue
            for item in data.get("results") or []:
                url = str(item.get("url") or "").strip()
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                results.append(
                    {
                        "url": url,
                        "title": str(item.get("title") or "").strip(),
                        "content": str(item.get("content") or "").strip(),
                        "query": query,
                    }
                )
        return results

    def _prioritized_candidate_results(
        self,
        search_results: Iterable[Dict[str, Any]],
        whitelisted_results: Iterable[Dict[str, Any]],
        *,
        industry: str,
        location: str,
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        seen_urls = set()

        def add(item: Dict[str, Any], *, company_name: str = "") -> None:
            url = str(item.get("url") or "").strip()
            if not url or url in seen_urls:
                return
            seen_urls.add(url)
            normalized = dict(item)
            if company_name:
                normalized["company_name"] = company_name
            candidates.append(normalized)

        for item in search_results or []:
            url = str(item.get("url") or "").strip()
            title = str(item.get("title") or "").strip()
            content = str(item.get("content") or "").strip()
            if not url or self._is_directory_or_noise_url(url):
                continue
            company_name = self._resolve_company_name("", title=title, url=url)
            if not self._looks_like_valid_company_name(company_name):
                continue
            if self._looks_like_operating_company_result(
                url=url,
                title=title,
                content=content,
                company_name=company_name,
                industry=industry,
                location=location,
            ) or self._official_site_candidate_score(
                url=url,
                title=title,
                content=content,
                company_name=company_name,
                industry=industry,
                location=location,
            ) >= 2:
                add(item, company_name=company_name)

        for item in whitelisted_results or []:
            add(item)

        return candidates

    async def _resolve_corporate_targets(
        self,
        directory_items: Iterable[Dict[str, Any]],
        *,
        industry: str,
        location: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        directory_items = list(directory_items)
        targets: List[Dict[str, Any]] = []
        seen_companies = set()
        seen_urls = set()
        gemini_candidates = await self._gemini_directory_company_candidates(
            directory_items,
            industry=industry,
            location=location,
            limit=max(12, limit),
        )
        candidate_items = gemini_candidates + directory_items

        shortlisted_items: List[Dict[str, Any]] = []
        for item in candidate_items:
            company_name = str(item.get("company_name") or "").strip() or self._company_name_from_directory_item(item)
            url = str(item.get("url") or item.get("official_url") or "").strip()
            title = str(item.get("title") or company_name).strip()
            content = str(item.get("content") or "").strip()
            if not company_name:
                company_name = self._resolve_company_name("", title=title, url=url)
            if not company_name:
                continue
            company_key = company_name.lower()
            if company_key in seen_companies:
                continue
            seen_companies.add(company_key)
            if url and not self._is_directory_or_noise_url(url):
                if self._looks_like_operating_company_result(
                    url=url,
                    title=title,
                    content=content,
                    company_name=company_name,
                    industry=industry,
                    location=location,
                ) or self._official_site_candidate_score(
                    url=url,
                    title=title,
                    content=content,
                    company_name=company_name,
                    industry=industry,
                    location=location,
                ) >= 2:
                    if url not in seen_urls:
                        seen_urls.add(url)
                        targets.append(
                            {
                                "url": url,
                                "title": title,
                                "content": content,
                                "company_name": company_name,
                                "directory_url": str(item.get("directory_url") or "").strip(),
                            }
                        )
                    if len(targets) >= max(1, min(max(limit, 18), 20)):
                        break
                    continue
            shortlisted_items.append(
                {
                    "company_name": company_name,
                    "url": url,
                    "title": title,
                    "content": content,
                }
            )
            if len(shortlisted_items) >= self.official_site_resolution_limit:
                break

        resolution_tasks = [
            self._find_official_company_site(
                item["company_name"],
                industry=industry,
                location=location,
            )
            for item in shortlisted_items
        ]
        official_results = await asyncio.gather(*resolution_tasks, return_exceptions=True)

        for source_item, official in zip(shortlisted_items, official_results):
            if isinstance(official, Exception) or not official:
                continue
            company_name = source_item["company_name"]
            if official:
                url = str(official.get("url") or "").strip()
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    targets.append(
                        {
                            "url": url,
                            "title": str(official.get("title") or company_name).strip(),
                            "content": str(official.get("content") or source_item.get("content") or "").strip(),
                            "company_name": company_name,
                            "directory_url": str(source_item.get("url") or "").strip(),
                        }
                    )
            if len(targets) >= max(1, min(max(limit, 18), 20)):
                break

        return targets

    async def _gemini_directory_company_candidates(
        self,
        directory_items: Iterable[Dict[str, Any]],
        *,
        industry: str,
        location: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        if not self.gemini_directory_analysis_enabled or not self._gemini_client:
            return []

        compact_items = []
        for item in list(directory_items)[:30]:
            compact_items.append(
                {
                    "directory_url": str(item.get("url") or "")[:300],
                    "title": str(item.get("title") or "")[:180],
                    "snippet": str(item.get("content") or "")[:450],
                }
            )
        if not compact_items:
            return []

        prompt = json.dumps(
            {
                "task": "Extract real company names from trusted B2B directory search snippets.",
                "intent_profile": {"industry": industry, "location": location},
                "rules": [
                    "Return only actual company/provider names, not article titles, category pages, job/person names, or generic labels.",
                    "Prefer companies that appear to match the requested industry and location.",
                    "If a snippet mentions multiple companies, include the strongest one only.",
                    "Keep the original directory_url that supplied the evidence.",
                    "If the snippet already contains a likely official company website URL or domain, include it as official_url.",
                ],
                "output_schema": {
                    "companies": [
                        {
                            "company_name": "string",
                            "directory_url": "string",
                            "official_url": "string",
                            "reason": "short evidence phrase",
                        }
                    ]
                },
                "directory_results": compact_items,
                "max_companies": max(1, min(limit, 20)),
            },
            ensure_ascii=False,
        )
        parsed = await self._gemini_json(
            prompt=prompt,
            system_prompt=(
                "You are a precise B2B directory analyst. Output only JSON. "
                "Do not invent companies that are not supported by the snippets."
            ),
            max_output_tokens=1200,
        )
        companies = parsed.get("companies") if isinstance(parsed, dict) else []
        if not isinstance(companies, list):
            return []

        normalized: List[Dict[str, Any]] = []
        seen = set()
        directory_by_url = {str(item.get("url") or ""): item for item in directory_items}
        for company in companies:
            if not isinstance(company, dict):
                continue
            company_name = self._clean_company_candidate(str(company.get("company_name") or ""))
            if not company_name or company_name.lower() in seen:
                continue
            directory_url = str(company.get("directory_url") or "").strip()
            source_item = directory_by_url.get(directory_url) or {}
            seen.add(company_name.lower())
            normalized.append(
                {
                    "company_name": company_name,
                    "url": str(company.get("official_url") or "").strip() or directory_url or str(source_item.get("url") or ""),
                    "directory_url": directory_url,
                    "title": str(source_item.get("title") or company_name),
                    "content": str(company.get("reason") or source_item.get("content") or ""),
                    "query": str(source_item.get("query") or ""),
                }
            )
            if len(normalized) >= max(1, min(limit, 20)):
                break
        return normalized

    async def _find_official_company_site(self, company_name: str, *, industry: str, location: str) -> Optional[Dict[str, Any]]:
        query = f'"{company_name}" "{location}" official website'
        try:
            data = await asyncio.wait_for(
                self.tavily.search(
                    {
                        "query": query,
                        "search_depth": DEFAULT_TAVILY_SEARCH_DEPTH,
                        "max_results": 8,
                        "include_raw_content": False,
                    }
                ),
                timeout=self.official_site_timeout_seconds,
            )
        except Exception as exc:
            self.logger.info("lead_pipeline_official_site_search_failed", extra={"payload": str(exc)})
            return None

        best_item: Optional[Dict[str, Any]] = None
        best_score = -1
        for item in data.get("results") or []:
            url = str(item.get("url") or "").strip()
            if not url or self._is_directory_or_noise_url(url):
                continue
            title = str(item.get("title") or "").strip()
            content = str(item.get("content") or "").strip()
            if not self._looks_like_operating_company_result(
                url=url,
                title=title,
                content=content,
                company_name=company_name,
                industry=industry,
                location=location,
            ):
                candidate_score = self._official_site_candidate_score(
                    url=url,
                    title=title,
                    content=content,
                    company_name=company_name,
                    industry=industry,
                    location=location,
                )
                if candidate_score > best_score:
                    best_item = item
                    best_score = candidate_score
                continue
            return item
        return best_item if best_score >= 2 else None

    async def _verify_targets(self, items: Iterable[Dict[str, Any]], *, limit: int) -> List[VerifiedTarget]:
        candidates = list(items)[:20]
        timeout = httpx.Timeout(connect=4.0, read=7.0, write=4.0, pool=4.0)
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers=BROWSER_VERIFICATION_HEADERS,
        ) as client:
            tasks = [self._verify_one(client, item) for item in candidates]
            verified = [target for target in await asyncio.gather(*tasks) if target]
        return verified[: max(1, min(limit, 20))]

    async def _verify_one(self, client: httpx.AsyncClient, item: Dict[str, Any]) -> Optional[VerifiedTarget]:
        url = str(item.get("url") or "").strip()
        if not url:
            return None
        if not re.match(r"^https?://", url, re.IGNORECASE):
            url = f"https://{url.lstrip('/')}"
        try:
            response = await client.get(url)
            text_sample = (response.text or "")[:1500].lower()
            blocked = "blocked page" in text_sample or "access denied" in text_sample
            if response.status_code >= 400 or blocked:
                return None
            return VerifiedTarget(
                url=str(response.url),
                title=str(item.get("title") or "").strip(),
                content=str(item.get("content") or "").strip(),
                company_name=str(item.get("company_name") or "").strip(),
                directory_url=str(item.get("directory_url") or "").strip(),
                status_code=response.status_code,
            )
        except Exception as exc:
            self.logger.info("lead_pipeline_verify_skip", extra={"payload": json.dumps({"url": url, "error": str(exc)})})
            return None

    async def _scrape_contact_pages_batch(self, targets: List[VerifiedTarget]) -> List[VerifiedTarget]:
        if not targets:
            return []
        timeout = httpx.Timeout(connect=4.0, read=6.0, write=4.0, pool=4.0)
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers=BROWSER_VERIFICATION_HEADERS,
        ) as client:
            scraped = await asyncio.gather(
                *(self._scrape_contact_pages_for_target(client, target) for target in targets),
                return_exceptions=True,
            )
        enriched: List[VerifiedTarget] = []
        for index, item in enumerate(scraped):
            target = targets[index]
            if isinstance(item, Exception) or not isinstance(item, dict):
                enriched.append(target)
                continue
            target.contact_signals = item
            signal_text = self._contact_signals_to_text(item)
            if signal_text:
                target.content = f"{target.content}\n\nFree contact page signals:\n{signal_text}".strip()
            enriched.append(target)
        return enriched

    async def _scrape_contact_pages_for_target(
        self,
        client: httpx.AsyncClient,
        target: VerifiedTarget,
    ) -> Dict[str, Any]:
        base_home = self._homepage_url(target.url)
        signals: Dict[str, Any] = {
            "emails": [],
            "phones": [],
            "linkedin_urls": [],
            "person_names": [],
            "person_titles": [],
            "pages": [],
            "text": "",
        }
        text_parts: List[str] = []
        seen_pages = set()
        for path in self.CONTACT_PATHS:
            page_url = urljoin(base_home.rstrip("/") + "/", path)
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
            visible_text = self._html_to_text(html)
            text_parts.append(visible_text[:8000])
            signals["pages"].append(page_url)
            self._collect_contact_page_signals(signals, html, visible_text, source_url=page_url)
        signals["text"] = "\n".join(part for part in text_parts if part)[:20000]
        return {key: value for key, value in signals.items() if value}

    async def _extract_targets(self, targets: List[VerifiedTarget]) -> List[Dict[str, Any]]:
        if not targets:
            return []
        normalized: List[Dict[str, Any]] = []
        extract_urls: List[str] = []
        target_by_url: Dict[str, VerifiedTarget] = {}
        for target in targets:
            if self._has_good_contact_signals(target.contact_signals):
                normalized.append(self._normalized_target_without_extract(target))
                continue
            extract_url = self._best_extract_url_for_target(target)
            if extract_url not in target_by_url:
                extract_urls.append(extract_url)
                target_by_url[extract_url] = target
        if not extract_urls:
            return normalized
        try:
            data = await self.tavily.extract(
                {
                    "urls": extract_urls,
                    "extract_depth": DEFAULT_TAVILY_EXTRACT_DEPTH,
                    "include_images": False,
                }
            )
        except Exception as exc:
            self.logger.warning("lead_pipeline_tavily_extract_failed", extra={"payload": str(exc)})
            normalized.extend(self._normalized_target_without_extract(target) for target in target_by_url.values())
            return normalized

        extracted_items = data.get("results") or data.get("data") or []
        if isinstance(extracted_items, dict):
            extracted_items = [extracted_items]

        extracted_target_ids = set()
        for item in extracted_items:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or item.get("source_url") or "").strip()
            target = target_by_url.get(url) or next((candidate for candidate in targets if candidate.url.rstrip("/") == url.rstrip("/")), None)
            text = self._clean_markdown(item.get("raw_content") or item.get("content") or item.get("text") or "")
            contact_text = self._contact_signals_to_text(target.contact_signals if target else {})
            combined_text = "\n\n".join(part for part in (text, target.content if target else "", contact_text) if part)
            if target:
                extracted_target_ids.add(id(target))
            normalized.append(
                {
                    "url": target.url if target else url,
                    "extract_url": url,
                    "title": str(item.get("title") or (target.title if target else "") or "").strip(),
                    "content": self._clean_markdown(combined_text),
                    "company_name": target.company_name if target else "",
                    "directory_url": target.directory_url if target else "",
                    "contact_signals": target.contact_signals if target else {},
                }
            )
        for target in target_by_url.values():
            if id(target) not in extracted_target_ids:
                normalized.append(self._normalized_target_without_extract(target))
        return normalized

    def _normalized_target_without_extract(self, target: VerifiedTarget) -> Dict[str, Any]:
        contact_text = self._contact_signals_to_text(target.contact_signals)
        combined_text = "\n\n".join(part for part in (target.content, contact_text) if part)
        return {
            "url": target.url,
            "extract_url": "",
            "title": target.title,
            "content": self._clean_markdown(combined_text),
            "raw_content": self._clean_markdown(combined_text),
            "company_name": target.company_name,
            "directory_url": target.directory_url,
            "contact_signals": target.contact_signals,
        }

    def _best_extract_url_for_target(self, target: VerifiedTarget) -> str:
        signals = target.contact_signals or {}
        pages = [str(page or "").strip() for page in signals.get("pages") or []]
        if not (signals.get("emails") and signals.get("phones")):
            contact_page = next((page for page in pages if "/contact" in urlparse(page).path.lower()), "")
            if contact_page:
                return contact_page
            return urljoin(self._homepage_url(target.url).rstrip("/") + "/", "contact")
        return target.url

    @staticmethod
    def _has_good_contact_signals(signals: Dict[str, Any]) -> bool:
        if not signals:
            return False
        emails = signals.get("emails") or []
        phones = signals.get("phones") or []
        people = signals.get("person_names") or []
        return bool(emails and phones and people)

    @classmethod
    def _contact_signals_to_text(cls, signals: Dict[str, Any]) -> str:
        if not signals:
            return ""
        lines: List[str] = []
        for label, key in (
            ("emails", "emails"),
            ("phones", "phones"),
            ("linkedin", "linkedin_urls"),
            ("people", "person_names"),
            ("titles", "person_titles"),
            ("pages", "pages"),
        ):
            values = [str(value).strip() for value in signals.get(key) or [] if str(value).strip()]
            if values:
                lines.append(f"{label}: {' | '.join(values[:5])}")
        if signals.get("text"):
            lines.append(f"text: {str(signals.get('text'))[:6000]}")
        return "\n".join(lines)

    @classmethod
    def _collect_contact_page_signals(
        cls,
        signals: Dict[str, Any],
        html: str,
        visible_text: str,
        *,
        source_url: str,
    ) -> None:
        combined = cls._deobfuscate_contact_text(f"{html}\n{visible_text}")
        for email in cls._valid_email_matches(combined):
            if email not in signals["emails"]:
                signals["emails"].append(email)
        phone_summary = phone_summary_from_text(combined, source_type="contact_page", source_url=source_url)
        for phone in [phone_summary.get("contact_phone", "")] + list(phone_summary.get("alternate_phones") or []):
            if phone and phone not in signals["phones"]:
                signals["phones"].append(phone)
        for url in re.findall(r"https?://(?:www\.)?linkedin\.com/[^\s\"'<>]+", combined, re.IGNORECASE):
            cleaned = url.strip(" .,:;()[]{}<>\"'")
            if cleaned and cleaned not in signals["linkedin_urls"]:
                signals["linkedin_urls"].append(cleaned)
        for name, title in cls._extract_people_from_contact_text(visible_text):
            if name and name not in signals["person_names"]:
                signals["person_names"].append(name)
            if title and title not in signals["person_titles"]:
                signals["person_titles"].append(title)

    @classmethod
    def _extract_people_from_contact_text(cls, text: str) -> List[Tuple[str, str]]:
        cleaned = re.sub(r"\s+", " ", text or "").strip()
        patterns = (
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s*[-,|]\s*((?:Founder|Co-Founder|CEO|Owner|President|Director|Managing Director|Principal)[^.;|]{0,80})",
            r"\b(?:Founder|Co-Founder|CEO|Owner|President|Director|Managing Director|Principal)\s*[-,|:]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})",
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}),?\s+(?:is\s+)?(?:the\s+)?((?:founder|co-founder|ceo|owner|president|director|managing director|principal)[^.;|]{0,80})",
        )
        people: List[Tuple[str, str]] = []
        seen = set()
        for pattern in patterns:
            for match in re.finditer(pattern, cleaned, re.IGNORECASE):
                if len(match.groups()) == 1:
                    name, title = match.group(1), ""
                elif re.search(r"founder|ceo|owner|president|director|principal", match.group(1), re.IGNORECASE):
                    title, name = match.group(1), match.group(2)
                else:
                    name, title = match.group(1), match.group(2)
                name = re.sub(r"\s+", " ", name).strip(" -|,")
                title = re.sub(r"\s+", " ", title).strip(" -|,")
                title = re.split(
                    r"\b(?:email|e-mail|call|phone|tel|mobile|contact)\b|[A-Z0-9._%+\-]+\s*(?:@|\[at\]|\(at\))",
                    title,
                    maxsplit=1,
                    flags=re.IGNORECASE,
                )[0].strip(" -|,")
                if not cls._looks_like_person_name(name):
                    continue
                key = name.lower()
                if key not in seen:
                    seen.add(key)
                    people.append((name[:120], title[:160]))
                if len(people) >= 5:
                    return people
        return people

    @staticmethod
    def _html_to_text(html: str) -> str:
        text = re.sub(r"(?is)<(script|style|noscript).*?</\1>", " ", html or "")
        text = re.sub(r"(?s)<[^>]+>", " ", text)
        text = (
            text.replace("&amp;", "&")
            .replace("&nbsp;", " ")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&#64;", "@")
        )
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _deobfuscate_contact_text(text: str) -> str:
        cleaned = text or ""
        cleaned = re.sub(r"\s*(?:\[|\()\s*at\s*(?:\]|\))\s*", "@", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+(?:at)\s+", "@", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*(?:\[|\()\s*dot\s*(?:\]|\))\s*", ".", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+(?:dot)\s+", ".", cleaned, flags=re.IGNORECASE)
        return cleaned

    async def _score_extracted_targets(
        self,
        items: List[Dict[str, Any]],
        *,
        industry: str,
        location: str,
        target_count: int,
    ) -> List[Dict[str, Any]]:
        candidate_items = items[: max(10, target_count)]
        gemini_by_url = await self._gemini_batch_contact_scores(
            candidate_items,
            industry=industry,
            location=location,
        )
        scored: List[Dict[str, Any]] = []
        for item in candidate_items:
            item_url = str(item.get("url") or "").strip()
            lead = await self._score_one(
                item,
                industry=industry,
                location=location,
                use_gemini=False,
                gemini_data=gemini_by_url.get(item_url, {}),
            )
            if lead:
                scored.append(lead)
        scored.sort(key=lambda item: item.get("quality_score", 0), reverse=True)
        return scored[:target_count]

    async def _enrich_leads_with_linkedin(
        self,
        leads: List[Dict[str, Any]],
        *,
        industry: str,
        location: str,
    ) -> List[Dict[str, Any]]:
        if not leads or self.linkedin_enrichment_limit <= 0:
            return leads
        if not os.path.exists(os.path.join(os.getcwd(), "linkedin_session.json")):
            return leads

        eligible_indexes = [
            index
            for index, lead in enumerate(leads)
            if not (
                self._first_non_empty(lead.get("contact_email"))
                and self._first_non_empty(lead.get("contact_person_name"), lead.get("founder_name"))
            )
        ]
        eligible_indexes = set(eligible_indexes[: self.linkedin_enrichment_limit])

        async def enrich_one(index: int, lead: Dict[str, Any]) -> Dict[str, Any]:
            item = dict(lead)
            if index not in eligible_indexes:
                return item

            linkedin_url = str(item.get("linkedin_url") or "").strip()
            linkedin_candidate: Dict[str, str] = {}
            if not linkedin_url:
                try:
                    linkedin_candidate = await asyncio.wait_for(
                        self._find_linkedin_candidate_for_company(
                            company_name=str(item.get("company_name") or ""),
                            industry=industry,
                            location=location,
                        ),
                        timeout=5,
                    )
                except Exception:
                    linkedin_candidate = {}
                linkedin_url = str(linkedin_candidate.get("url") or "").strip()
                if linkedin_url:
                    item["linkedin_url"] = linkedin_url
                    item.setdefault("source_details", []).append(
                        {
                            "stage": "linkedin_search",
                            "type": "linkedin_search_result",
                            "provider": "tavily_search",
                            "url": linkedin_url,
                            "value": self._first_non_empty(
                                linkedin_candidate.get("person_name"),
                                linkedin_candidate.get("title_hint"),
                            ),
                        }
                    )

            if linkedin_url:
                linkedin_data = await self._run_linkedin_research(
                    linkedin_url=linkedin_url,
                    company_name=str(item.get("company_name") or ""),
                    website_url=str(item.get("company_website") or ""),
                )
                if linkedin_data:
                    item = self._merge_linkedin_data(item, linkedin_data)
                if linkedin_candidate and not self._first_non_empty(
                    item.get("contact_person_name"),
                    item.get("founder_name"),
                ):
                    item = self._merge_linkedin_search_fallback(item, linkedin_candidate)
            return item

        enriched_results = await asyncio.gather(
            *(enrich_one(index, lead) for index, lead in enumerate(leads)),
            return_exceptions=True,
        )
        enriched = [
            dict(leads[index]) if isinstance(item, Exception) else item
            for index, item in enumerate(enriched_results)
        ]

        enriched.sort(
            key=lambda item: (
                item.get("quality_score", 0),
                1 if item.get("contact_email") else 0,
                1 if item.get("contact_person_name") or item.get("founder_name") else 0,
                1 if item.get("linkedin_url") else 0,
            ),
            reverse=True,
        )
        return enriched

    async def _score_one(
        self,
        item: Dict[str, Any],
        *,
        industry: str,
        location: str,
        use_gemini: bool = True,
        gemini_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        url = str(item.get("url") or "").strip()
        if not url:
            return None
        title = str(item.get("title") or "").strip()
        text = self._clean_markdown(item.get("content") or "")
        contact_signals = item.get("contact_signals") if isinstance(item.get("contact_signals"), dict) else {}
        gemini_data = gemini_data or {}
        ai_score = (
            await self._gemini_score(title=title, url=url, text=text, industry=industry, location=location)
            if use_gemini
            else {
                "qualification_score": self._first_non_empty(
                    gemini_data.get("qualification_score"),
                    self._heuristic_score(text=text, industry=industry, location=location),
                ),
                "value_proposition": self._first_non_empty(gemini_data.get("value_proposition"), gemini_data.get("summary")),
                **gemini_data,
            }
        )

        score_10 = self._coerce_score_10(ai_score.get("qualification_score", 0))
        summary = str(ai_score.get("value_proposition") or ai_score.get("summary") or "").strip()
        company_name = self._resolve_company_name(
            str(ai_score.get("company_name") or item.get("company_name") or "").strip(),
            title=title,
            url=url,
        )
        directory_url = str(item.get("directory_url") or "").strip()
        if not self._looks_like_valid_company_name(company_name):
            return None
        if not self._looks_like_operating_company_result(
            url=url,
            title=title,
            content=text,
            company_name=company_name,
            industry=industry,
            location=location,
        ):
            return None
        summary = summary or self._fallback_value_proposition(
            company_name=company_name,
            text=text,
            industry=industry,
            location=location,
        )
        phone_summary = phone_summary_from_text(text, source_type="tavily_extract", source_url=url)
        scraped_phone_summary = build_phone_summary(
            phone_candidates_from_values(
                list(contact_signals.get("phones") or []) + [ai_score.get("contact_phone")],
                source_type="contact_page",
                source_url=url,
            )
        )
        contact_email = self._first_non_empty(
            *(contact_signals.get("emails") or [])[:1],
            ai_score.get("contact_email"),
            self._first_valid_email(text),
        )
        person_name = self._first_non_empty(
            *(contact_signals.get("person_names") or [])[:1],
            ai_score.get("person_name"),
            ai_score.get("contact_person_name"),
            ai_score.get("founder_name"),
        )
        person_title = self._first_non_empty(
            *(contact_signals.get("person_titles") or [])[:1],
            ai_score.get("person_title"),
            ai_score.get("contact_person_title"),
        )
        linkedin_url = self._first_non_empty(
            *(contact_signals.get("linkedin_urls") or [])[:1],
            ai_score.get("linkedin_url"),
            self._first_linkedin_url(text),
        )
        lead = {
            "company_name": company_name,
            "company_website": url,
            "industry": industry or "General",
            "location": location or "",
            "location_evidence": self._location_evidence_excerpt(text=f"{title}\n{text}", location=location),
            "value_proposition": summary,
            "lead_summary": "",
            "company_size": "",
            "contact_person_name": person_name,
            "founder_name": person_name,
            "contact_person_title": person_title,
            "contact_email": contact_email,
            "contact_phone": self._first_non_empty(scraped_phone_summary.get("contact_phone"), phone_summary.get("contact_phone")),
            "alternate_phones": phone_summary.get("alternate_phones", []),
            "phone_confidence": max(int(scraped_phone_summary.get("phone_confidence") or 0), int(phone_summary.get("phone_confidence") or 0)),
            "phone_source": self._first_non_empty(scraped_phone_summary.get("phone_source"), phone_summary.get("phone_source")),
            "phone_validation_status": self._first_non_empty(scraped_phone_summary.get("phone_validation_status"), phone_summary.get("phone_validation_status")),
            "phone_type": self._first_non_empty(scraped_phone_summary.get("phone_type"), phone_summary.get("phone_type")),
            "phone_candidates": scraped_phone_summary.get("phone_candidates") or phone_summary.get("phone_candidates", []),
            "linkedin_url": linkedin_url,
            "contact_page": url,
            "source": "trusted_directory",
            "source_details": [
                {
                    "stage": "production_pipeline",
                    "type": "verified_extracted_target",
                    "provider": "free_contact_scrape" if not item.get("extract_url") else "tavily_extract",
                    "url": str(item.get("extract_url") or url),
                    "title": title,
                }
            ] + (
                [
                    {
                        "stage": "target_discovery",
                        "type": "trusted_directory_seed",
                        "provider": "tavily_search",
                        "url": directory_url,
                    }
                ]
                if directory_url
                else []
            ),
            "confidence": "high" if score_10 >= 8 else "medium" if score_10 >= 5 else "low",
            "qualification_score": score_10,
            "quality_score": score_10 * 10,
            "usable": score_10 >= 4,
            "rejected": score_10 < 4,
            "verification_status": "valid",
            "verification_message": "Verified with browser-like headers before extraction.",
            "notes": summary,
        }
        self._apply_phone_summary(lead, phone_summary)
        self._apply_phone_summary(lead, scraped_phone_summary)
        lead["lead_summary"] = build_plain_lead_summary(lead)
        quality = score_lead(lead)
        lead.update({k: v for k, v in quality.items() if k != "quality_score"})
        lead["quality_score"] = max(int(lead["quality_score"]), int(quality.get("quality_score") or 0))
        lead["quality_score"] = max(0, min(int(lead["quality_score"]), 100))
        return lead

    async def _find_linkedin_candidate_for_company(
        self,
        *,
        company_name: str,
        industry: str,
        location: str,
    ) -> Dict[str, str]:
        cleaned_name = str(company_name or "").strip()
        if not cleaned_name:
            return {}

        query = f'site:linkedin.com "{cleaned_name}" founder OR ceo OR owner'
        best_company_candidate: Dict[str, str] = {}
        try:
            data = await self.tavily.search(
                {
                    "query": query,
                    "search_depth": DEFAULT_TAVILY_SEARCH_DEPTH,
                    "max_results": 6,
                    "include_raw_content": False,
                }
            )
        except Exception as exc:
            self.logger.info("lead_pipeline_linkedin_search_failed", extra={"payload": str(exc)})
            return {}

        for result in data.get("results") or []:
            candidate_url = str(result.get("url") or "").strip()
            if not candidate_url or "linkedin.com/" not in candidate_url:
                continue
            title = str(result.get("title") or "").strip()
            content = str(result.get("content") or "").strip()
            if "linkedin.com/in/" in candidate_url:
                return {
                    "url": candidate_url,
                    "person_name": self._extract_person_name_from_linkedin_result(title=title, content=content),
                    "title_hint": self._extract_person_title_from_linkedin_result(title=title, content=content),
                }
            if "linkedin.com/company/" in candidate_url and not best_company_candidate:
                best_company_candidate = {
                    "url": candidate_url,
                    "person_name": "",
                    "title_hint": "",
                }
        return best_company_candidate

    async def _run_linkedin_research(self, *, linkedin_url: str, company_name: str, website_url: str) -> Dict[str, Any]:
        context = ToolContext(
            session_id=f"lead-pipeline-{uuid.uuid4().hex[:12]}",
            trace_id=f"lead-pipeline-{uuid.uuid4().hex[:12]}",
            state={
                "company_name": company_name,
                "company_website": website_url,
            },
        )
        try:
            result = await asyncio.wait_for(
                self._linkedin_tool.run(
                    {
                        "url": linkedin_url,
                        "company_name": company_name,
                        "website_url": website_url,
                        "target_roles": ["founder", "owner", "ceo", "director", "managing director"],
                        "allow_fallback_contact_paths": True,
                    },
                    context,
                ),
                timeout=self.linkedin_timeout_seconds,
            )
        except Exception:
            return {}
        if not isinstance(result, dict) or result.get("error"):
            return {}
        return result

    @classmethod
    def _merge_linkedin_data(cls, lead: Dict[str, Any], linkedin_data: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(lead)
        linkedin_company_name = cls._clean_company_candidate(str(linkedin_data.get("company_name") or ""))
        if not cls._looks_like_valid_company_name(linkedin_company_name):
            linkedin_company_name = ""
        merged.update(
            {
                "company_name": cls._first_non_empty(linkedin_company_name, merged.get("company_name")),
                "company_website": cls._first_non_empty(linkedin_data.get("company_website"), merged.get("company_website")),
                "contact_person_name": cls._first_non_empty(
                    linkedin_data.get("contact_person_name"),
                    linkedin_data.get("full_name"),
                    merged.get("contact_person_name"),
                    merged.get("founder_name"),
                ),
                "founder_name": cls._first_non_empty(
                    linkedin_data.get("founder_name"),
                    linkedin_data.get("full_name"),
                    merged.get("founder_name"),
                    merged.get("contact_person_name"),
                ),
                "contact_person_title": cls._first_non_empty(
                    linkedin_data.get("contact_person_title"),
                    merged.get("contact_person_title"),
                ),
                "contact_email": cls._first_non_empty(merged.get("contact_email"), linkedin_data.get("contact_email")),
                "contact_phone": cls._first_non_empty(merged.get("contact_phone"), linkedin_data.get("contact_phone")),
                "linkedin_url": cls._first_non_empty(linkedin_data.get("linkedin_url"), merged.get("linkedin_url")),
                "contact_page": cls._first_non_empty(merged.get("contact_page"), linkedin_data.get("contact_page")),
                "source": cls._first_non_empty(merged.get("source"), linkedin_data.get("source"), "linkedin"),
                "value_proposition": cls._first_non_empty(
                    merged.get("value_proposition"),
                    linkedin_data.get("value_proposition"),
                    linkedin_data.get("tagline"),
                ),
                "company_size": cls._first_non_empty(linkedin_data.get("company_size"), merged.get("company_size")),
                "confidence": cls._first_non_empty(linkedin_data.get("confidence"), merged.get("confidence")),
            }
        )
        linkedin_phone_summary = build_phone_summary(
            phone_candidates_from_values(
                [linkedin_data.get("contact_phone")],
                source_type="linkedin_research",
                source_url=merged.get("linkedin_url") or merged.get("company_website") or "",
            )
        )
        cls._apply_phone_summary(merged, linkedin_phone_summary)
        merged.setdefault("source_details", []).append(
            {
                "stage": "linkedin_enrichment",
                "type": "linkedin_research",
                "provider": "linkedin_research",
                "url": cls._first_non_empty(linkedin_data.get("linkedin_url"), merged.get("linkedin_url")),
                "value": cls._first_non_empty(linkedin_data.get("contact_person_name"), linkedin_data.get("full_name")),
            }
        )
        quality = score_lead(merged)
        merged.update({k: v for k, v in quality.items() if k != "quality_score"})
        merged["quality_score"] = max(int(merged.get("quality_score") or 0), int(quality.get("quality_score") or 0))
        merged["quality_score"] = max(0, min(int(merged["quality_score"]), 100))
        merged["lead_summary"] = build_plain_lead_summary(merged)
        return merged

    @classmethod
    def _merge_linkedin_search_fallback(cls, lead: Dict[str, Any], linkedin_candidate: Dict[str, str]) -> Dict[str, Any]:
        merged = dict(lead)
        person_name = cls._first_non_empty(linkedin_candidate.get("person_name"))
        title_hint = cls._first_non_empty(linkedin_candidate.get("title_hint"))
        if person_name:
            merged["contact_person_name"] = cls._first_non_empty(merged.get("contact_person_name"), person_name)
            merged["founder_name"] = cls._first_non_empty(merged.get("founder_name"), person_name)
        if title_hint:
            merged["contact_person_title"] = cls._first_non_empty(merged.get("contact_person_title"), title_hint)
        if linkedin_candidate.get("url"):
            merged["linkedin_url"] = cls._first_non_empty(merged.get("linkedin_url"), linkedin_candidate.get("url"))
        merged.setdefault("source_details", []).append(
            {
                "stage": "linkedin_search_fallback",
                "type": "linkedin_search_name_hint",
                "provider": "tavily_search",
                "url": cls._first_non_empty(linkedin_candidate.get("url"), merged.get("linkedin_url")),
                "value": person_name,
            }
        )
        quality = score_lead(merged)
        merged.update({k: v for k, v in quality.items() if k != "quality_score"})
        merged["quality_score"] = max(int(merged.get("quality_score") or 0), int(quality.get("quality_score") or 0))
        merged["quality_score"] = max(0, min(int(merged["quality_score"]), 100))
        merged["lead_summary"] = build_plain_lead_summary(merged)
        return merged

    async def _gemini_batch_contact_scores(
        self,
        items: List[Dict[str, Any]],
        *,
        industry: str,
        location: str,
    ) -> Dict[str, Dict[str, Any]]:
        if not self._gemini_client or self.gemini_max_calls_per_run <= 0 or not items:
            return {}
        compact_items = []
        for item in items[:20]:
            compact_items.append(
                {
                    "url": str(item.get("url") or "")[:300],
                    "title": str(item.get("title") or "")[:180],
                    "company_name_hint": str(item.get("company_name") or "")[:160],
                    "text": str(item.get("content") or "")[:7000],
                }
            )
        prompt = json.dumps(
            {
                "task": "Score B2B leads and extract contact details from website/contact-page text.",
                "intent_profile": {"industry": industry, "location": location},
                "rules": [
                    "Return one result for each input URL.",
                    "Use only details supported by the supplied text.",
                    "Extract obfuscated emails such as name [at] company [dot] com when clear.",
                    "Prefer founders, CEOs, owners, presidents, directors, or managing directors as person contacts.",
                    "Use qualification_score from 0 to 10.",
                ],
                "output_schema": {
                    "leads": [
                        {
                            "url": "string",
                            "company_name": "string",
                            "qualification_score": 0,
                            "value_proposition": "string",
                            "contact_email": "string",
                            "contact_phone": "string",
                            "person_name": "string",
                            "person_title": "string",
                            "linkedin_url": "string",
                        }
                    ]
                },
                "targets": compact_items,
            },
            ensure_ascii=False,
        )
        parsed = await self._gemini_json(
            prompt=prompt,
            system_prompt=(
                "You are a precise B2B lead data extractor. Output only JSON. "
                "Do not invent contact details or people that are not supported by the text."
            ),
            max_output_tokens=2400,
        )
        leads = parsed.get("leads") if isinstance(parsed, dict) else []
        if not isinstance(leads, list):
            return {}
        by_url: Dict[str, Dict[str, Any]] = {}
        for lead in leads:
            if not isinstance(lead, dict):
                continue
            url = str(lead.get("url") or "").strip()
            if not url:
                continue
            by_url[url] = lead
        return by_url

    async def _gemini_score(self, *, title: str, url: str, text: str, industry: str, location: str) -> Dict[str, Any]:
        if not self._gemini_client:
            return {"qualification_score": 5, "value_proposition": ""}
        prompt = json.dumps(
            {
                "intent_profile": {"industry": industry, "location": location},
                "target": {"title": title, "url": url},
                "text": text[:12000],
                "required_json_keys": ["qualification_score", "value_proposition"],
            },
            ensure_ascii=False,
        )
        parsed = await self._gemini_json(
            prompt=prompt,
            system_prompt=GEMINI_LEAD_ANALYST_PROMPT,
            max_output_tokens=300,
        )
        if parsed:
            if "qualification score" in parsed and "qualification_score" not in parsed:
                parsed["qualification_score"] = parsed.get("qualification score")
            return parsed
        return {"qualification_score": 5, "value_proposition": ""}

    async def _gemini_json(
        self,
        *,
        prompt: str,
        system_prompt: str,
        max_output_tokens: int,
    ) -> Dict[str, Any]:
        if not self._gemini_client:
            self.logger.warning("lead_pipeline_gemini_client_not_initialized", extra={"payload": "GEMINI_API_KEY is missing or invalid."})
            return {}
        
        last_error = ""
        for model in dict.fromkeys(self.gemini_fallback_models):
            try:
                response = await self._gemini_client.aio.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=max_output_tokens,
                        system_instruction=system_prompt,
                        response_mime_type="application/json",
                    ),
                )
                raw = (getattr(response, "text", "") or "").strip()

                if not raw:
                    last_error = f"{model}: Empty response from Gemini API."
                    continue

                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, dict):
                        return parsed
                    else:
                        last_error = f"{model}: Gemini API returned non-dict JSON: {raw[:200]}"
                        continue
                except json.JSONDecodeError as json_exc:
                    last_error = f"{model}: JSONDecodeError: {json_exc}. Raw response: {raw[:500]}"
                    continue
            except Exception as exc: # Catch general exceptions from the API call itself
                last_error = f"{model}: General Exception during Gemini API call: {exc}"
                continue
        if last_error:
            self.logger.warning("lead_pipeline_gemini_json_failed", extra={"payload": last_error})
        return {}

    @staticmethod
    def _clean_markdown(value: Any) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        return text[:30000]

    @staticmethod
    def _coerce_score_10(value: Any) -> int:
        if isinstance(value, (int, float)):
            return max(0, min(int(round(value)), 10))
        match = re.search(r"\d+(?:\.\d+)?", str(value or ""))
        if not match:
            return 5
        return max(0, min(int(round(float(match.group(0)))), 10))

    @staticmethod
    def _heuristic_score(*, text: str, industry: str, location: str) -> int:
        evidence = (text or "").lower()
        score = 4
        if industry and any(token in evidence for token in re.findall(r"[a-z0-9]+", industry.lower()) if len(token) > 2):
            score += 2
        if location and location.lower() in evidence:
            score += 1
        if re.search(r"[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}", text or "", re.IGNORECASE):
            score += 1
        if "linkedin.com" in evidence:
            score += 1
        if any(word in evidence for word in ("contact", "services", "solutions", "development", "consulting")):
            score += 1
        return max(0, min(score, 10))

    @classmethod
    def _company_name_from_title_or_url(cls, title: str, url: str) -> str:
        domain_name = cls._company_name_from_url(url)
        domain_tokens = set(cls._company_name_tokens(domain_name))
        segments = re.split(r"\s+(?:\||-|\u2013|\u2014)\s+|\s*:\s*", title or "")
        scored: List[Tuple[int, str]] = []
        for index, segment in enumerate(segments):
            candidate = re.sub(r"\s+", " ", segment).strip(" -.,")
            if not candidate or cls._is_generic_company_title(candidate):
                continue
            tokens = cls._company_name_tokens(candidate)
            if not tokens or len(tokens) > 8:
                continue
            overlap = len(set(tokens) & domain_tokens)
            score = overlap * 5
            score += 3 if 1 <= len(tokens) <= 5 else 1
            score += 1 if index == 0 else 0
            if domain_tokens and set(tokens) == domain_tokens:
                score += 4
            scored.append((score, candidate[:120]))
        if scored:
            return max(scored, key=lambda item: (item[0], len(item[1])))[1]
        return domain_name

    @classmethod
    def _resolve_company_name(cls, value: str, *, title: str, url: str) -> str:
        current = cls._clean_company_candidate(value)
        resolved = cls._clean_company_candidate(cls._company_name_from_title_or_url(title, url))
        if not cls._looks_like_valid_company_name(current):
            return resolved or cls._company_name_from_url(url)
        if not cls._looks_like_valid_company_name(resolved):
            return current

        current_tokens = set(cls._company_name_tokens(current))
        resolved_tokens = set(cls._company_name_tokens(resolved))
        if current_tokens and current_tokens < resolved_tokens:
            return resolved
        return current

    @staticmethod
    def _company_name_tokens(value: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", value.lower())

    @staticmethod
    def _is_generic_company_title(value: str) -> bool:
        cleaned = re.sub(r"[^a-z0-9]+", " ", value or "", flags=re.IGNORECASE).strip().lower()
        generic_titles = {
            "home",
            "homepage",
            "home page",
            "official site",
            "official website",
            "website",
            "contact",
            "contact us",
            "about",
            "about us",
            "services",
            "service",
            "products",
            "locations",
            "gallery",
            "blog",
            "index",
            "welcome",
        }
        title_parts = re.split(r"\s+(?:\||-|\u2013|\u2014)\s+|\s*:\s*", value or "")
        first_part = re.sub(r"[^a-z0-9]+", " ", title_parts[0], flags=re.IGNORECASE).strip().lower()
        if cleaned in generic_titles or (len(title_parts) > 1 and first_part in generic_titles):
            return True
        if re.match(
            r"^(?:(?:a|an|the|your)\s+)?(?:leading|premier|trusted|professional|expert|best|top|local|full service|one stop)\b",
            cleaned,
        ):
            return True
        return bool(
            re.search(
                r"\b(?:offers?|provides?|speciali[sz]es?|serving|learn more|near me|solutions for|services in)\b",
                cleaned,
            )
        )

    @staticmethod
    def _company_name_from_url(url: str) -> str:
        host = (urlparse(url or "").hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        label = host.split(".")[0].replace("-", " ").replace("_", " ").strip()
        if not label:
            return "Unknown Company"
        if " " not in label:
            words = (
                "upfitters",
                "upfitter",
                "outfitters",
                "interiors",
                "conversion",
                "conversions",
                "trailers",
                "trailer",
                "trucks",
                "truck",
                "campervans",
                "campers",
                "camper",
                "quest",
                "vans",
                "van",
                "motors",
                "auto",
                "autos",
                "coach",
                "coaches",
                "designs",
                "design",
                "customs",
                "custom",
                "adventure",
                "offroad",
                "coast",
                "life",
                "solutions",
                "systems",
                "homes",
                "rv",
                "ak",
            )
            remaining = label
            suffixes: List[str] = []
            while remaining:
                suffix = next(
                    (
                        word
                        for word in sorted(words, key=len, reverse=True)
                        if len(remaining) > len(word) and remaining.endswith(word)
                    ),
                    "",
                )
                if not suffix:
                    break
                suffixes.insert(0, suffix)
                remaining = remaining[: -len(suffix)]
            label = " ".join(([remaining] if remaining else []) + suffixes)
        tokens = [token for token in re.split(r"\s+", label) if token]
        acronym_tokens = {"abc", "rv", "usa", "us", "ak", "4x4"}
        pretty_tokens = [token.upper() if token.lower() in acronym_tokens else token.capitalize() for token in tokens]
        return " ".join(pretty_tokens) if pretty_tokens else "Unknown Company"

    @classmethod
    def _company_name_from_directory_item(cls, item: Dict[str, Any]) -> str:
        text = " ".join(
            str(item.get(key) or "").strip()
            for key in ("content", "title")
            if str(item.get(key) or "").strip()
        )
        text = re.sub(r"\s+", " ", text).strip()
        patterns = (
            r"([A-Z][A-Za-z0-9&.,'() -]{2,80}?)\s+is\s+(?:a|an|the)\s+",
            r"([A-Z][A-Za-z0-9&.,'() -]{2,80}?),\s+(?:a|an)\s+",
            r"([A-Z][A-Za-z0-9&.,'() -]{2,80}?)\s+is\s+(?:located|in)\s+",
            r"([A-Z][A-Za-z0-9&.,'() -]{2,80}?)\.\s+(?:Not yet reviewed|Their services|The team)",
        )
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                candidate = cls._clean_company_candidate(match.group(1))
                if candidate:
                    return candidate
        return ""

    @staticmethod
    def _clean_company_candidate(value: str) -> str:
        candidate = re.sub(r"\s+", " ", value or "").strip(" -.,")
        candidate = re.sub(
            r"^(?:branding company|software development company|web development|website development company|custom software development company|located in|based in|the team at|launched in \d{4},?|founded in \d{4},?|is a|is an)\s+",
            "",
            candidate,
            flags=re.IGNORECASE,
        ).strip(" -.,")
        candidate = re.sub(r"^(?:top|best)\s+", "", candidate, flags=re.IGNORECASE).strip(" -.,")
        candidate = re.sub(r"^[^A-Za-z0-9]+", "", candidate).strip(" -.,")
        if "," in candidate:
            parts = [part.strip() for part in candidate.split(",") if part.strip()]
            location_markers = {"india", "mumbai", "navi mumbai", "delhi", "bengaluru", "bangalore", "pune"}
            while len(parts) > 1 and parts[0].lower() in location_markers:
                parts.pop(0)
            if len(parts) > 1 and parts[0].lower() in location_markers:
                candidate = parts[-1]
            elif len(parts) > 2:
                candidate = parts[-1]
        bad_prefixes = ("Top ", "Best ", "Page ", "Located in ", "Based in ")
        for prefix in bad_prefixes:
            if candidate.startswith(prefix):
                return ""
        lowered = candidate.lower()
        if ProductionLeadPipeline._is_generic_company_title(candidate):
            return ""
        if any(marker in lowered for marker in (" companies", " rankings", " services provided", " read more", "official website", "community events")):
            return ""
        return candidate[:120] if len(candidate) >= 3 else ""

    @classmethod
    def _looks_like_valid_company_name(cls, value: str) -> bool:
        candidate = cls._clean_company_candidate(value)
        if not candidate:
            return False
        lowered = candidate.lower()
        bad_markers = (
            "launched in ",
            "founded in ",
            "chaos engineering",
            "what",
            "illinois",
            "welcome to",
            "event",
            "community",
            "legal entities",
            "not a bbb accredited business",
            "it's good company",
            "i am from mumbai",
            "restaurant",
            "canteen",
        )
        if lowered in {"what", "illinois"}:
            return False
        if any(marker in lowered for marker in bad_markers):
            return False
        return True

    @staticmethod
    def _is_directory_or_noise_url(url: str) -> bool:
        host = (urlparse(url or "").hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        noisy_domains = (
            "clutch.co",
            "glassdoor.co.in",
            "chamberofcommerce.com",
            "bbb.org",
            "linkedin.com",
            "facebook.com",
            "instagram.com",
            "twitter.com",
            "x.com",
            "crunchbase.com",
            "zoominfo.com",
            "goodfirms.co",
            "ambitionbox.com",
            "justdial.com",
            "tradeindia.com",
            "indiamart.com",
            "falconebiz.com",
            "tofler.in",
            "zaubacorp.com",
            "thecompanycheck.com",
            "cleartax.in",
            "tracxn.com",
            "exportersindia.com",
            "youtube.com",
            "youtu.be",
            "globalwiki.org",
            "neusourcestartup.com",
            "ace.atlassian.com",
        )
        if any(host == domain or host.endswith(f".{domain}") for domain in noisy_domains):
            return True
        path = (urlparse(url or "").path or "").lower()
        noisy_path_markers = (
            "/events/",
            "/event/",
            "/legal-entities/",
            "/company/",
            "/companies/",
            "/wiki/",
            "/channel/",
            "/watch",
        )
        return any(marker in path for marker in noisy_path_markers)

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
        if cls._is_directory_or_noise_url(url):
            return False
        combined = f"{title} {content} {url}".lower()
        bad_markers = (
            "event",
            "community",
            "legal entities",
            "company details",
            "company profile",
            "welcome to the new",
            "youtube",
            "wikipedia",
            "wiki",
            "exportersindia",
            "cleartax",
            "tracxn",
            "directory",
        )
        if any(marker in combined for marker in bad_markers):
            return False
        company_tokens = [token for token in re.findall(r"[a-z0-9]+", company_name.lower()) if len(token) > 2]
        if company_tokens and not any(token in combined for token in company_tokens[:2]):
            return False
        host = (urlparse(url or "").hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        host_company_match = any(token in host for token in company_tokens[:2]) if company_tokens else False
        if location and location.lower() not in combined and industry and industry.lower() not in combined and not host_company_match:
            return False
        industry_markers = cls._industry_markers(industry)
        good_markers = ("contact", "about", "services", "solutions", "technology", "software", "consulting")
        has_good_markers = any(marker in combined for marker in good_markers)
        if industry_markers and not any(marker in combined for marker in industry_markers) and not (host_company_match and has_good_markers):
            return False
        return has_good_markers or host_company_match or len(company_tokens) >= 2

    @staticmethod
    def _industry_markers(industry: str) -> List[str]:
        lowered = (industry or "").lower()
        if "it" in lowered or "software" in lowered or "technology" in lowered:
            return [
                "it services",
                "technology",
                "software",
                "digital",
                "web development",
                "app development",
                "cloud",
                "cybersecurity",
                "managed services",
                "consulting",
                "automation",
                "ai",
            ]
        return [token for token in re.findall(r"[a-z0-9]+", lowered) if len(token) > 2]

    @staticmethod
    def _email_pattern() -> re.Pattern[str]:
        return re.compile(r"[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}", re.IGNORECASE)

    @classmethod
    def _valid_email_matches(cls, text: str) -> List[str]:
        asset_extensions = {"avif", "bmp", "css", "gif", "ico", "jpeg", "jpg", "js", "png", "svg", "webp"}
        values: List[str] = []
        seen = set()
        for raw in cls._email_pattern().findall(text or ""):
            email = raw.strip(" .,:;()[]{}<>\"'").lower()
            local, _, domain = email.partition("@")
            extension = domain.rsplit(".", 1)[-1] if "." in domain else ""
            if not local or not domain or extension in asset_extensions:
                continue
            if domain in {"domain.com", "example.com", "example.net", "example.org", "email.com"}:
                continue
            if local in {"user", "username", "name", "yourname", "test"} and domain in {"domain.com", "yourdomain.com"}:
                continue
            if re.search(r"(?:^|[-_.])(?:logo|icon|sprite|image)(?:[-_.]|$)", local, re.IGNORECASE):
                continue
            if email not in seen:
                seen.add(email)
                values.append(email)
        return values

    @classmethod
    def _first_valid_email(cls, text: str) -> str:
        values = cls._valid_email_matches(text)
        return values[0] if values else ""

    @staticmethod
    def _phone_pattern() -> re.Pattern[str]:
        return re.compile(r"(?:\+?\d[\d\s().-]{7,}\d)")

    @staticmethod
    def _homepage_url(url: str) -> str:
        parsed = urlparse(url or "")
        if not parsed.scheme:
            parsed = urlparse(f"https://{str(url or '').lstrip('/')}")
        if not parsed.netloc:
            return url or ""
        return f"{parsed.scheme}://{parsed.netloc}/"

    @staticmethod
    def _first_match(pattern: re.Pattern[str], text: str) -> str:
        match = pattern.search(text or "")
        return match.group(0).strip() if match else ""

    @staticmethod
    def _first_non_empty(*values: Any) -> str:
        for value in values:
            cleaned = str(value or "").strip()
            if cleaned and cleaned not in {"—", "-", "None", "null"}:
                return cleaned
        return ""

    @classmethod
    def _extract_all_phones(cls, text: str) -> List[str]:
        summary = phone_summary_from_text(text or "", source_type="text")
        phones = [summary.get("contact_phone", "")]
        phones.extend(summary.get("alternate_phones") or [])
        return [phone for phone in phones if phone]

    @classmethod
    def _apply_phone_summary(cls, lead: Dict[str, Any], summary: Dict[str, Any]) -> Dict[str, Any]:
        primary = cls._first_non_empty(summary.get("contact_phone"))
        if not primary:
            return lead
        existing = cls._first_non_empty(lead.get("contact_phone"))
        current_confidence = int(lead.get("phone_confidence") or 0)
        new_confidence = int(summary.get("phone_confidence") or 0)
        alternates = list(lead.get("alternate_phones") or [])
        if existing and existing != primary and existing not in alternates:
            alternates.insert(0, existing)
        for phone in summary.get("alternate_phones") or []:
            if phone and phone != primary and phone not in alternates:
                alternates.append(phone)
        if not existing or new_confidence >= current_confidence:
            lead["contact_phone"] = primary
            lead["phone_confidence"] = new_confidence
            lead["phone_source"] = summary.get("phone_source", "")
            lead["phone_validation_status"] = summary.get("phone_validation_status", "")
            lead["phone_type"] = summary.get("phone_type", "")
            lead["phone_candidates"] = summary.get("phone_candidates", [])
        lead["alternate_phones"] = alternates[:4]
        return lead

    @staticmethod
    def _location_evidence_excerpt(*, text: str, location: str, max_length: int = 800) -> str:
        haystack = re.sub(r"\s+", " ", str(text or "")).strip()
        needle = re.sub(r"\s+", " ", str(location or "")).strip()
        if not haystack or not needle:
            return ""
        lowered = haystack.lower()
        terms = {needle.lower()}
        if needle.lower() == "alaska":
            terms.update({"ak", "anchorage", "fairbanks", "wasilla", "palmer", "juneau", "kenai", "ships to alaska", "serves alaska", "alaska delivery"})
        for term in terms:
            index = lowered.find(term)
            if index >= 0:
                start = max(0, index - 220)
                end = min(len(haystack), index + len(term) + 420)
                return haystack[start:end][:max_length]
        return ""

    @staticmethod
    def _first_linkedin_url(text: str) -> str:
        match = re.search(r"https?://(?:www\.)?linkedin\.com/[^\s)\"']+", text or "", re.IGNORECASE)
        return match.group(0).strip() if match else ""

    @classmethod
    def _extract_person_name_from_linkedin_result(cls, *, title: str, content: str) -> str:
        title_clean = re.sub(r"\s+", " ", str(title or "")).strip()
        content_clean = re.sub(r"\s+", " ", str(content or "")).strip()
        for candidate in (
            title_clean.split("|", 1)[0].split("-", 1)[0].strip(),
            content_clean.split("\n", 1)[0].strip(),
        ):
            if not candidate:
                continue
            if cls._looks_like_person_name(candidate):
                return candidate[:120]
        return ""

    @classmethod
    def _extract_person_title_from_linkedin_result(cls, *, title: str, content: str) -> str:
        title_clean = re.sub(r"\s+", " ", str(title or "")).strip()
        content_clean = re.sub(r"\s+", " ", str(content or "")).strip()
        for candidate in (
            title_clean.split("-", 1)[1].strip() if "-" in title_clean else "",
            cls._first_sentence(content_clean),
        ):
            if candidate and 4 <= len(candidate) <= 160:
                return candidate
        return ""

    @staticmethod
    def _first_sentence(text: str) -> str:
        cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
        parts = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)
        return parts[0].strip() if parts else cleaned

    @staticmethod
    def _looks_like_person_name(value: str) -> bool:
        cleaned = re.sub(r"\s+", " ", str(value or "")).strip(" -|,")
        if not cleaned or len(cleaned) < 5 or len(cleaned) > 80:
            return False
        if any(char.isdigit() for char in cleaned):
            return False
        words = [part for part in cleaned.split() if part]
        if len(words) < 2 or len(words) > 5:
            return False
        banned = {"linkedin", "founder", "director", "ceo", "owner", "contact", "email", "phone", "number"}
        lowered = {word.lower() for word in words}
        if lowered & banned:
            return False
        return all(word[:1].isalpha() for word in words)

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
        combined = f"{title} {content} {url}".lower()
        company_tokens = [token for token in re.findall(r"[a-z0-9]+", company_name.lower()) if len(token) > 2]
        host = (urlparse(url or "").hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        score = 0
        if any(token in host for token in company_tokens[:2]):
            score += 2
        if company_name.lower() in combined:
            score += 2
        if location and location.lower() in combined:
            score += 1
        if any(marker in combined for marker in cls._industry_markers(industry)):
            score += 1
        if any(marker in combined for marker in ("contact", "about", "services", "solutions", "technology", "software", "consulting")):
            score += 1
        return score

    @staticmethod
    def _fallback_value_proposition(*, company_name: str, text: str, industry: str, location: str) -> str:
        cleaned = re.sub(r"\s+", " ", text or "").strip()
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        for sentence in sentences:
            if company_name.lower() in sentence.lower() and 40 <= len(sentence) <= 220:
                return sentence.strip()
        return f"{company_name} is a verified {industry or 'B2B'} company matched from trusted directory signals around {location or 'the target market'}."
