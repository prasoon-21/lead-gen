from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urljoin, urlparse

import httpx

from core.services.lead_discovery_policy import EXCLUDED_LEAD_SOURCE_DOMAINS, is_excluded_lead_source_url
from core.services.lead_quality_service import score_lead
from core.services.lead_quality_gate import apply_quality_gate
from core.services.production_lead_pipeline import (
    BROWSER_VERIFICATION_HEADERS,
    ProductionLeadPipeline,
)
from core.services.velit.discovery import (
    VELIT_EXTRACT_DEPTH,
    VELIT_SEARCH_DEPTH,
    build_velit_shortfall_queries,
    build_velit_queries,
    is_noise_url,
    looks_like_velit_text,
    normalize_company_key,
    normalize_domain,
    normalize_velit_location,
)
from core.utils.lead_summary import build_plain_lead_summary
from core.utils.phone_quality import (
    build_phone_summary,
    extract_phone_candidates,
    phone_candidates_from_values,
    phone_summary_from_text,
    rank_phone_candidates,
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
            0,
            self._bounded_int_env("VELIT_LINKEDIN_ENRICHMENT_LIMIT", default=min(self.linkedin_enrichment_limit, 4), minimum=0, maximum=20),
        )
        self.official_site_resolution_limit = self._bounded_int_env(
            "VELIT_OFFICIAL_SITE_RESOLUTION_LIMIT",
            default=min(max(self.official_site_resolution_limit, 8), 12),
            minimum=0,
            maximum=40,
        )
        self.max_search_results_per_query = self._bounded_int_env(
            "VELIT_SEARCH_RESULTS_PER_QUERY",
            default=5,
            minimum=4,
            maximum=12,
        )
        self.discovery_query_limit = self._bounded_int_env(
            "VELIT_DISCOVERY_QUERY_LIMIT",
            default=8,
            minimum=1,
            maximum=8,
        )
        self.shortfall_max_passes = self._bounded_int_env(
            "VELIT_SHORTFALL_MAX_PASSES",
            default=3,
            minimum=0,
            maximum=3,
        )
        self.shortfall_extra_query_limit = self._bounded_int_env(
            "VELIT_SHORTFALL_MAX_EXTRA_QUERIES",
            default=12,
            minimum=0,
            maximum=30,
        )
        self.max_contact_scrapes = self._bounded_int_env(
            "VELIT_CONTACT_SCRAPE_LIMIT",
            default=24,
            minimum=8,
            maximum=80,
        )
        self.contact_search_limit = self._bounded_int_env(
            "VELIT_CONTACT_SEARCH_LIMIT",
            default=4,
            minimum=0,
            maximum=20,
        )
        self.contact_queries_per_company = self._bounded_int_env(
            "VELIT_CONTACT_QUERIES_PER_COMPANY",
            default=1,
            minimum=0,
            maximum=3,
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
        self._partial_result: Dict[str, Any] = {
            "leads": [],
            "steps": [],
            "metadata": {
                "pipeline": "velit_specialized_pipeline",
                "status": "not_started",
                "partial_result": True,
            },
        }

    async def run(
        self,
        *,
        industry: str,
        location: str,
        seed_query: str = "",
        target_count: int = 15,
    ) -> Dict[str, Any]:
        target_count = max(1, min(int(target_count or 15), 100))
        location = normalize_velit_location(location)
        phase_steps: List[Dict[str, Any]] = []
        self._store_partial_result(
            leads=[],
            steps=phase_steps,
            stage="started",
            metadata={"location": location, "target_count": target_count},
        )
        try:
            discovery_status = "completed"
            try:
                search_results = await asyncio.wait_for(
                    self._discover_targets(industry=industry, location=location, seed_query=seed_query),
                    timeout=self._phase_timeout_seconds("discovery", 60),
                )
            except asyncio.TimeoutError:
                search_results = []
                discovery_status = "timeout"
            phase_steps.append(
                {
                    "step": 1,
                    "stage": "velit_direct_discovery",
                    "type": "tavily_search",
                    "status": discovery_status,
                    "search_depth": VELIT_SEARCH_DEPTH,
                    "query_count": len(build_velit_queries(location=location, seed_query=seed_query, max_queries=self.discovery_query_limit)),
                    "result_count": len(search_results),
                }
            )
            self._store_emergency_partial_from_items(
                search_results,
                steps=phase_steps,
                stage="discovery_emergency_ready",
                industry=industry or "Upfitter",
                location=location,
                target_count=target_count,
            )

            resolution_status = "completed"
            try:
                corporate_targets = await asyncio.wait_for(
                    self._resolve_corporate_targets(
                        search_results,
                        industry=industry,
                        location=location,
                        limit=max(24, target_count * 2),
                    ),
                    timeout=self._phase_timeout_seconds("site_resolution", 120),
                )
            except asyncio.TimeoutError:
                corporate_targets = []
                resolution_status = "timeout"
            phase_steps.append(
                {
                    "step": 2,
                    "stage": "velit_site_resolution",
                    "type": "direct_site_resolution",
                    "status": resolution_status,
                    "input_count": len(search_results),
                    "result_count": len(corporate_targets),
                }
            )
            self._store_emergency_partial_from_items(
                corporate_targets or search_results,
                steps=phase_steps,
                stage="site_resolution_emergency_ready",
                industry=industry or "Upfitter",
                location=location,
                target_count=target_count,
            )

            verification_status = "completed"
            try:
                verified = await asyncio.wait_for(
                    self._verify_targets(corporate_targets, limit=max(24, target_count * 2)),
                    timeout=self._phase_timeout_seconds("site_verification", 120),
                )
            except asyncio.TimeoutError:
                verified = []
                verification_status = "timeout"
            phase_steps.append(
                {
                    "step": 3,
                    "stage": "velit_site_verification",
                    "type": "async_http_verification",
                    "status": verification_status,
                    "input_count": len(corporate_targets),
                    "result_count": len(verified),
                }
            )
            self._store_emergency_partial_from_items(
                verified or corporate_targets or search_results,
                steps=phase_steps,
                stage="site_verification_emergency_ready",
                industry=industry or "Upfitter",
                location=location,
                target_count=target_count,
            )

            extraction_status = "completed"
            try:
                extracted = await asyncio.wait_for(
                    self._extract_targets(verified),
                    timeout=self._phase_timeout_seconds("extraction", 180),
                )
            except asyncio.TimeoutError:
                extracted = []
                extraction_status = "timeout"
            preliminary_leads = self._build_backfill_leads_from_extracted(
                extracted,
                existing_leads=[],
                industry=industry or "Upfitter",
                location=location,
                target_count=target_count,
            )
            preliminary_leads = self._sanitize_outreach_leads(preliminary_leads, location=location, target_count=target_count)
            if preliminary_leads:
                self._store_partial_result(
                    leads=preliminary_leads,
                    steps=phase_steps,
                    stage="extraction_backfill_ready",
                    metadata={"extracted_targets": len(extracted), "extraction_status": extraction_status},
                )
            score_pool_size = min(
                max(target_count * 2, target_count + 12),
                max(target_count, len(extracted)),
                60,
            )
            leads = await self._score_extracted_targets(
                extracted,
                industry=industry or "Upfitter",
                location=location,
                target_count=score_pool_size,
            )
            leads.extend(
                self._build_backfill_leads_from_extracted(
                    extracted,
                    existing_leads=leads,
                    industry=industry or "Upfitter",
                    location=location,
                    target_count=score_pool_size,
                )
            )
            leads.sort(key=self._velit_rank_key, reverse=True)
            leads = leads[:score_pool_size]
            self._store_partial_result(
                leads=leads[:target_count],
                steps=phase_steps,
                stage="scored_leads_ready",
                metadata={"extracted_targets": len(extracted), "score_pool_size": score_pool_size},
            )
            leads = await self._enrich_leads_with_linkedin(
                leads,
                industry=industry or "Upfitter",
                location=location,
            )
            self._store_partial_result(
                leads=leads[:target_count],
                steps=phase_steps,
                stage="linkedin_enrichment_complete",
                metadata={"extracted_targets": len(extracted), "score_pool_size": score_pool_size},
            )
            leads = await self._recover_missing_contacts(
                leads,
                industry=industry or "Upfitter",
                location=location,
            )
            self._store_partial_result(
                leads=leads[:target_count],
                steps=phase_steps,
                stage="contact_recovery_complete",
                metadata={"extracted_targets": len(extracted), "score_pool_size": score_pool_size},
            )
            leads = self._sanitize_outreach_leads(leads, location=location)
            leads, quality_gate_stats = apply_quality_gate(leads, drop_rejected=True)
            leads = leads[:target_count]
            initial_lead_count = len(leads)
            initial_quality_gate_stats = quality_gate_stats
            phase_steps.append(
                {
                    "step": 4,
                    "stage": "velit_enrichment_scoring",
                    "type": "site_scraping_gemini_linkedin",
                    "status": extraction_status,
                    "extract_depth": VELIT_EXTRACT_DEPTH,
                    "input_count": len(verified),
                    "lead_count": len(leads),
                    "model": self.gemini_model,
                }
            )
            self._store_partial_result(
                leads=leads,
                steps=phase_steps,
                stage="quality_gate_complete",
                metadata={
                    "extracted_targets": len(extracted),
                    "score_pool_size": score_pool_size,
                    "quality_gate": quality_gate_stats,
                },
            )
            shortfall_stats = self._empty_shortfall_stats(initial_lead_count, target_count=target_count)
            shortfall_steps: List[Dict[str, Any]] = []
            if len(leads) < target_count:
                leads, shortfall_stats, quality_gate_stats, shortfall_steps = await self._run_shortfall_passes(
                    leads=leads,
                    industry=industry or "Upfitter",
                    location=location,
                    seed_query=seed_query,
                    target_count=target_count,
                    seen_urls=self._urls_from_items(search_results, corporate_targets, verified),
                    seen_domains=self._domains_from_items(corporate_targets, verified) | self._domains_from_leads(leads),
                    initial_quality_gate_stats=quality_gate_stats,
                )
                phase_steps.extend(shortfall_steps)

            returned_emergency_partial = False
            if not leads and self._partial_result.get("leads"):
                leads = [
                    dict(lead)
                    for lead in (self._partial_result.get("leads") or [])[:target_count]
                    if isinstance(lead, dict)
                ]
                quality_gate_stats = dict(quality_gate_stats or {})
                quality_gate_stats["returned_emergency_partial"] = True
                returned_emergency_partial = True

            result = {
                "leads": leads,
                "steps": phase_steps,
                "metadata": {
                    "pipeline": "velit_specialized_pipeline",
                    "search_depth": VELIT_SEARCH_DEPTH,
                    "extract_depth": VELIT_EXTRACT_DEPTH,
                    "discovered_targets": len(search_results) + int(shortfall_stats.get("extra_discovered_targets") or 0),
                    "corporate_targets": len(corporate_targets) + int(shortfall_stats.get("extra_corporate_targets") or 0),
                    "verified_targets": len(verified) + int(shortfall_stats.get("extra_verified_targets") or 0),
                    "scored_targets": len(leads),
                    "initial_lead_count": initial_lead_count,
                    "quality_gate": quality_gate_stats,
                    "initial_quality_gate": initial_quality_gate_stats,
                    "returned_emergency_partial": returned_emergency_partial,
                    "shortfall_passes_used": shortfall_stats.get("passes_used", 0),
                    "shortfall_extra_discovered_targets": shortfall_stats.get("extra_discovered_targets", 0),
                    "shortfall_stop_reason": shortfall_stats.get("stop_reason", "no_shortfall"),
                    "shortfall": shortfall_stats,
                    "linkedin_enriched_targets": sum(
                        1 for lead in leads if lead.get("contact_person_name") or lead.get("founder_name")
                    ),
                    "email_backed_targets": sum(1 for lead in leads if lead.get("contact_email")),
                    "contact_scrape_limit": self.max_contact_scrapes,
                    "contact_search_limit": self.contact_search_limit,
                    "contact_queries_per_company": self.contact_queries_per_company,
                    "discovery_query_limit": self.discovery_query_limit,
                    "shortfall_max_passes": self.shortfall_max_passes,
                    "shortfall_extra_query_limit": self.shortfall_extra_query_limit,
                    "gemini_max_calls_per_run": self.gemini_max_calls_per_run,
                },
            }
            self._store_partial_result(
                leads=leads,
                steps=phase_steps,
                stage="complete",
                metadata={**result["metadata"], "status": "completed", "partial_result": False},
            )
            return result
        except Exception as exc:
            self.logger.error("velit_pipeline_run_failed", extra={"payload": str(exc), "stacktrace": True})
            partial = self.get_partial_result(
                error=f"Internal Server Error during Velit lead generation: {exc}",
                status="failed_partial" if self._partial_result.get("leads") else "failed",
            )
            if partial.get("leads"):
                partial["metadata"]["error_type"] = type(exc).__name__
                return partial
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

    def _store_emergency_partial_from_items(
        self,
        items: Iterable[Any],
        *,
        steps: List[Dict[str, Any]],
        stage: str,
        industry: str,
        location: str,
        target_count: int,
    ) -> None:
        item_list = list(items or [])
        leads = self._build_emergency_leads_from_discovery(
            item_list,
            industry=industry,
            location=location,
            target_count=target_count,
            source_stage=stage,
        )
        if leads:
            self._store_partial_result(
                leads=leads,
                steps=steps,
                stage=stage,
                metadata={"emergency_partial": True, "emergency_source_count": len(item_list)},
            )

    def _build_emergency_leads_from_discovery(
        self,
        items: Iterable[Any],
        *,
        industry: str,
        location: str,
        target_count: int,
        source_stage: str,
    ) -> List[Dict[str, Any]]:
        leads: List[Dict[str, Any]] = []
        seen_domains: set[str] = set()
        for raw in items or []:
            item = self._emergency_item_to_dict(raw)
            url = self._first_non_empty(item.get("url"), item.get("company_website"), item.get("contact_page"))
            if not url or is_noise_url(url):
                continue
            domain = normalize_domain(url)
            if domain and domain in seen_domains:
                continue
            title = self._first_non_empty(item.get("title"), item.get("company_name"))
            content = self._first_non_empty(item.get("content"), item.get("raw_content"), item.get("snippet"))
            company_name = self._clean_company_candidate(
                self._resolve_company_name(str(item.get("company_name") or ""), title=title, url=url)
            )
            if not self._looks_like_valid_company_name(company_name):
                continue
            signals = item.get("contact_signals") if isinstance(item.get("contact_signals"), dict) else {}
            signal_text = json.dumps(signals, default=str) if signals else ""
            combined_text = "\n".join(part for part in (title, content, signal_text) if part)
            if not looks_like_velit_text(f"{combined_text} {company_name} {url}"):
                continue
            email = self._pick_best_email(
                existing=self._first_valid_email(combined_text),
                candidates=list(signals.get("emails") or []),
                website_domain=domain,
            )
            phone_summary = phone_summary_from_text(combined_text, source_type="emergency_partial", source_url=url)
            lead = {
                "company_name": company_name,
                "company_website": self._homepage_url(url),
                "industry": industry or "Upfitter",
                "location": location or "",
                "location_evidence": self._location_evidence_excerpt(text=combined_text, location=location),
                "value_proposition": self._fallback_value_proposition(
                    company_name=company_name,
                    text=combined_text,
                    industry=industry,
                    location=location,
                ),
                "lead_summary": "",
                "contact_person_name": "",
                "founder_name": "",
                "contact_person_title": "",
                "contact_email": email,
                "contact_phone": self._first_non_empty(phone_summary.get("contact_phone")),
                "phone_validation_status": self._first_non_empty(phone_summary.get("phone_validation_status")),
                "phone_type": self._first_non_empty(phone_summary.get("phone_type")),
                "company_linkedin_url": self._first_non_empty(*(signals.get("linkedin") or signals.get("linkedin_urls") or [])),
                "linkedin_url": "",
                "contact_page": url,
                "source": "emergency_partial",
                "source_details": [
                    {
                        "stage": source_stage,
                        "type": "emergency_partial_lead",
                        "provider": "local_rules",
                        "url": url,
                        "title": title,
                    }
                ],
            }
            quality = score_lead(lead)
            lead.update({key: value for key, value in quality.items() if key != "quality_score"})
            lead["quality_score"] = max(0, min(int(quality.get("quality_score") or 0), 100))
            lead["lead_summary"] = build_plain_lead_summary(lead)
            if domain:
                seen_domains.add(domain)
            leads.append(lead)
            if len(leads) >= max(1, target_count):
                break
        leads = self._sanitize_outreach_leads(leads, location=location, target_count=target_count)
        return leads

    def _emergency_item_to_dict(self, item: Any) -> Dict[str, Any]:
        if isinstance(item, dict):
            return dict(item)
        return {
            "url": getattr(item, "url", ""),
            "title": getattr(item, "title", ""),
            "content": getattr(item, "content", ""),
            "company_name": getattr(item, "company_name", ""),
            "directory_url": getattr(item, "directory_url", ""),
            "contact_signals": getattr(item, "contact_signals", {}),
        }

    def _store_partial_result(
        self,
        *,
        leads: List[Dict[str, Any]],
        steps: List[Dict[str, Any]],
        stage: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        safe_leads = [dict(lead) for lead in (leads or []) if isinstance(lead, dict)]
        safe_steps = [dict(step) for step in (steps or []) if isinstance(step, dict)]
        safe_metadata = {
            "pipeline": "velit_specialized_pipeline",
            "status": "partial",
            "partial_result": True,
            "partial_stage": stage,
            "partial_lead_count": len(safe_leads),
        }
        safe_metadata.update(metadata or {})
        self._partial_result = {
            "leads": safe_leads,
            "steps": safe_steps,
            "metadata": safe_metadata,
        }

    def get_partial_result(self, *, error: str = "", status: str = "timeout") -> Dict[str, Any]:
        partial = self._partial_result or {}
        metadata = dict(partial.get("metadata") or {})
        metadata.update(
            {
                "pipeline": metadata.get("pipeline", "velit_specialized_pipeline"),
                "status": status,
                "partial_result": True,
                "partial_lead_count": len(partial.get("leads") or []),
            }
        )
        result = {
            "leads": [dict(lead) for lead in (partial.get("leads") or []) if isinstance(lead, dict)],
            "steps": [dict(step) for step in (partial.get("steps") or []) if isinstance(step, dict)],
            "metadata": metadata,
        }
        if error:
            result["error"] = error
        return result

    @staticmethod
    def _empty_shortfall_stats(initial_lead_count: int, *, target_count: int = 0) -> Dict[str, Any]:
        return {
            "needed": bool(target_count and initial_lead_count < target_count),
            "initial_lead_count": initial_lead_count,
            "passes_used": 0,
            "extra_queries_used": 0,
            "extra_discovered_targets": 0,
            "extra_corporate_targets": 0,
            "extra_verified_targets": 0,
            "final_lead_count": initial_lead_count,
            "stop_reason": "no_shortfall",
            "passes": [],
        }

    async def _run_shortfall_passes(
        self,
        *,
        leads: List[Dict[str, Any]],
        industry: str,
        location: str,
        seed_query: str,
        target_count: int,
        seen_urls: set[str],
        seen_domains: set[str],
        initial_quality_gate_stats: Dict[str, Any],
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
        current_leads = list(leads or [])
        current_quality_gate_stats = dict(initial_quality_gate_stats or {})
        stats = self._empty_shortfall_stats(len(current_leads), target_count=target_count)
        stats["needed"] = True
        stats["stop_reason"] = ""
        steps: List[Dict[str, Any]] = []
        query_budget = max(0, int(self.shortfall_extra_query_limit or 0))

        if self.shortfall_max_passes <= 0:
            stats["stop_reason"] = "max_passes_reached"
            stats["final_lead_count"] = len(current_leads)
            return current_leads, stats, current_quality_gate_stats, steps

        for pass_index in range(1, self.shortfall_max_passes + 1):
            if len(current_leads) >= target_count:
                stats["stop_reason"] = "target_reached"
                break
            if query_budget <= 0:
                stats["stop_reason"] = "query_budget_exhausted"
                break

            query_limit = min(self.discovery_query_limit, query_budget)
            queries = build_velit_shortfall_queries(
                location=location,
                pass_index=pass_index,
                seed_query=seed_query,
                max_queries=query_limit,
            )
            if not queries:
                stats["stop_reason"] = "query_budget_exhausted"
                break

            query_budget -= len(queries)
            stats["extra_queries_used"] += len(queries)
            pass_info: Dict[str, Any] = {
                "pass": pass_index,
                "query_count": len(queries),
                "discovered_targets": 0,
                "corporate_targets": 0,
                "verified_targets": 0,
                "candidate_leads": 0,
                "added_leads": 0,
                "lead_count_after": len(current_leads),
                "stop_reason": "",
            }

            discovered = await self._discover_targets(
                industry=industry,
                location=location,
                seed_query=seed_query,
                queries=queries,
                seen_urls=seen_urls,
            )
            pass_info["discovered_targets"] = len(discovered)
            stats["extra_discovered_targets"] += len(discovered)
            if not discovered:
                pass_info["stop_reason"] = "no_new_targets"
                stats["passes_used"] += 1
                stats["passes"].append(pass_info)
                steps.append(self._shortfall_step(pass_info))
                stats["stop_reason"] = "no_new_targets"
                break

            corporate_targets = await self._resolve_corporate_targets(
                discovered,
                industry=industry,
                location=location,
                limit=max(24, (target_count - len(current_leads)) * 2),
                skip_domains=seen_domains,
            )
            pass_info["corporate_targets"] = len(corporate_targets)
            stats["extra_corporate_targets"] += len(corporate_targets)
            seen_domains.update(self._domains_from_items(corporate_targets))
            seen_urls.update(self._urls_from_items(corporate_targets))
            if not corporate_targets:
                pass_info["stop_reason"] = "no_new_targets"
                stats["passes_used"] += 1
                stats["passes"].append(pass_info)
                steps.append(self._shortfall_step(pass_info))
                stats["stop_reason"] = "no_new_targets"
                break

            verified = await self._verify_targets(corporate_targets, limit=max(24, (target_count - len(current_leads)) * 2))
            pass_info["verified_targets"] = len(verified)
            stats["extra_verified_targets"] += len(verified)
            seen_domains.update(self._domains_from_items(verified))
            seen_urls.update(self._urls_from_items(verified))
            if not verified:
                pass_info["stop_reason"] = "no_new_verified_sites"
                stats["passes_used"] += 1
                stats["passes"].append(pass_info)
                steps.append(self._shortfall_step(pass_info))
                stats["stop_reason"] = "no_new_verified_sites"
                break

            extracted = await self._extract_targets(verified)
            score_pool_size = min(
                max((target_count - len(current_leads)) * 2, target_count + 12),
                max(target_count, len(extracted)),
                60,
            )
            pass_leads = await self._score_extracted_targets(
                extracted,
                industry=industry,
                location=location,
                target_count=score_pool_size,
            )
            pass_leads.extend(
                self._build_backfill_leads_from_extracted(
                    extracted,
                    existing_leads=current_leads + pass_leads,
                    industry=industry,
                    location=location,
                    target_count=score_pool_size,
                )
            )
            pass_leads.sort(key=self._velit_rank_key, reverse=True)
            pass_leads = pass_leads[:score_pool_size]
            pass_leads = await self._enrich_leads_with_linkedin(
                pass_leads,
                industry=industry,
                location=location,
            )
            pass_leads = await self._recover_missing_contacts(
                pass_leads,
                industry=industry,
                location=location,
            )
            pass_leads = self._sanitize_outreach_leads(pass_leads, location=location)

            before_count = len(current_leads)
            current_leads, current_quality_gate_stats = apply_quality_gate(current_leads + pass_leads, drop_rejected=True)
            current_leads = current_leads[:target_count]
            after_count = len(current_leads)
            pass_info["candidate_leads"] = len(pass_leads)
            pass_info["added_leads"] = max(0, after_count - before_count)
            pass_info["lead_count_after"] = after_count
            pass_info["quality_gate"] = current_quality_gate_stats
            if after_count >= target_count:
                pass_info["stop_reason"] = "target_reached"
                stats["stop_reason"] = "target_reached"
            elif after_count <= before_count:
                pass_info["stop_reason"] = "no_new_leads"
                stats["stop_reason"] = "no_new_leads"
            stats["passes_used"] += 1
            stats["passes"].append(pass_info)
            steps.append(self._shortfall_step(pass_info))

            if stats["stop_reason"] in {"target_reached", "no_new_leads"}:
                break

        if not stats["stop_reason"]:
            stats["stop_reason"] = "target_reached" if len(current_leads) >= target_count else "max_passes_reached"
        stats["final_lead_count"] = len(current_leads)
        return current_leads, stats, current_quality_gate_stats, steps

    @staticmethod
    def _shortfall_step(pass_info: Dict[str, Any]) -> Dict[str, Any]:
        pass_index = int(pass_info.get("pass") or 0)
        return {
            "step": 4 + pass_index,
            "stage": "velit_shortfall_discovery",
            "type": "extra_tavily_search",
            "query_count": pass_info.get("query_count", 0),
            "discovered_targets": pass_info.get("discovered_targets", 0),
            "corporate_targets": pass_info.get("corporate_targets", 0),
            "verified_targets": pass_info.get("verified_targets", 0),
            "candidate_leads": pass_info.get("candidate_leads", 0),
            "added_leads": pass_info.get("added_leads", 0),
            "lead_count_after": pass_info.get("lead_count_after", 0),
            "stop_reason": pass_info.get("stop_reason", ""),
        }

    @staticmethod
    def _urls_from_items(*groups: Iterable[Any]) -> set[str]:
        urls: set[str] = set()
        for group in groups:
            for item in group or []:
                if isinstance(item, dict):
                    value = item.get("url") or item.get("company_website") or item.get("contact_page") or item.get("directory_url")
                else:
                    value = getattr(item, "url", "")
                cleaned = str(value or "").strip()
                if cleaned:
                    urls.add(cleaned)
        return urls

    @staticmethod
    def _domains_from_items(*groups: Iterable[Any]) -> set[str]:
        domains: set[str] = set()
        for url in VelitLeadPipeline._urls_from_items(*groups):
            domain = normalize_domain(url)
            if domain:
                domains.add(domain)
        return domains

    @staticmethod
    def _domains_from_leads(leads: Iterable[Dict[str, Any]]) -> set[str]:
        domains: set[str] = set()
        for lead in leads or []:
            domain = normalize_domain(str(lead.get("company_website") or lead.get("website_url") or lead.get("contact_page") or ""))
            if domain:
                domains.add(domain)
        return domains

    async def _discover_targets(
        self,
        *,
        industry: str,
        location: str,
        seed_query: str,
        queries: Optional[List[str]] = None,
        seen_urls: Optional[set[str]] = None,
    ) -> List[Dict[str, Any]]:
        if queries is None:
            queries = build_velit_queries(location=location, seed_query=seed_query, max_queries=self.discovery_query_limit)
        seen_urls = seen_urls if seen_urls is not None else set()
        results: List[Dict[str, Any]] = []
        for query in queries:
            try:
                data = await self.tavily.search(
                    {
                        "query": query,
                        "search_depth": VELIT_SEARCH_DEPTH,
                        "max_results": self.max_search_results_per_query,
                        "exclude_domains": list(EXCLUDED_LEAD_SOURCE_DOMAINS),
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
                if is_excluded_lead_source_url(url):
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
        skip_domains: Optional[set[str]] = None,
        skip_company_keys: Optional[set[str]] = None,
    ) -> List[Dict[str, Any]]:
        targets: List[Dict[str, Any]] = []
        seen_companies = set(skip_company_keys or set())
        seen_domains = set(skip_domains or set())
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
                            "exclude_domains": list(EXCLUDED_LEAD_SOURCE_DOMAINS),
                            "include_raw_content": False,
                        }
                    ),
                    timeout=self.official_site_timeout_seconds,
                )
            except Exception:
                continue
            for item in data.get("results") or []:
                url = str(item.get("url") or "").strip()
                if not url or is_excluded_lead_source_url(url) or is_noise_url(url):
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
        gemini_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        lead = await super()._score_one(
            item,
            industry=industry or "Upfitter",
            location=location,
            use_gemini=use_gemini,
            gemini_data=gemini_data,
        )
        if not lead:
            return None
        signals = item.get("contact_signals") if isinstance(item.get("contact_signals"), dict) else {}
        if signals:
            lead["official_site_emails"] = list(signals.get("emails") or [])
            lead["official_site_phones"] = list(signals.get("phones") or [])
            lead["official_site_phone_candidates"] = list(signals.get("phone_candidates") or [])
            lead["contact_email"] = self._pick_best_official_email(
                existing=lead.get("contact_email"),
                candidates=signals.get("emails") or [],
                website_domain=normalize_domain(lead.get("company_website") or ""),
            )
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

    def _build_backfill_leads_from_extracted(
        self,
        items: List[Dict[str, Any]],
        *,
        existing_leads: List[Dict[str, Any]],
        industry: str,
        location: str,
        target_count: int,
    ) -> List[Dict[str, Any]]:
        """Create grounded fallback leads from extracted site text/contact signals.

        Gemini can occasionally fail JSON parsing or skip valid thin-market sites.
        This path never invents fields; it only uses the verified URL, extracted text,
        and direct site scrape signals already gathered by the pipeline.
        """
        existing_domains = {
            normalize_domain(str(lead.get("company_website") or ""))
            for lead in existing_leads or []
            if normalize_domain(str(lead.get("company_website") or ""))
        }
        backfill: List[Dict[str, Any]] = []
        for item in items or []:
            if len(backfill) + len(existing_leads) >= max(target_count, 1):
                break
            url = str(item.get("url") or "").strip()
            domain = normalize_domain(url)
            if not url or not domain or domain in existing_domains:
                continue
            title = str(item.get("title") or "").strip()
            text = self._clean_markdown(item.get("content") or "")
            contact_signals = item.get("contact_signals") if isinstance(item.get("contact_signals"), dict) else {}
            if not self._backfill_candidate_is_usable(
                url=url,
                title=title,
                text=text,
                industry=industry,
                location=location,
                contact_signals=contact_signals,
            ):
                continue
            company_name = self._clean_company_candidate(
                self._resolve_company_name(str(item.get("company_name") or "").strip(), title=title, url=url)
            )
            if not self._looks_like_valid_company_name(company_name):
                continue

            email = self._pick_best_email(
                existing=self._first_valid_email(text),
                candidates=[str(value) for value in contact_signals.get("emails") or []],
                website_domain=domain,
            )
            phone_summary = phone_summary_from_text(text, source_type="tavily_extract", source_url=url)
            signal_phone_summary = build_phone_summary(
                phone_candidates_from_values(
                    list(contact_signals.get("phones") or []) + list(contact_signals.get("phone_candidates") or []),
                    source_type="official_site_scrape",
                    source_url=url,
                )
            )
            if signal_phone_summary.get("contact_phone"):
                phone_summary = signal_phone_summary

            lead = {
                "company_name": company_name,
                "company_website": self._homepage_url(url),
                "industry": industry or "Upfitter",
                "location": location or "",
                "location_evidence": self._location_evidence_excerpt(text=f"{title}\n{text}", location=location),
                "value_proposition": self._fallback_value_proposition(
                    company_name=company_name,
                    text=text,
                    industry=industry,
                    location=location,
                ),
                "lead_summary": "",
                "company_size": "",
                "contact_person_name": "",
                "founder_name": "",
                "contact_person_title": "",
                "contact_email": email,
                "contact_phone": phone_summary.get("contact_phone", ""),
                "alternate_phones": phone_summary.get("alternate_phones", []),
                "phone_confidence": phone_summary.get("phone_confidence", 0),
                "phone_source": phone_summary.get("phone_source", ""),
                "phone_validation_status": phone_summary.get("phone_validation_status", ""),
                "phone_type": phone_summary.get("phone_type", ""),
                "phone_candidates": phone_summary.get("phone_candidates", []),
                "linkedin_url": self._first_linkedin_url(text),
                "company_linkedin_url": self._first_non_empty(*(contact_signals.get("linkedin") or [])),
                "instagram_url": self._first_non_empty(*(contact_signals.get("instagram") or [])),
                "facebook_url": self._first_non_empty(*(contact_signals.get("facebook") or [])),
                "youtube_url": self._first_non_empty(*(contact_signals.get("youtube") or [])),
                "contact_page": url,
                "source": "official_site_backfill",
                "source_details": [
                    {
                        "stage": "velit_backfill",
                        "type": "verified_extracted_site",
                        "provider": "local_rules",
                        "url": url,
                        "title": title,
                        "value": "Backfilled from verified website text/contact scrape because Gemini did not return this row.",
                    }
                ],
                "confidence": "medium",
                "qualification_score": max(4, self._heuristic_score(text=text, industry=industry, location=location)),
                "quality_score": 0,
                "usable": True,
                "rejected": False,
                "verification_status": "valid",
                "verification_message": "Verified with browser-like headers before extraction.",
                "notes": "",
                "quality_warnings": ["backfilled_without_gemini_pick"],
            }
            lead["lead_summary"] = build_plain_lead_summary(lead)
            quality = score_lead(lead)
            lead.update({key: value for key, value in quality.items() if key != "quality_score"})
            lead["quality_score"] = max(int(lead.get("quality_score") or 0), int(quality.get("quality_score") or 0))
            lead["quality_score"] = max(0, min(int(lead["quality_score"]), 100))
            backfill.append(lead)
            existing_domains.add(domain)
        return backfill

    @classmethod
    def _backfill_candidate_is_usable(
        cls,
        *,
        url: str,
        title: str,
        text: str,
        industry: str,
        location: str,
        contact_signals: Dict[str, Any] | None = None,
    ) -> bool:
        combined = f"{title} {text} {url}".lower()
        signals = contact_signals if isinstance(contact_signals, dict) else {}
        host = (urlparse(url or "").hostname or "").lower()
        if host.startswith("blog.") or ".blog." in host:
            return False
        if is_noise_url(url):
            return False
        if any(marker in combined for marker in ("travel advice", "destination guide", "rv rental marketplace", "classified ads")):
            return False
        if not looks_like_velit_text(combined):
            return False
        has_contact_signal = (
            bool(cls._valid_email_matches(combined))
            or bool(extract_phone_candidates(combined, source_type="backfill", source_url=url))
            or bool(signals.get("emails"))
            or bool(signals.get("phones"))
            or bool(signals.get("phone_candidates"))
        )
        has_social_signal = any(marker in combined for marker in ("linkedin.com/", "instagram.com/", "facebook.com/", "youtube.com/")) or any(
            signals.get(key) for key in ("linkedin", "instagram", "facebook", "youtube")
        )
        return has_contact_signal or has_social_signal

    def _sanitize_outreach_leads(
        self,
        leads: List[Dict[str, Any]],
        *,
        location: str = "",
        target_count: int = 0,
    ) -> List[Dict[str, Any]]:
        sanitized: List[Dict[str, Any]] = []
        for lead in leads or []:
            if not isinstance(lead, dict):
                continue
            item = self._sanitize_outreach_lead(dict(lead))
            if self._is_disallowed_company_name(item.get("company_name")):
                continue
            if not self._is_location_relevant(item, location=location):
                continue
            quality = score_lead(item)
            item.update({key: value for key, value in quality.items() if key != "quality_score"})
            item["quality_score"] = max(int(item.get("quality_score") or 0), int(quality.get("quality_score") or 0))
            item["quality_score"] = max(0, min(int(item.get("quality_score") or 0), 100))
            item["lead_summary"] = build_plain_lead_summary(item)
            sanitized.append(item)
        sanitized.sort(key=self._velit_rank_key, reverse=True)
        if target_count > 0:
            return sanitized[:target_count]
        return sanitized

    def _sanitize_outreach_lead(self, item: Dict[str, Any]) -> Dict[str, Any]:
        notes: List[str] = []
        website = self._first_non_empty(item.get("company_website"), item.get("contact_page"))
        domain = normalize_domain(website)

        resolved_name = self._company_name_from_url(website) if website else ""
        current_name = self._first_non_empty(item.get("company_name"))
        if resolved_name and self._should_replace_company_name(current_name, resolved_name):
            item["company_name"] = resolved_name
            notes.append("company_name_corrected_from_official_domain")

        self._sanitize_linkedin_fields(item, notes=notes)
        self._prefer_official_site_contact_fields(item, domain=domain, notes=notes)
        self._sanitize_email_field(item, domain=domain, notes=notes)

        if notes:
            existing_notes = item.get("outreach_safety_notes")
            if isinstance(existing_notes, list):
                merged_notes = list(existing_notes)
            elif existing_notes:
                merged_notes = [str(existing_notes)]
            else:
                merged_notes = []
            for note in notes:
                if note not in merged_notes:
                    merged_notes.append(note)
            item["outreach_safety_notes"] = merged_notes
            item.setdefault("source_details", []).append(
                {
                    "stage": "velit_outreach_safety",
                    "type": "final_contact_sanitization",
                    "provider": "local_rules",
                    "url": website,
                    "value": ", ".join(notes),
                }
            )

        return item

    @staticmethod
    def _is_disallowed_company_name(value: Any) -> bool:
        lowered = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()
        return lowered in {
            "instagram",
            "facebook",
            "youtube",
            "linkedin",
            "twitter",
            "x",
            "homepage",
            "home",
            "contact",
        }

    @classmethod
    def _should_replace_company_name(cls, current_name: str, resolved_name: str) -> bool:
        current = cls._clean_company_candidate(current_name)
        resolved = cls._clean_company_candidate(resolved_name)
        if not resolved or resolved.lower() == "unknown company":
            return False
        if not current:
            return True

        lowered = re.sub(r"[^a-z0-9]+", " ", current.lower()).strip()
        noisy_exact = {
            "instagram",
            "facebook",
            "youtube",
            "linkedin",
            "homepage",
            "home page",
            "home",
            "contact",
            "contact us",
            "a leading commercial van upfitter",
            "custom camper van conversions",
            "camper van conversions",
            "custom van conversions",
        }
        if lowered in noisy_exact or cls._is_generic_company_title(current):
            return True
        if re.match(r"^\d+\+?\s+years?\s+experience\b", lowered):
            return True

        current_tokens = set(cls._company_name_tokens(current))
        resolved_tokens = set(cls._company_name_tokens(resolved))
        meaningful_current = current_tokens - cls._generic_company_words()
        meaningful_resolved = resolved_tokens - cls._generic_company_words()
        state_tokens = set()
        for state_name in cls._US_STATE_CODES_BY_NAME:
            state_tokens.update(cls._company_name_tokens(state_name))
        has_location_stuffing = bool(current_tokens & state_tokens)
        if has_location_stuffing and meaningful_resolved and meaningful_resolved <= meaningful_current:
            if any(
                marker in lowered
                for marker in (
                    "camper van",
                    "campervan",
                    "van conversion",
                    "camper conversion",
                    "custom conversion",
                    "van upfit",
                )
            ):
                return True
        if meaningful_resolved and not (meaningful_current & meaningful_resolved):
            generic_markers = (
                "van conversion",
                "van conversions",
                "campervan",
                "camper van",
                "commercial van",
                "new york campervan",
                "van upfitter",
                "truck upfitter",
                "rv builder",
            )
            if any(marker in lowered for marker in generic_markers):
                return True
        return False

    @staticmethod
    def _generic_company_words() -> set[str]:
        return {
            "a",
            "an",
            "and",
            "the",
            "company",
            "custom",
            "commercial",
            "conversion",
            "conversions",
            "camper",
            "rv",
            "van",
            "vans",
            "vehicle",
            "vehicles",
            "upfitter",
            "upfitters",
            "builder",
            "builders",
            "truck",
            "trucks",
            "trailer",
            "trailers",
            "service",
            "services",
            "solutions",
            "interiors",
        }

    @classmethod
    def _is_location_relevant(cls, item: Dict[str, Any], *, location: str) -> bool:
        location_text = " ".join(str(location or "").lower().split()).strip()
        if not location_text:
            return True
        allowed_terms = cls._allowed_location_terms(location_text)
        allowed_states = cls._allowed_state_names(location_text)
        evidence = cls._location_evidence_text(item)
        compact_evidence = re.sub(r"[^a-z0-9]+", " ", evidence).strip()

        area_code = cls._phone_area_code(item.get("contact_phone"))
        allowed_area_codes = cls._allowed_area_codes(location_text)
        if area_code and allowed_area_codes and area_code not in allowed_area_codes and not cls._is_toll_free_area_code(area_code):
            return False
        if area_code and allowed_area_codes and (area_code in allowed_area_codes or cls._is_toll_free_area_code(area_code)):
            return True

        if any(cls._contains_location_term(compact_evidence, term) for term in allowed_terms):
            return True

        found_states = cls._states_found_in_text(compact_evidence)
        disallowed_states = found_states - allowed_states
        if disallowed_states:
            return False

        # For state/city-specific searches, do not let generated summaries alone
        # make a company look local. If the page has no conflicting state and
        # the official site is clearly in-market with a contact channel, keep it
        # as reviewable rather than dropping sparse-market results to zero.
        if allowed_area_codes or allowed_states:
            if cls._has_reviewable_location_fallback(item):
                fallback_notes = item.setdefault("outreach_safety_notes", [])
                if isinstance(fallback_notes, list) and "location_requires_manual_review" not in fallback_notes:
                    fallback_notes.append("location_requires_manual_review")
                return True
            return False
        return True

    @classmethod
    def _has_reviewable_location_fallback(cls, item: Dict[str, Any]) -> bool:
        website = cls._first_non_empty(item.get("company_website"), item.get("contact_page"))
        if is_noise_url(website):
            return False
        has_contact = bool(
            cls._first_non_empty(
                item.get("contact_email"),
                item.get("contact_phone"),
                item.get("company_linkedin_url"),
                item.get("linkedin_url"),
            )
        )
        if not has_contact:
            return False
        text = " ".join(
            str(item.get(key) or "").lower()
            for key in ("industry", "lead_summary", "value_proposition", "notes", "company_name", "company_website")
        )
        return looks_like_velit_text(text) or any(
            marker in text
            for marker in ("commercial vehicle", "fleet", "truck body", "work truck", "vehicle upfit")
        )

    @classmethod
    def _location_evidence_text(cls, item: Dict[str, Any]) -> str:
        chunks = [
            item.get("company_name"),
            item.get("company_website"),
            item.get("contact_page"),
            item.get("city"),
            item.get("state"),
            item.get("location_evidence"),
        ]
        source_details = item.get("source_details") if isinstance(item.get("source_details"), list) else []
        for detail in source_details:
            if not isinstance(detail, dict):
                continue
            chunks.extend(
                str(detail.get(key) or "")
                for key in ("url", "title")
                if str(detail.get(key) or "").strip()
            )
        return " ".join(str(chunk or "").lower() for chunk in chunks)

    @classmethod
    def _allowed_location_terms(cls, location_text: str) -> set[str]:
        terms = {location_text}
        location_tokens = [token for token in re.findall(r"[a-z0-9]+", location_text) if len(token) > 2]
        if len(location_tokens) == 1:
            terms.update(location_tokens)
        if location_text in {"new york", "new york city", "nyc", "ny"}:
            terms.update(
                {
                    "new york",
                    "nyc",
                    "ny",
                    "brooklyn",
                    "queens",
                    "bronx",
                    "long island",
                    "staten island",
                    "new jersey",
                    "nj",
                    "connecticut",
                    "ct",
                    "pennsylvania",
                    "pa",
                    "tri state",
                    "tri-state",
                    "northeast",
                }
            )
        state_code = cls._US_STATE_CODES_BY_NAME.get(location_text)
        if state_code:
            terms.add(state_code)
        state_name = cls._US_STATE_NAMES_BY_CODE.get(location_text)
        if state_name:
            terms.add(state_name)
        return {term for term in terms if term}

    @classmethod
    def _allowed_area_codes(cls, location_text: str) -> set[str]:
        if location_text in {"new york", "new york city", "nyc", "ny"}:
            return {
                "212", "315", "332", "347", "516", "518", "585", "607", "631", "646",
                "680", "716", "718", "838", "845", "914", "917", "929", "934",
                "201", "551", "609", "640", "732", "848", "856", "862", "908", "973",
                "203", "475", "860", "959",
                "215", "223", "267", "272", "412", "445", "484", "570", "610", "717",
                "724", "814", "878",
            }
        if location_text in {"new jersey", "nj"}:
            return {
                "201", "551", "609", "640", "732", "848", "856", "862", "908", "973",
                "212", "315", "332", "347", "516", "518", "585", "607", "631", "646",
                "680", "716", "718", "838", "845", "914", "917", "929", "934",
                "215", "223", "267", "272", "412", "445", "484", "570", "610", "717",
                "724", "814", "878",
            }
        if location_text in {"connecticut", "ct"}:
            return {
                "203", "475", "860", "959",
                "212", "315", "332", "347", "516", "518", "585", "607", "631", "646",
                "680", "716", "718", "838", "845", "914", "917", "929", "934",
                "413", "508", "617", "774", "781", "857", "978",
                "401",
            }
        if location_text in {"alaska", "ak"}:
            return {"907"}
        return set()

    @staticmethod
    def _phone_area_code(value: Any) -> str:
        digits = re.sub(r"\D", "", str(value or ""))
        if len(digits) == 11 and digits.startswith("1"):
            digits = digits[1:]
        return digits[:3] if len(digits) >= 10 else ""

    @staticmethod
    def _is_toll_free_area_code(area_code: str) -> bool:
        return area_code in {"800", "833", "844", "855", "866", "877", "888"}

    @classmethod
    def _allowed_state_names(cls, location_text: str) -> set[str]:
        states: set[str] = set()
        if location_text in cls._US_STATE_CODES_BY_NAME:
            states.add(location_text)
        if location_text in cls._US_STATE_NAMES_BY_CODE:
            states.add(cls._US_STATE_NAMES_BY_CODE[location_text])
        if location_text in {"new york", "new york city", "nyc", "ny"}:
            states.update({"new york", "new jersey", "connecticut", "pennsylvania"})
        return states

    @staticmethod
    def _contains_location_term(text: str, term: str) -> bool:
        cleaned = " ".join(str(term or "").lower().split()).strip()
        if not cleaned:
            return False
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(cleaned)}(?![a-z0-9])", text or ""))

    @classmethod
    def _states_found_in_text(cls, text: str) -> set[str]:
        found: set[str] = set()
        for state_name in cls._US_STATE_CODES_BY_NAME:
            if cls._contains_location_term(text, state_name):
                found.add(state_name)
        return found

    _US_STATE_CODES_BY_NAME = {
        "alabama": "al",
        "alaska": "ak",
        "arizona": "az",
        "arkansas": "ar",
        "california": "ca",
        "colorado": "co",
        "connecticut": "ct",
        "delaware": "de",
        "florida": "fl",
        "georgia": "ga",
        "hawaii": "hi",
        "idaho": "id",
        "illinois": "il",
        "indiana": "in",
        "iowa": "ia",
        "kansas": "ks",
        "kentucky": "ky",
        "louisiana": "la",
        "maine": "me",
        "maryland": "md",
        "massachusetts": "ma",
        "michigan": "mi",
        "minnesota": "mn",
        "mississippi": "ms",
        "missouri": "mo",
        "montana": "mt",
        "nebraska": "ne",
        "nevada": "nv",
        "new hampshire": "nh",
        "new jersey": "nj",
        "new mexico": "nm",
        "new york": "ny",
        "north carolina": "nc",
        "north dakota": "nd",
        "ohio": "oh",
        "oklahoma": "ok",
        "oregon": "or",
        "pennsylvania": "pa",
        "rhode island": "ri",
        "south carolina": "sc",
        "south dakota": "sd",
        "tennessee": "tn",
        "texas": "tx",
        "utah": "ut",
        "vermont": "vt",
        "virginia": "va",
        "washington": "wa",
        "west virginia": "wv",
        "wisconsin": "wi",
        "wyoming": "wy",
    }
    _US_STATE_NAMES_BY_CODE = {code: name for name, code in _US_STATE_CODES_BY_NAME.items()}

    def _sanitize_linkedin_fields(self, item: Dict[str, Any], *, notes: List[str]) -> None:
        person_linkedin = self._first_non_empty(item.get("linkedin_url"))
        company_linkedin = self._first_non_empty(item.get("company_linkedin_url"))

        if self._is_company_linkedin_url(person_linkedin):
            if not company_linkedin:
                item["company_linkedin_url"] = person_linkedin
            item["linkedin_url"] = ""
            person_linkedin = ""
            notes.append("company_linkedin_moved_out_of_person_linkedin")
        elif person_linkedin and not self._is_person_linkedin_url(person_linkedin):
            item["linkedin_url"] = ""
            person_linkedin = ""
            notes.append("non_person_linkedin_removed")

        if company_linkedin and not self._is_company_linkedin_url(company_linkedin):
            if self._is_person_linkedin_url(company_linkedin) and not person_linkedin:
                item["linkedin_url"] = company_linkedin
            item["company_linkedin_url"] = ""
            notes.append("invalid_company_linkedin_removed")

        has_person_fields = any(
            self._first_non_empty(item.get(field))
            for field in ("contact_person_name", "founder_name", "contact_person_title", "linkedin_url")
        )
        if has_person_fields and not self._person_has_company_evidence(item):
            for field in ("contact_person_name", "founder_name", "contact_person_title", "linkedin_url"):
                item[field] = ""
            notes.append("unverified_person_contact_removed")

    @staticmethod
    def _is_person_linkedin_url(value: str) -> bool:
        return bool(re.search(r"linkedin\.com/(?:in|pub)/", value or "", flags=re.IGNORECASE))

    @staticmethod
    def _is_company_linkedin_url(value: str) -> bool:
        return bool(re.search(r"linkedin\.com/(?:company|showcase)/", value or "", flags=re.IGNORECASE))

    def _person_has_company_evidence(self, item: Dict[str, Any]) -> bool:
        person_name = self._first_non_empty(item.get("contact_person_name"), item.get("founder_name"))
        person_linkedin = self._first_non_empty(item.get("linkedin_url"))
        if person_linkedin and not self._is_person_linkedin_url(person_linkedin):
            return False
        if not person_name and not person_linkedin:
            return False

        company_name = self._first_non_empty(item.get("company_name"))
        website = self._first_non_empty(item.get("company_website"), item.get("contact_page"))
        domain_name = self._company_name_from_url(website) if website else ""
        company_tokens = (
            set(self._company_name_tokens(company_name))
            | set(self._company_name_tokens(domain_name))
            | set(re.findall(r"[a-z0-9]+", normalize_domain(website).split(".")[0] if website else ""))
        ) - self._generic_company_words()
        if not company_tokens:
            return False

        person_tokens = [token for token in re.findall(r"[a-z0-9]+", person_name.lower()) if len(token) >= 3]
        slug = self._linkedin_slug(person_linkedin)
        source_details = item.get("source_details") if isinstance(item.get("source_details"), list) else []
        for detail in source_details:
            if not isinstance(detail, dict):
                continue
            detail_text = json.dumps(detail, default=str).lower()
            compact_detail = re.sub(r"[^a-z0-9]+", "", detail_text)
            has_company = any(
                token in detail_text or token in compact_detail
                for token in company_tokens
                if len(token) >= 3
            )
            if not has_company:
                continue
            has_person = bool(
                person_tokens
                and any(token in detail_text or token in compact_detail for token in person_tokens)
            )
            has_person_linkedin = bool(person_linkedin and person_linkedin.lower() in detail_text)
            has_slug_name = bool(slug and person_tokens and any(token in slug for token in person_tokens))
            if has_person or has_person_linkedin or has_slug_name:
                return True
        return False

    @staticmethod
    def _linkedin_slug(value: str) -> str:
        path = urlparse(value or "").path.strip("/")
        if not path:
            return ""
        parts = path.split("/")
        if len(parts) >= 2 and parts[0].lower() in {"in", "pub"}:
            return re.sub(r"[^a-z0-9]+", "", parts[1].lower())
        return ""

    def _prefer_official_site_contact_fields(self, item: Dict[str, Any], *, domain: str, notes: List[str]) -> None:
        official_emails = [str(value).strip() for value in item.get("official_site_emails") or [] if str(value).strip()]
        if official_emails:
            previous_email = self._first_non_empty(item.get("contact_email"))
            best_email = self._pick_best_official_email(
                existing=previous_email,
                candidates=official_emails,
                website_domain=domain,
            )
            if best_email and best_email != previous_email:
                item["contact_email"] = best_email
                notes.append("email_preferred_from_official_site")

        official_phone_candidates = [
            candidate
            for candidate in item.get("official_site_phone_candidates") or []
            if isinstance(candidate, dict)
        ]
        if official_phone_candidates:
            ranked = rank_phone_candidates(official_phone_candidates)
            official_summary = build_phone_summary(ranked)
            previous_phone = self._first_non_empty(item.get("contact_phone"))
            self._replace_phone_summary(item, official_summary)
            if self._first_non_empty(item.get("contact_phone")) and item.get("contact_phone") != previous_phone:
                notes.append("phone_preferred_from_official_site")
            return

        phone_source = str(item.get("phone_source") or "").strip()
        phone_confidence = int(item.get("phone_confidence") or 0)
        if phone_source == "tavily_contact_search" and phone_confidence < 70:
            item["contact_phone"] = ""
            item["alternate_phones"] = []
            item["phone_validation_status"] = "needs_manual_review"
            notes.append("low_confidence_search_phone_removed")

    def _sanitize_email_field(self, item: Dict[str, Any], *, domain: str, notes: List[str]) -> None:
        email = self._first_non_empty(item.get("contact_email")).lower()
        if not email:
            return
        if self._is_placeholder_email(email):
            item["contact_email"] = ""
            notes.append("placeholder_email_removed")
            return
        email_domain = email.rsplit("@", 1)[-1] if "@" in email else ""
        if domain and email_domain and not self._email_matches_domain(email, domain) and email_domain not in self._free_email_domains():
            item["contact_email"] = ""
            notes.append("mismatched_email_domain_removed")

    @classmethod
    def _pick_best_official_email(cls, *, existing: Any, candidates: List[str], website_domain: str) -> str:
        official_pool: List[str] = []
        for candidate in candidates or []:
            cleaned = cls._first_non_empty(candidate).lower()
            if not cleaned or cls._is_placeholder_email(cleaned):
                continue
            email_domain = cleaned.rsplit("@", 1)[-1] if "@" in cleaned else ""
            domain_safe = (
                not website_domain
                or cls._email_matches_domain(cleaned, website_domain)
                or email_domain in cls._free_email_domains()
            )
            if domain_safe and cleaned not in official_pool:
                official_pool.append(cleaned)

        existing_email = cls._first_non_empty(existing).lower()
        if existing_email and not cls._is_placeholder_email(existing_email):
            existing_domain = existing_email.rsplit("@", 1)[-1] if "@" in existing_email else ""
            if existing_email in official_pool or cls._email_matches_domain(existing_email, website_domain):
                if existing_email not in official_pool:
                    official_pool.append(existing_email)
            elif not website_domain or existing_domain in cls._free_email_domains():
                if existing_email not in official_pool:
                    official_pool.append(existing_email)

        if not official_pool:
            return existing_email

        role_locals = {"info", "support", "contact", "hello", "sales", "admin", "office", "team", "help"}

        def score(value: str) -> tuple[int, int, int]:
            local, _, email_domain = value.partition("@")
            return (
                1 if cls._email_matches_domain(value, website_domain) else 0,
                0 if local.lower() in role_locals else 1,
                0 if email_domain.lower() in cls._free_email_domains() else 1,
            )

        return sorted(official_pool, key=score, reverse=True)[0]

    @staticmethod
    def _email_matches_domain(email: str, website_domain: str) -> bool:
        if "@" not in (email or "") or not website_domain:
            return False
        email_domain = email.rsplit("@", 1)[-1].lower().strip()
        normalized_site = normalize_domain(website_domain) or str(website_domain or "").lower().strip()
        if normalized_site.startswith("www."):
            normalized_site = normalized_site[4:]
        if not email_domain or not normalized_site:
            return False
        return email_domain == normalized_site or email_domain.endswith(f".{normalized_site}")

    @staticmethod
    def _free_email_domains() -> set[str]:
        return {
            "gmail.com",
            "googlemail.com",
            "yahoo.com",
            "hotmail.com",
            "outlook.com",
            "live.com",
            "icloud.com",
            "aol.com",
            "proton.me",
            "protonmail.com",
        }

    @staticmethod
    def _is_placeholder_email(value: str) -> bool:
        email = str(value or "").lower().strip()
        if "@" not in email:
            return True
        local, _, domain = email.partition("@")
        if domain in {"domain.com", "example.com", "example.net", "example.org", "yourdomain.com", "email.com"}:
            return True
        return local in {"user", "username", "name", "yourname", "test"} and domain in {"domain.com", "yourdomain.com"}

    @classmethod
    def _replace_phone_summary(cls, item: Dict[str, Any], summary: Dict[str, Any]) -> None:
        primary = cls._first_non_empty(summary.get("contact_phone"))
        if not primary:
            return
        alternates: List[str] = []
        for value in [item.get("contact_phone"), *(item.get("alternate_phones") or []), *(summary.get("alternate_phones") or [])]:
            cleaned = cls._first_non_empty(value)
            if cleaned and cleaned != primary and cleaned not in alternates:
                alternates.append(cleaned)
        item["contact_phone"] = primary
        item["alternate_phones"] = alternates[:4]
        item["phone_confidence"] = int(summary.get("phone_confidence") or 0)
        item["phone_source"] = summary.get("phone_source", "")
        item["phone_validation_status"] = summary.get("phone_validation_status", "")
        item["phone_type"] = summary.get("phone_type", "")
        item["phone_candidates"] = summary.get("phone_candidates", [])

    async def _search_company_contact_signals(
        self,
        *,
        company_name: str,
        domain: str,
        location: str,
        person_name: str = "",
    ) -> Dict[str, List[str]]:
        queries = [
            f'site:{domain} contact' if domain else "",
            f'"{company_name}" "{location}" email phone',
            f'site:{domain} linkedin' if domain else "",
            f'"{person_name}" "{company_name}" email' if person_name else "",
            f'"{company_name}" founder owner ceo linkedin',
            f'site:{domain} team' if domain else "",
            f'site:{domain} instagram OR youtube OR facebook' if domain else "",
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
        query_count = 0
        for raw_query in queries:
            query = " ".join((raw_query or "").split()).strip()
            if not query or query in seen_queries:
                continue
            seen_queries.add(query)
            if query_count >= self.contact_queries_per_company:
                break
            query_count += 1
            try:
                data = await self.tavily.search(
                    {
                        "query": query,
                        "search_depth": VELIT_SEARCH_DEPTH,
                        "max_results": 3,
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
            if signals["emails"] and signals["phones"]:
                break
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
        negative_markers = (
            "does not directly cater to van upfitters",
            "does not directly cater to",
            "travel advice",
            "destination guides",
            "trip planning",
            "travel guide",
            "tourism",
        )
        if any(marker in combined for marker in negative_markers):
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
            "commercial van upfitter",
            "fleet upfit",
            "work truck upfitter",
            "van shelving",
            "truck body",
            "commercial vehicle equipment",
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

    @classmethod
    def _phase_timeout_seconds(cls, phase: str, default: int) -> int:
        env_name = f"VELIT_{phase.upper()}_PHASE_TIMEOUT_SECONDS"
        return cls._bounded_int_env(env_name, default=default, minimum=10, maximum=600)
