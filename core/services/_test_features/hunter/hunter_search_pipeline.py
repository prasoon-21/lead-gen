from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List
from urllib.parse import urlparse

import httpx

from core.services.lead_discovery_policy import DEFAULT_TAVILY_SEARCH_DEPTH, build_targeted_directory_queries
from core.services.lead_quality_service import score_lead
from core.tools.external.tavily_client import TavilyClient


class HunterSearchPipeline:
    """Standalone Hunter.io lead discovery pipeline - TEST FEATURE."""

    def __init__(self, *, api_key: str | None = None):
        self.api_key = api_key or os.getenv("HUNTER_API_KEY")
        self.base_url = "https://api.hunter.io/v2"
        self.logger = logging.getLogger("hunter.pipeline")
        self.max_domain_searches = self._env_int("HUNTER_MAX_DOMAIN_SEARCHES", 10)
        self.max_email_finders = self._env_int("HUNTER_MAX_EMAIL_FINDERS", 20)
        self.max_email_verifications = self._env_int("HUNTER_MAX_EMAIL_VERIFICATIONS", 10)

    async def run(
        self,
        *,
        industry: str,
        location: str,
        seed_query: str = "",
        target_count: int = 15,
    ) -> Dict[str, Any]:
        target_count = max(1, min(int(target_count or 15), 100))
        errors: List[str] = []
        steps: List[Dict[str, Any]] = []

        if not self.api_key:
            return {
                "success": False,
                "leads": [],
                "steps": [{"name": "configuration", "status": "failed", "message": "HUNTER_API_KEY is not configured"}],
                "metadata": self._metadata(
                    domains_discovered=0,
                    domains_searched=0,
                    emails_found=0,
                    emails_verified=0,
                    returned=0,
                    errors=["HUNTER_API_KEY is not configured"],
                ),
                "error": "HUNTER_API_KEY is not configured",
            }

        domains = await self._discover_domains(industry=industry, location=location, seed_query=seed_query, errors=errors)
        steps.append(
            {
                "name": "domain_discovery",
                "status": "completed" if domains else "partial",
                "message": f"Discovered {len(domains)} candidate domains with Tavily.",
                "count": len(domains),
            }
        )

        domain_limit = min(self.max_domain_searches, max(target_count * 2, target_count))
        domain_payloads: List[Dict[str, Any]] = []
        for domain in domains[:domain_limit]:
            try:
                data = await self._hunter_domain_search(domain)
                domain_payloads.append({"domain": domain, "data": data})
            except httpx.HTTPStatusError as exc:
                message = self._hunter_http_error("domain-search", domain, exc)
                errors.append(message)
                if exc.response.status_code in {401, 429}:
                    break
            except Exception as exc:
                message = f"Hunter domain-search failed for {domain}: {exc}"
                errors.append(message)
                self.logger.warning("hunter_domain_search_failed", extra={"payload": message})
            if self._lead_capacity_reached(domain_payloads, target_count):
                break

        leads = self._assemble_domain_search_leads(
            domain_payloads=domain_payloads,
            industry=industry,
            location=location,
            target_count=target_count,
        )
        steps.append(
            {
                "name": "hunter_domain_search",
                "status": "completed" if domain_payloads else "partial",
                "message": f"Searched {len(domain_payloads)} domains with Hunter Domain Search.",
                "count": len(domain_payloads),
            }
        )

        email_finder_calls = 0
        for lead in leads:
            if email_finder_calls >= self.max_email_finders:
                break
            if lead.get("contact_email"):
                continue
            name_parts = self._split_name(str(lead.get("contact_person_name") or lead.get("founder_name") or ""))
            if not name_parts:
                continue
            try:
                finder = await self._hunter_email_finder(str(lead.get("domain") or ""), name_parts[0], name_parts[1])
                email_finder_calls += 1
                if finder.get("email"):
                    lead["contact_email"] = finder.get("email")
                    lead["email_confidence"] = finder.get("score") or finder.get("confidence") or lead.get("email_confidence")
                    lead["quality_score"] = max(int(lead.get("quality_score") or 0), int(lead.get("email_confidence") or 0))
                    lead["source"] = "hunter_email_finder"
                    lead["notes"] = f"Found via Hunter.io email finder for {lead.get('domain')}"
            except httpx.HTTPStatusError as exc:
                message = self._hunter_http_error("email-finder", str(lead.get("domain") or ""), exc)
                errors.append(message)
                if exc.response.status_code in {401, 429}:
                    break
            except Exception as exc:
                errors.append(f"Hunter email-finder failed for {lead.get('domain')}: {exc}")

        steps.append(
            {
                "name": "hunter_email_finder",
                "status": "completed",
                "message": f"Ran {email_finder_calls} Hunter Email Finder lookups.",
                "count": email_finder_calls,
            }
        )

        emails_verified = 0
        for lead in leads:
            if emails_verified >= self.max_email_verifications:
                break
            email = str(lead.get("contact_email") or "").strip()
            if not email:
                continue
            try:
                verification = await self._hunter_verify_email(email)
                emails_verified += 1
                lead["email_verification_status"] = verification.get("status") or verification.get("result") or ""
                lead["verification_status"] = self._verification_status(verification)
                lead["email_verification"] = verification
            except httpx.HTTPStatusError as exc:
                message = self._hunter_http_error("email-verifier", email, exc)
                errors.append(message)
                if exc.response.status_code in {401, 429}:
                    break
            except Exception as exc:
                errors.append(f"Hunter email-verifier failed for {email}: {exc}")

        steps.append(
            {
                "name": "hunter_email_verification",
                "status": "completed",
                "message": f"Verified {emails_verified} emails with Hunter Email Verifier.",
                "count": emails_verified,
            }
        )

        scored = []
        for lead in leads:
            quality = score_lead(lead, verification=lead.get("email_verification") or None)
            hunter_score = int(lead.get("quality_score") or 0)
            lead.update({key: value for key, value in quality.items() if key != "quality_score"})
            lead["quality_score"] = max(hunter_score, int(quality.get("quality_score") or 0))
            lead["quality_score"] = max(0, min(int(lead["quality_score"]), 100))
            scored.append(lead)
        scored.sort(key=lambda item: item.get("quality_score", 0), reverse=True)
        returned = scored[:target_count]

        steps.append(
            {
                "name": "scoring_assembly",
                "status": "completed",
                "message": f"Returned {len(returned)} Hunter-sourced leads.",
                "count": len(returned),
            }
        )

        return {
            "success": not errors or bool(returned),
            "leads": returned,
            "steps": steps,
            "metadata": self._metadata(
                domains_discovered=len(domains),
                domains_searched=len(domain_payloads),
                emails_found=sum(1 for lead in scored if lead.get("contact_email")),
                emails_verified=emails_verified,
                returned=len(returned),
                errors=errors,
            ),
            "errors": errors,
        }

    async def _discover_domains(self, *, industry: str, location: str, seed_query: str, errors: List[str]) -> List[str]:
        try:
            tavily = TavilyClient()
        except Exception as exc:
            errors.append(f"Tavily unavailable for domain discovery: {exc}")
            return []

        queries = build_targeted_directory_queries(
            industry=industry,
            location=location,
            seed_query=seed_query,
            max_queries=3,
        )
        domains: List[str] = []
        seen = set()
        for query in queries:
            try:
                data = await tavily.search(
                    {
                        "query": query,
                        "search_depth": DEFAULT_TAVILY_SEARCH_DEPTH,
                        "max_results": 10,
                        "include_raw_content": False,
                    }
                )
            except Exception as exc:
                message = f"Tavily search failed for '{query}': {exc}"
                errors.append(message)
                self.logger.warning("hunter_tavily_search_failed", extra={"payload": message})
                continue

            for item in data.get("results") or []:
                domain = self._domain_from_url(str(item.get("url") or ""))
                if not domain or domain in seen:
                    continue
                seen.add(domain)
                domains.append(domain)
        return domains

    async def _hunter_domain_search(self, domain: str) -> Dict[str, Any]:
        """GET https://api.hunter.io/v2/domain-search?domain={domain}&api_key={key}"""
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{self.base_url}/domain-search",
                params={"domain": domain, "api_key": self.api_key},
            )
            resp.raise_for_status()
            return resp.json().get("data", {})

    async def _hunter_email_finder(self, domain: str, first_name: str, last_name: str) -> Dict[str, Any]:
        """GET https://api.hunter.io/v2/email-finder"""
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{self.base_url}/email-finder",
                params={
                    "domain": domain,
                    "first_name": first_name,
                    "last_name": last_name,
                    "api_key": self.api_key,
                },
            )
            resp.raise_for_status()
            return resp.json().get("data", {})

    async def _hunter_verify_email(self, email: str) -> Dict[str, Any]:
        """GET https://api.hunter.io/v2/email-verifier"""
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{self.base_url}/email-verifier",
                params={"email": email, "api_key": self.api_key},
            )
            resp.raise_for_status()
            return resp.json().get("data", {})

    def _assemble_domain_search_leads(
        self,
        *,
        domain_payloads: List[Dict[str, Any]],
        industry: str,
        location: str,
        target_count: int,
    ) -> List[Dict[str, Any]]:
        leads: List[Dict[str, Any]] = []
        seen_emails = set()
        for payload in domain_payloads:
            domain = payload.get("domain") or ""
            data = payload.get("data") or {}
            organization_name = data.get("organization") or self._name_from_domain(domain)
            city = data.get("city") or ""
            state = data.get("state") or ""
            resolved_location = ", ".join(value for value in [city, state] if value) or location
            emails = data.get("emails") or []
            if not emails:
                leads.append(
                    self._lead_from_hunter_email(
                        domain=domain,
                        email_item={},
                        organization_name=organization_name,
                        industry=industry,
                        location=resolved_location,
                        city=city,
                        state=state,
                    )
                )
                continue

            for email_item in emails:
                email = str(email_item.get("value") or "").strip().lower()
                if email and email in seen_emails:
                    continue
                if email:
                    seen_emails.add(email)
                leads.append(
                    self._lead_from_hunter_email(
                        domain=domain,
                        email_item=email_item,
                        organization_name=organization_name,
                        industry=industry,
                        location=resolved_location,
                        city=city,
                        state=state,
                    )
                )
                if len(leads) >= max(target_count * 3, target_count):
                    return leads
        return leads

    def _lead_from_hunter_email(
        self,
        *,
        domain: str,
        email_item: Dict[str, Any],
        organization_name: str,
        industry: str,
        location: str,
        city: str,
        state: str,
    ) -> Dict[str, Any]:
        first_name = str(email_item.get("first_name") or "").strip()
        last_name = str(email_item.get("last_name") or "").strip()
        person_name = " ".join(value for value in [first_name, last_name] if value)
        confidence = int(email_item.get("confidence") or 0)
        verification_status = "hunter_verified" if email_item.get("value") else "hunter_domain_search"
        domain_url = f"https://{domain}" if domain else ""
        value_prop = f"{organization_name} appears in Hunter.io results for {industry} in {location}."
        return {
            "company_name": organization_name,
            "company_website": domain_url,
            "domain": domain,
            "industry": industry,
            "location": location,
            "city": city,
            "state": state,
            "contact_person_name": person_name,
            "founder_name": person_name,
            "contact_person_title": email_item.get("position") or "",
            "contact_email": email_item.get("value") or "",
            "email_confidence": confidence,
            "contact_phone": email_item.get("phone_number") or "",
            "phone": email_item.get("phone_number") or "",
            "linkedin_url": email_item.get("linkedin") or "",
            "quality_score": confidence,
            "source": "hunter_domain_search",
            "source_urls": [domain_url] if domain_url else [],
            "value_proposition": value_prop,
            "lead_summary": value_prop,
            "email_verification_status": "",
            "verification_status": verification_status,
            "notes": f"Found via Hunter.io domain search for {domain}",
        }

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        try:
            return max(0, int(os.getenv(name, str(default))))
        except Exception:
            return default

    @staticmethod
    def _domain_from_url(url: str) -> str:
        parsed = urlparse(url if "://" in url else f"https://{url}")
        host = (parsed.hostname or "").lower().strip()
        if host.startswith("www."):
            host = host[4:]
        return host

    @staticmethod
    def _name_from_domain(domain: str) -> str:
        root = (domain or "").split(".")[0]
        return root.replace("-", " ").replace("_", " ").title() if root else ""

    @staticmethod
    def _split_name(name: str) -> tuple[str, str] | None:
        parts = [part for part in name.split() if part]
        if len(parts) < 2:
            return None
        return parts[0], parts[-1]

    @staticmethod
    def _verification_status(verification: Dict[str, Any]) -> str:
        status = str(verification.get("status") or verification.get("result") or "").lower()
        if status == "valid":
            return "hunter_verified"
        if status:
            return f"hunter_{status}"
        return "hunter_checked"

    @staticmethod
    def _metadata(
        *,
        domains_discovered: int,
        domains_searched: int,
        emails_found: int,
        emails_verified: int,
        returned: int,
        errors: List[str],
    ) -> Dict[str, Any]:
        return {
            "provider": "hunter",
            "domains_discovered": domains_discovered,
            "domains_searched": domains_searched,
            "emails_found": emails_found,
            "emails_verified": emails_verified,
            "returned": returned,
            "errors": errors,
        }

    def _hunter_http_error(self, endpoint: str, subject: str, exc: httpx.HTTPStatusError) -> str:
        detail = ""
        try:
            parsed = exc.response.json()
            if isinstance(parsed, dict):
                detail = str(parsed.get("errors") or parsed.get("details") or parsed.get("message") or "")
        except Exception:
            detail = exc.response.text[:200]
        message = f"Hunter {endpoint} failed for {subject}: HTTP {exc.response.status_code}"
        if detail:
            message = f"{message} - {detail}"
        self.logger.warning("hunter_http_error", extra={"payload": message})
        return message

    @staticmethod
    def _lead_capacity_reached(domain_payloads: List[Dict[str, Any]], target_count: int) -> bool:
        email_count = sum(len((payload.get("data") or {}).get("emails") or []) for payload in domain_payloads)
        return email_count >= max(target_count * 2, target_count)
