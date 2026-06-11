import json
import logging
import os
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

import httpx

from core.tools.base import BaseTool, ToolContext

logger = logging.getLogger(__name__)


class LinkedInResearchTool(BaseTool):
    """
    Scrapes LinkedIn profile data using LinkedIn Voyager HTTP endpoints and
    then builds stronger outreach-friendly enrichment using both LinkedIn and
    fallback company-website contact paths.

    Requires a valid `linkedin_session.json` with a li_at cookie in the
    project root.
    """

    name = "linkedin_research"
    description = (
        "Extract detailed person or company information from a LinkedIn URL and "
        "return outreach-friendly contact paths, decision-maker hints, and "
        "fallback website contact details."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Full LinkedIn profile or company URL.",
            },
            "type": {
                "type": "string",
                "enum": ["person", "company"],
                "description": "Optional explicit page type. If omitted it is inferred from the URL.",
            },
            "company_name": {
                "type": "string",
                "description": "Optional company name for additional context.",
            },
            "website_url": {
                "type": "string",
                "description": "Optional company website for fallback contact discovery.",
            },
            "target_roles": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional priority role hints such as founder, owner, ceo, or director.",
            },
            "allow_fallback_contact_paths": {
                "type": "boolean",
                "description": "When true, inspect the website for fallback contact pages, emails, and phone numbers.",
            },
        },
        "required": ["url"],
    }

    _priority_roles = (
        ("founder", 100),
        ("co-founder", 95),
        ("owner", 92),
        ("chief executive officer", 90),
        ("ceo", 90),
        ("managing director", 88),
        ("director", 80),
        ("head", 74),
        ("vp", 70),
        ("vice president", 70),
        ("operations", 64),
        ("partnerships", 64),
        ("business development", 64),
        ("marketing", 58),
    )
    _contact_link_keywords = ("contact", "about", "team", "leadership", "company", "staff", "management")
    _email_pattern = re.compile(r"[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}", re.IGNORECASE)
    _phone_pattern = re.compile(r"(?:\+?\d[\d\s().-]{7,}\d)")

    def _load_session(self, session_path: str) -> Dict[str, str]:
        with open(session_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        cookies_list = data.get("cookies", [])
        return {cookie["name"]: cookie["value"] for cookie in cookies_list}

    @staticmethod
    def _extract_profile_id(url: str) -> Optional[str]:
        match = re.search(r"linkedin\.com/in/([^/?#]+)", url)
        return match.group(1).strip("/") if match else None

    @staticmethod
    def _extract_company_id(url: str) -> Optional[str]:
        match = re.search(r"linkedin\.com/company/([^/?#]+)", url)
        return match.group(1).strip("/") if match else None

    def _build_headers(self, li_at: str, jsessionid: str) -> Dict[str, str]:
        csrf_token = jsessionid.replace('"', "")
        return {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "application/vnd.linkedin.normalized+json+2.1",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "x-li-lang": "en_US",
            "x-li-page-instance": "urn:li:page:d_flagship3_profile_view_base;DUMMY",
            "x-li-track": json.dumps(
                {
                    "clientVersion": "1.13.5049",
                    "mpVersion": "1.13.5049",
                    "osName": "web",
                    "timezoneOffset": 5.5,
                    "timezone": "Asia/Calcutta",
                    "deviceFormFactor": "DESKTOP",
                    "mpName": "voyager-web",
                    "displayDensity": 1,
                    "displayWidth": 1920,
                    "displayHeight": 1080,
                }
            ),
            "x-restli-protocol-version": "2.0.0",
            "csrf-token": csrf_token,
            "Cookie": f'li_at={li_at}; JSESSIONID="{csrf_token}"',
            "Referer": "https://www.linkedin.com/",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
        }

    @staticmethod
    def _normalize_url(url: str) -> str:
        parsed = urlparse((url or "").strip())
        if not parsed.scheme or not parsed.netloc:
            return ""
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")

    @staticmethod
    def _clean_text(value: Any) -> str:
        return " ".join(str(value or "").split()).strip()

    def _role_priority(self, headline: str, target_roles: List[str]) -> Dict[str, Any]:
        text = self._clean_text(headline).lower()
        best_match = ""
        best_score = 15

        for role, score in self._priority_roles:
            if role in text and score > best_score:
                best_match = role
                best_score = score

        for role in target_roles:
            normalized = self._clean_text(role).lower()
            if normalized and normalized in text:
                best_match = normalized
                best_score = max(best_score, 96)

        if best_score >= 88:
            band = "high"
        elif best_score >= 64:
            band = "medium"
        else:
            band = "low"

        return {
            "decision_maker_match": best_match or None,
            "decision_maker_score": best_score,
            "decision_maker_priority": band,
        }

    @staticmethod
    def _strip_html(html: str) -> str:
        cleaned = re.sub(r"(?is)<script.*?>.*?</script>", " ", html or "")
        cleaned = re.sub(r"(?is)<style.*?>.*?</style>", " ", cleaned)
        cleaned = re.sub(r"(?is)<[^>]+>", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    def _extract_emails(self, text: str) -> List[str]:
        seen: List[str] = []
        for match in self._email_pattern.findall(text or ""):
            normalized = match.strip().lower()
            if normalized not in seen:
                seen.append(normalized)
        return seen[:5]

    def _extract_phones(self, text: str) -> List[str]:
        seen: List[str] = []
        for match in self._phone_pattern.findall(text or ""):
            normalized = self._clean_text(match)
            if len(re.sub(r"\D", "", normalized)) < 8:
                continue
            if normalized not in seen:
                seen.append(normalized)
        return seen[:5]

    def _discover_links(self, base_url: str, html: str) -> List[str]:
        links: List[str] = []
        base_host = urlparse(base_url).netloc.lower()
        for href in re.findall(r'href=["\']([^"\']+)["\']', html or "", flags=re.IGNORECASE):
            absolute = urljoin(base_url, href)
            parsed = urlparse(absolute)
            if not parsed.scheme.startswith("http"):
                continue
            if parsed.netloc.lower() != base_host:
                continue
            path_lower = parsed.path.lower()
            if any(keyword in path_lower for keyword in self._contact_link_keywords):
                normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")
                if normalized not in links:
                    links.append(normalized)
        return links[:8]

    async def _fetch_page(self, client: httpx.AsyncClient, url: str) -> Optional[Dict[str, Any]]:
        try:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()
            html = response.text
            title_match = re.search(r"(?is)<title[^>]*>(.*?)</title>", html)
            return {
                "url": str(response.url),
                "title": self._strip_html(title_match.group(1))[:200] if title_match else "",
                "html": html,
                "text": self._strip_html(html)[:5000],
            }
        except httpx.RequestError as exc:
            logger.warning(f"HTTP request failed for {url}: {exc}")
            return None
        except Exception as exc:
            logger.error(f"Unexpected error fetching {url}: {exc}")
            return None

    async def _discover_website_contact_paths(self, website_url: str) -> Dict[str, Any]:
        normalized_site = self._normalize_url(website_url)
        if not normalized_site:
            return {}

        page_candidates = [
            normalized_site,
            urljoin(normalized_site + "/", "contact"),
            urljoin(normalized_site + "/", "contact-us"),
            urljoin(normalized_site + "/", "about"),
            urljoin(normalized_site + "/", "team"),
        ]

        pages: List[Dict[str, Any]] = []
        seen_urls = set()

        try:
            async with httpx.AsyncClient(timeout=12) as client:
                homepage = await self._fetch_page(client, normalized_site)
                homepage_html = ""
                if homepage:
                    pages.append(homepage)
                    seen_urls.add(homepage["url"].rstrip("/"))
                    homepage_html = homepage["html"]

                for discovered in self._discover_links(normalized_site, homepage_html):
                    if discovered.rstrip("/") not in seen_urls:
                        page_candidates.append(discovered)

                for candidate in page_candidates:
                    key = candidate.rstrip("/")
                    if key in seen_urls:
                        continue
                    page = await self._fetch_page(client, candidate)
                    if not page:
                        continue
                    seen_urls.add(key)
                    pages.append(page)
                    if len(pages) >= 6:
                        break
        except Exception as exc:
            logger.error(f"Website contact discovery failed for {website_url}: {exc}")
            return {}

        emails: List[str] = []
        phones: List[str] = []
        contact_page = ""
        for page in pages:
            text = page.get("text", "")
            for email in self._extract_emails(text):
                if email not in emails:
                    emails.append(email)
            for phone in self._extract_phones(text):
                if phone not in phones:
                    phones.append(phone)
            page_url = page.get("url", "").lower()
            if not contact_page and any(keyword in page_url for keyword in ("contact", "about", "team", "leadership")):
                contact_page = page.get("url", "")

        return {
            "website_url": normalized_site,
            "contact_page": contact_page or normalized_site,
            "website_emails": emails[:3],
            "website_phones": phones[:3],
            "website_pages_scanned": [
                {"url": page.get("url", ""), "title": page.get("title", "")}
                for page in pages
            ],
        }

    def _build_contact_paths(
        self,
        linkedin_url: str,
        email: str,
        phone: str,
        contact_page: str,
        website_url: str,
        website_emails: List[str],
        website_phones: List[str],
    ) -> List[Dict[str, str]]:
        paths: List[Dict[str, str]] = []

        def add_path(path_type: str, value: str, source: str) -> None:
            cleaned = self._clean_text(value)
            if not cleaned:
                return
            if any(existing["type"] == path_type and existing["value"] == cleaned for existing in paths):
                return
            paths.append({"type": path_type, "value": cleaned, "source": source})

        add_path("direct_email", email, "linkedin")
        add_path("phone", phone, "linkedin")
        add_path("linkedin_profile", linkedin_url, "linkedin")
        add_path("contact_page", contact_page, "website")
        add_path("company_website", website_url, "website")
        for website_email in website_emails[:2]:
            add_path("website_email", website_email, "website")
        for website_phone in website_phones[:2]:
            add_path("website_phone", website_phone, "website")
        return paths

    @staticmethod
    def _confidence_level(
        direct_email: str,
        direct_phone: str,
        linkedin_url: str,
        website_emails: List[str],
        contact_page: str,
        priority_band: str,
    ) -> str:
        if direct_email and priority_band == "high":
            return "high"
        if direct_phone and direct_email:
            return "high"
        if linkedin_url and (website_emails or contact_page):
            return "medium"
        if linkedin_url or website_emails or contact_page:
            return "medium"
        return "low"

    async def run(self, arguments: Dict[str, Any], context: ToolContext) -> Dict[str, Any]:
        url = (arguments.get("url") or "").strip()
        page_type = arguments.get("type")
        website_url = (arguments.get("website_url") or context.state.get("company_website") or "").strip()
        target_roles = [self._clean_text(item).lower() for item in arguments.get("target_roles") or [] if self._clean_text(item)]
        allow_fallback_contact_paths = bool(arguments.get("allow_fallback_contact_paths", True))

        logger.info(f"Running LinkedIn research with arguments: {arguments}")

        if not url:
            return {"error": "LinkedIn URL is required"}

        if not page_type:
            if "/in/" in url:
                page_type = "person"
            elif "/company/" in url:
                page_type = "company"
            else:
                return {"error": "Could not infer page type from URL. Please specify 'person' or 'company'."}

        session_path = os.path.join(os.getcwd(), "linkedin_session.json")
        if os.getenv("VERCEL") == "1":
            session_path = os.path.join("/tmp", "linkedin_session.json")
        if not os.path.exists(session_path):
            logger.error("LinkedIn session file not found at %s", session_path)
            return {
                "error": "LinkedIn session file not found. Please use the LinkedIn Session Sync widget to authenticate."
            }

        try:
            cookies = self._load_session(session_path)
            li_at = cookies.get("li_at", "")
            if not li_at:
                logger.error("No li_at cookie found in session file.")
                return {"error": "No li_at cookie found in session. Please re-sync via the LinkedIn Session Sync widget."}
        except Exception as exc:
            logger.error(f"Failed to load LinkedIn session: {exc}", exc_info=True)
            return {"error": f"Failed to load session: {exc}"}

        try:
            async with httpx.AsyncClient(follow_redirects=False, timeout=20, verify=True) as client:
                home_headers = {
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"
                    ),
                    "Accept-Language": "en-US,en;q=0.9",
                }
                try:
                    home_response = await client.get("https://www.linkedin.com", headers=home_headers)
                    home_response.raise_for_status()
                except Exception as exc:
                    logger.error("Failed to fetch LinkedIn homepage to get a JSESSIONID: %s", exc)
                    return {"error": "Failed to initialize a session with LinkedIn. The site may be blocking requests."}

                jsessionid = client.cookies.get("JSESSIONID", "")
                if not jsessionid:
                    logger.error("Could not retrieve a JSESSIONID cookie from LinkedIn homepage.")
                    return {"error": "Could not retrieve a required JSESSIONID cookie. LinkedIn might be blocking the request."}

                client.cookies.set("li_at", li_at, domain=".linkedin.com")
                headers = self._build_headers(li_at, jsessionid)

                if page_type == "person":
                    return await self._scrape_person(
                        client=client,
                        headers=headers,
                        url=url,
                        website_url=website_url,
                        target_roles=target_roles,
                        allow_fallback_contact_paths=allow_fallback_contact_paths,
                    )
                return await self._scrape_company(
                    client=client,
                    headers=headers,
                    url=url,
                    website_url=website_url,
                    target_roles=target_roles,
                    allow_fallback_contact_paths=allow_fallback_contact_paths,
                    company_name_hint=(arguments.get("company_name") or context.state.get("company_name") or ""),
                )
        except Exception as exc:
            logger.error("LinkedIn scraping failed: %s", exc, exc_info=True)
            return {"error": f"Scraping failed: {exc}"}

    async def _scrape_person(
        self,
        client: httpx.AsyncClient,
        headers: Dict[str, str],
        url: str,
        website_url: str,
        target_roles: List[str],
        allow_fallback_contact_paths: bool,
    ) -> Dict[str, Any]:
        profile_id = self._extract_profile_id(url)
        if not profile_id:
            return {"error": f"Could not extract profile ID from URL: {url}"}

        result: Dict[str, Any] = {
            "profile_url": url,
            "linkedin_url": url,
            "profile_id": profile_id,
            "source": "linkedin",
        }

        profile_url = f"https://www.linkedin.com/voyager/api/identity/profiles/{profile_id}"
        try:
            resp = await client.get(profile_url, headers=headers)
            logger.info("Profile API status for %s: %s", profile_id, resp.status_code)
            resp.raise_for_status()

            data = resp.json()
            full_name = f"{data.get('firstName', '')} {data.get('lastName', '')}".strip()
            headline = data.get("headline", "") or ""
            result["full_name"] = full_name
            result["contact_person_name"] = full_name
            result["founder_name"] = full_name
            result["contact_person_title"] = headline
            result["headline"] = headline
            result["summary"] = data.get("summary", "") or ""
            result["location"] = data.get("geoLocationName", "") or data.get("geoCountryName", "")

        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (401, 403):
                logger.error("LinkedIn session is not authenticated (li_at expired?).")
                return {"error": "LinkedIn session is not authenticated. Your li_at cookie may have expired. Please re-sync."}
            if exc.response.status_code == 404:
                return {"error": f"Profile not found: {profile_id}"}
            logger.warning("HTTP error fetching profile %s: %s", profile_id, exc)
            if exc.response.status_code == 410: # GONE
                logger.info("Profile %s returned 410, attempting alternate fetch.", profile_id)
                result.update(await self._scrape_person_alternate(client, headers, url, profile_id))
        except json.JSONDecodeError as exc:
            logger.error("Failed to decode JSON from profile API for %s: %s", profile_id, exc)
            return {"error": f"Failed to parse LinkedIn profile response for {profile_id}."}
        except Exception as exc:
            logger.error("Profile fetch for %s failed: %s", profile_id, exc, exc_info=True)
            # Don't proceed if the primary profile fetch fails catastrophically
            return {"error": f"An unexpected error occurred while fetching profile {profile_id}."}

        contact_url = f"https://www.linkedin.com/voyager/api/identity/profiles/{profile_id}/profileContactInfo"
        try:
            resp2 = await client.get(contact_url, headers=headers)
            logger.info("Contact API status for %s: %s", profile_id, resp2.status_code)
            resp2.raise_for_status()
            
            cdata = resp2.json()
            email_addr = cdata.get("emailAddress", "") or ""
            if email_addr:
                result["contact_email"] = email_addr

            phone_nums = cdata.get("phoneNumbers", []) or []
            if phone_nums:
                result["contact_phone"] = phone_nums[0].get("number", "") or ""

            websites = cdata.get("websites", []) or []
            if websites and not website_url:
                website_url = websites[0].get("url", "") or ""
            if website_url:
                result["company_website"] = website_url
        except Exception as exc:
            logger.warning("Contact info fetch for %s failed: %s", profile_id, exc)

        role_data = self._role_priority(result.get("contact_person_title", ""), target_roles)
        result.update(role_data)
        if result.get("decision_maker_priority") == "high":
            result["recommended_for_outreach"] = True

        website_fallback: Dict[str, Any] = {}
        if allow_fallback_contact_paths and website_url:
            website_fallback = await self._discover_website_contact_paths(website_url)
            if website_fallback.get("website_url"):
                result.setdefault("company_website", website_fallback["website_url"])
            if website_fallback.get("contact_page"):
                result["contact_page"] = website_fallback["contact_page"]
            if not result.get("contact_phone") and website_fallback.get("website_phones"):
                result["contact_phone"] = website_fallback["website_phones"][0]

        contact_paths = self._build_contact_paths(
            linkedin_url=result.get("linkedin_url", ""),
            email=result.get("contact_email", ""),
            phone=result.get("contact_phone", ""),
            contact_page=result.get("contact_page", ""),
            website_url=result.get("company_website", ""),
            website_emails=website_fallback.get("website_emails", []),
            website_phones=website_fallback.get("website_phones", []),
        )
        result["contact_paths"] = contact_paths
        result["fallback_contact_paths"] = [path for path in contact_paths if path["type"] != "direct_email"]
        result["confidence"] = self._confidence_level(
            direct_email=result.get("contact_email", ""),
            direct_phone=result.get("contact_phone", ""),
            linkedin_url=result.get("linkedin_url", ""),
            website_emails=website_fallback.get("website_emails", []),
            contact_page=result.get("contact_page", ""),
            priority_band=result.get("decision_maker_priority", "low"),
        )
        if website_fallback:
            result["website_research"] = website_fallback

        if not result.get("full_name") and not result.get("contact_email"):
            result["info"] = (
                f"Successfully accessed LinkedIn profile for '{profile_id}', but no direct public "
                "email or phone number was available. Fallback contact paths may still be usable."
            )

        return result

    async def _scrape_person_alternate(
        self,
        client: httpx.AsyncClient,
        headers: Dict[str, str],
        url: str,
        profile_id: str,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {"profile_url": url, "linkedin_url": url, "profile_id": profile_id}

        norm_url = (
            f"https://www.linkedin.com/voyager/api/identity/profiles/{profile_id}"
            f"?decorationId=com.linkedin.voyager.dash.deco.identity.profile.FullProfileWithEntities-100"
        )
        try:
            resp = await client.get(norm_url, headers=headers)
            resp.raise_for_status()
            
            data = resp.json()
            full_name = f"{data.get('firstName', '')} {data.get('lastName', '')}".strip()
            result["full_name"] = full_name
            result["contact_person_name"] = full_name
            result["founder_name"] = full_name
            result["contact_person_title"] = data.get("headline", "") or ""
            result["headline"] = data.get("headline", "") or ""
            result["location"] = data.get("geoLocationName", "") or ""
        except Exception as exc:
            logger.warning("Alternate profile fetch for %s failed: %s", profile_id, exc)

        contact_url = f"https://www.linkedin.com/voyager/api/identity/profiles/{profile_id}/profileContactInfo"
        try:
            resp2 = await client.get(contact_url, headers=headers)
            resp2.raise_for_status()
            
            cdata = resp2.json()
            result["contact_email"] = cdata.get("emailAddress", "") or ""
            phones = cdata.get("phoneNumbers", []) or []
            if phones:
                result["contact_phone"] = phones[0].get("number", "") or ""
        except Exception as exc:
            logger.warning("Alternate contact info fetch for %s failed: %s", profile_id, exc)

        return result

    async def _scrape_company(
        self,
        client: httpx.AsyncClient,
        headers: Dict[str, str],
        url: str,
        website_url: str,
        target_roles: List[str],
        allow_fallback_contact_paths: bool,
        company_name_hint: str,
    ) -> Dict[str, Any]:
        company_id = self._extract_company_id(url)
        if not company_id:
            return {"error": f"Could not extract company ID from URL: {url}"}

        result: Dict[str, Any] = {
            "profile_url": url,
            "linkedin_url": url,
            "company_id": company_id,
            "source": "linkedin",
            "company_name": self._clean_text(company_name_hint),
        }

        company_url = f"https://www.linkedin.com/voyager/api/organization/companies?q=universalName&universalName={company_id}"
        try:
            resp = await client.get(company_url, headers=headers)
            resp.raise_for_status()
            
            data = resp.json()
            elements = data.get("elements", []) or []
            if elements:
                company = elements[0]
                result["company_name"] = company.get("name", "") or result.get("company_name", "")
                result["tagline"] = company.get("tagline", "") or ""
                result["description"] = company.get("description", "") or ""
                result["value_proposition"] = company.get("tagline", "") or company.get("description", "") or ""
                result["industry"] = company.get("industries", [{}])[0].get("localizedName", "") if company.get("industries") else ""
                result["company_size"] = str(company.get("staffCountRange", {}).get("start", "")) or ""
                result["location"] = company.get("headquarter", {}).get("city", "") or ""
                if company.get("companyPageUrl"):
                    result["linkedin_company_url"] = company.get("companyPageUrl")
                website_url = (
                    website_url
                    or company.get("companyWebsite", "")
                    or company.get("websiteUrl", "")
                    or company.get("url", "")
                )
                if website_url:
                    result["company_website"] = website_url
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (401, 403):
                logger.error("LinkedIn session is not authenticated (li_at expired?).")
                return {"error": "LinkedIn session is not authenticated. Please re-sync your li_at cookie."}
            logger.warning("HTTP error fetching company %s: %s", company_id, exc)
        except Exception as exc:
            logger.error("Company fetch for %s failed: %s", company_id, exc, exc_info=True)

        website_fallback: Dict[str, Any] = {}
        if allow_fallback_contact_paths and website_url:
            website_fallback = await self._discover_website_contact_paths(website_url)
            if website_fallback.get("website_url"):
                result["company_website"] = website_fallback["website_url"]
            if website_fallback.get("contact_page"):
                result["contact_page"] = website_fallback["contact_page"]
            if website_fallback.get("website_emails"):
                result["contact_email"] = website_fallback["website_emails"][0]
            if website_fallback.get("website_phones"):
                result["contact_phone"] = website_fallback["website_phones"][0]

        fallback_title = target_roles[0] if target_roles else "decision-maker"
        result.setdefault("contact_person_title", fallback_title)
        role_data = self._role_priority(result.get("contact_person_title", ""), target_roles)
        result.update(role_data)

        result["contact_paths"] = self._build_contact_paths(
            linkedin_url=result.get("linkedin_url", ""),
            email=result.get("contact_email", ""),
            phone=result.get("contact_phone", ""),
            contact_page=result.get("contact_page", ""),
            website_url=result.get("company_website", ""),
            website_emails=website_fallback.get("website_emails", []),
            website_phones=website_fallback.get("website_phones", []),
        )
        result["fallback_contact_paths"] = result["contact_paths"]
        result["confidence"] = self._confidence_level(
            direct_email=result.get("contact_email", ""),
            direct_phone=result.get("contact_phone", ""),
            linkedin_url=result.get("linkedin_url", ""),
            website_emails=website_fallback.get("website_emails", []),
            contact_page=result.get("contact_page", ""),
            priority_band=result.get("decision_maker_priority", "low"),
        )
        result["needs_people_enrichment"] = True
        if website_fallback:
            result["website_research"] = website_fallback

        return result
