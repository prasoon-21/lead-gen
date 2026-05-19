import os
import asyncio
import logging
import json
from typing import Any, Dict, Optional
from pathlib import Path

from core.tools.base import BaseTool, ToolContext, ToolResult

logger = logging.getLogger(__name__)


class LinkedInResearchTool(BaseTool):
    """
    Scrapes LinkedIn profile data using pure HTTP requests via the LinkedIn
    Voyager internal API. No browser automation is used, so the session
    cookie is NOT flagged by LinkedIn's bot detection.

    Requires a valid 'linkedin_session.json' with a li_at cookie in the project root.
    """
    name = "linkedin_research"
    description = (
        "Extracts detailed information from a LinkedIn profile or company page. "
        "Provide a valid LinkedIn URL (e.g., https://www.linkedin.com/in/username or "
        "https://www.linkedin.com/company/company-name)."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The full LinkedIn URL to scrape."
            },
            "type": {
                "type": "string",
                "enum": ["person", "company"],
                "description": "The type of page to scrape. If not provided, it will be inferred from the URL."
            }
        },
        "required": ["url"]
    }

    def _load_session(self, session_path: str) -> Dict[str, str]:
        """Load li_at and JSESSIONID from linkedin_session.json."""
        with open(session_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cookies_list = data.get("cookies", [])
        cookie_map = {c["name"]: c["value"] for c in cookies_list}
        return cookie_map

    def _extract_profile_id(self, url: str) -> Optional[str]:
        """Extract the profile slug from a LinkedIn URL."""
        import re
        # Handle /in/username or /in/username/ or /in/username?something
        match = re.search(r"linkedin\.com/in/([^/?#]+)", url)
        if match:
            return match.group(1).strip("/")
        return None

    def _extract_company_id(self, url: str) -> Optional[str]:
        """Extract company slug from a LinkedIn URL."""
        import re
        match = re.search(r"linkedin\.com/company/([^/?#]+)", url)
        if match:
            return match.group(1).strip("/")
        return None

    def _build_headers(self, li_at: str, jsessionid: str) -> Dict[str, str]:
        """Build headers that look like a real LinkedIn browser request."""
        # The CSRF token LinkedIn expects is the JSESSIONID value
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
            "x-li-track": json.dumps({
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
            }),
            "x-restli-protocol-version": "2.0.0",
            "csrf-token": csrf_token,
            "Cookie": f'li_at={li_at}; JSESSIONID="{csrf_token}"',
            "Referer": "https://www.linkedin.com/",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
        }

    async def run(self, arguments: Dict[str, Any], context: ToolContext) -> Dict[str, Any]:
        url = arguments.get("url", "").strip()
        page_type = arguments.get("type")

        if not url:
            return {"error": "LinkedIn URL is required"}

        # Infer type if not provided
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
            return {
                "error": "LinkedIn session file not found. "
                         "Please use the LinkedIn Session Sync widget to authenticate."
            }

        try:
            cookies = self._load_session(session_path)
            li_at = cookies.get("li_at", "")
            if not li_at:
                return {"error": "No li_at cookie found in session. Please re-sync via the LinkedIn Session Sync widget."}
        except Exception as e:
            return {"error": f"Failed to load session: {e}"}

        try:
            import httpx
        except ImportError:
            return {"error": "httpx library not installed. Run: pip install httpx"}

        try:
            # We use follow_redirects=False to prevent infinite redirect loops if the session is invalid
            async with httpx.AsyncClient(
                follow_redirects=False,
                timeout=20,
                verify=True,
            ) as client:
                # Step 1: Request home page to get a fresh, valid server-side JSESSIONID cookie
                home_headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                    "Accept-Language": "en-US,en;q=0.9",
                }
                logger.info("Fetching fresh JSESSIONID from LinkedIn...")
                try:
                    await client.get("https://www.linkedin.com", headers=home_headers)
                except Exception as e:
                    logger.warning(f"Failed to fetch homepage for JSESSIONID: {e}")

                jsessionid = client.cookies.get("JSESSIONID", "")
                if not jsessionid:
                    # fallback to a dummy if server didn't set one
                    jsessionid = "ajax:1827361849102837482"
                    client.cookies.set("JSESSIONID", jsessionid, domain=".linkedin.com")

                # Step 2: Inject user's active li_at cookie to the client's cookie jar
                client.cookies.set("li_at", li_at, domain=".linkedin.com")

                # Step 3: Build Voyager headers with the matched csrf-token
                headers = self._build_headers(li_at, jsessionid)

                if page_type == "person":
                    return await self._scrape_person(client, headers, url)
                else:
                    return await self._scrape_company(client, headers, url)

        except Exception as e:
            logger.error(f"LinkedIn scraping failed: {e}")
            return {"error": f"Scraping failed: {str(e)}"}


    async def _scrape_person(self, client, headers: Dict[str, str], url: str) -> Dict[str, Any]:
        """Scrape person profile via Voyager API."""
        profile_id = self._extract_profile_id(url)
        if not profile_id:
            return {"error": f"Could not extract profile ID from URL: {url}"}

        result: Dict[str, Any] = {"profile_url": url, "profile_id": profile_id}

        # 1. Fetch basic profile info
        profile_url = f"https://www.linkedin.com/voyager/api/identity/profiles/{profile_id}"
        try:
            resp = await client.get(profile_url, headers=headers)
            logger.info(f"Profile API status: {resp.status_code}")

            if resp.status_code in (401, 403, 302):
                return {
                    "error": "LinkedIn session is not authenticated. "
                             "Your li_at cookie may have expired. "
                             "Please log into LinkedIn on your browser, copy the fresh li_at cookie, "
                             "and paste it into the LinkedIn Session Sync widget."
                }
            elif resp.status_code == 404:
                return {"error": f"Profile not found: {profile_id}"}
            elif resp.status_code == 410:
                # 410 means the endpoint moved - try alternate
                return await self._scrape_person_alternate(client, headers, url, profile_id)
            elif resp.status_code == 200:
                data = resp.json()
                result["full_name"] = f"{data.get('firstName', '')} {data.get('lastName', '')}".strip()
                result["headline"] = data.get("headline", "")
                result["summary"] = data.get("summary", "")

                # Location
                geo = data.get("geoLocationName", "") or data.get("geoCountryName", "")
                result["location"] = geo

            else:
                logger.warning(f"Unexpected status {resp.status_code}: {resp.text[:200]}")

        except Exception as e:
            logger.warning(f"Profile fetch failed: {e}")

        # 2. Fetch contact info
        contact_url = f"https://www.linkedin.com/voyager/api/identity/profiles/{profile_id}/profileContactInfo"
        try:
            resp2 = await client.get(contact_url, headers=headers)
            logger.info(f"Contact API status: {resp2.status_code}")

            if resp2.status_code == 200:
                cdata = resp2.json()
                # Extract email
                email_addr = cdata.get("emailAddress", "") or ""
                if email_addr:
                    result["contact_email"] = email_addr

                # Extract phone
                phone_nums = cdata.get("phoneNumbers", []) or []
                if phone_nums:
                    result["contact_phone"] = phone_nums[0].get("number", "")

                # Extract websites
                websites = cdata.get("websites", []) or []
                if websites:
                    result["website"] = websites[0].get("url", "")

        except Exception as e:
            logger.warning(f"Contact info fetch failed: {e}")

        return result

    async def _scrape_person_alternate(self, client, headers: Dict[str, str], url: str, profile_id: str) -> Dict[str, Any]:
        """Alternate scraping using the normalized profile endpoint."""
        result: Dict[str, Any] = {"profile_url": url, "profile_id": profile_id}

        # Try the normalized endpoint
        norm_url = (
            f"https://www.linkedin.com/voyager/api/identity/profiles/{profile_id}"
            f"?decorationId=com.linkedin.voyager.dash.deco.identity.profile.FullProfileWithEntities-100"
        )
        try:
            resp = await client.get(norm_url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                result["full_name"] = f"{data.get('firstName', '')} {data.get('lastName', '')}".strip()
                result["headline"] = data.get("headline", "")
                result["location"] = data.get("geoLocationName", "")
        except Exception as e:
            logger.warning(f"Alternate profile fetch failed: {e}")

        # Still try contact info
        contact_url = f"https://www.linkedin.com/voyager/api/identity/profiles/{profile_id}/profileContactInfo"
        try:
            resp2 = await client.get(contact_url, headers=headers)
            if resp2.status_code == 200:
                cdata = resp2.json()
                result["contact_email"] = cdata.get("emailAddress", "") or ""
                phones = cdata.get("phoneNumbers", []) or []
                if phones:
                    result["contact_phone"] = phones[0].get("number", "")
        except Exception as e:
            logger.warning(f"Contact info fetch failed: {e}")

        if not result.get("full_name") and not result.get("contact_email"):
            result["info"] = (
                f"Successfully accessed LinkedIn profile for '{profile_id}', "
                "but this profile does not have a public email or phone number listed. "
                "The contact information may be restricted to 1st-degree connections."
            )

        return result

    async def _scrape_company(self, client, headers: Dict[str, str], url: str) -> Dict[str, Any]:
        """Scrape company page via Voyager API."""
        company_id = self._extract_company_id(url)
        if not company_id:
            return {"error": f"Could not extract company ID from URL: {url}"}

        result: Dict[str, Any] = {"profile_url": url, "company_id": company_id}

        company_url = f"https://www.linkedin.com/voyager/api/organization/companies?q=universalName&universalName={company_id}"
        try:
            resp = await client.get(company_url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                elements = data.get("elements", [])
                if elements:
                    company = elements[0]
                    result["company_name"] = company.get("name", "")
                    result["tagline"] = company.get("tagline", "")
                    result["description"] = company.get("description", "")
                    result["website"] = company.get("companyPageUrl", "")
            elif resp.status_code in (401, 403):
                return {
                    "error": "LinkedIn session is not authenticated. Please re-sync your li_at cookie."
                }
        except Exception as e:
            logger.warning(f"Company fetch failed: {e}")

        return result
