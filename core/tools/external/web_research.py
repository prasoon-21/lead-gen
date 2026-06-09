import json
import re
from typing import Any, Dict, List
from urllib.parse import urljoin, urlparse

import httpx
from core.tools.base import BaseTool, ToolContext, ToolExecutionError
from core.tools.external.tavily_client import TavilyClient


DEFAULT_RESEARCH_SCHEMA = {
    "summary": "Concise synthesis of findings",
    "key_points": ["List of key facts"],
    "sources": [{"title": "Source title", "url": "https://...", "relevance": "why this source matters"}],
    "confidence": "low|medium|high",
    "gaps": ["What could not be verified"],
}


class WebResearchTool(BaseTool):
    name = "web_research"
    description = "Run deep web research, expand discovery, and inspect company site pages such as about, team, and contact."
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer"},
            "search_depth": {"type": "string"},
            "include_domains": {"type": "array"},
            "exclude_domains": {"type": "array"},
            "output_schema": {"type": "object"},
            "company_name": {"type": "string"},
            "website_url": {"type": "string"},
            "industry": {"type": "string"},
            "location": {"type": "string"},
            "deep_site_scan": {"type": "boolean"},
            "site_paths": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["query"],
    }
    timeout_seconds = 50
    _default_site_paths = (
        "",
        "/about",
        "/about-us",
        "/team",
        "/our-team",
        "/leadership",
        "/company",
        "/contact",
        "/contact-us",
    )
    _site_link_keywords = ("about", "team", "leadership", "contact", "company", "staff", "management", "founder")
    _excluded_research_domains = {
        "linkedin.com",
        "www.linkedin.com",
        "facebook.com",
        "www.facebook.com",
        "instagram.com",
        "www.instagram.com",
        "x.com",
        "twitter.com",
        "www.twitter.com",
        "crunchbase.com",
        "www.crunchbase.com",
        "zoominfo.com",
        "www.zoominfo.com",
        "clutch.co",
        "www.clutch.co",
        "glassdoor.com",
        "www.glassdoor.com",
        "indeed.com",
        "www.indeed.com",
        "yelp.com",
        "www.yelp.com",
        "wikipedia.org",
        "www.wikipedia.org",
    }

    @staticmethod
    def _simplify_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        simplified = []
        for item in results or []:
            simplified.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", "")[:2000],
                    "score": item.get("score"),
                    "published_date": item.get("published_date"),
                }
            )
        return simplified

    @staticmethod
    def _normalize_query(text: str) -> str:
        return " ".join((text or "").strip().split())

    @staticmethod
    def _normalize_domains(domains: List[str]) -> List[str]:
        cleaned: List[str] = []
        for domain in domains or []:
            host = (domain or "").strip().lower()
            if not host:
                continue
            host = host.replace("https://", "").replace("http://", "").strip("/")
            if host.startswith("www."):
                host = host[4:]
            if host and host not in cleaned:
                cleaned.append(host)
        return cleaned

    def _filter_results_by_domains(
        self,
        results: List[Dict[str, Any]],
        include_domains: List[str],
        exclude_domains: List[str],
    ) -> List[Dict[str, Any]]:
        include_set = set(self._normalize_domains(include_domains))
        exclude_set = set(self._normalize_domains(exclude_domains))
        if not include_set and not exclude_set:
            return results

        filtered: List[Dict[str, Any]] = []
        for item in results:
            host = (urlparse(item.get("url") or "").hostname or "").lower()
            if host.startswith("www."):
                host = host[4:]
            if exclude_set and host in exclude_set:
                continue
            if include_set and host not in include_set:
                continue
            filtered.append(item)
        return filtered

    def _build_query_variants(self, arguments: Dict[str, Any], context: ToolContext) -> List[str]:
        query = self._normalize_query(arguments.get("query", ""))
        industry = self._normalize_query(arguments.get("industry") or context.state.get("industry") or "")
        location = self._normalize_query(arguments.get("location") or context.state.get("location") or "")
        company_name = self._normalize_query(arguments.get("company_name") or context.state.get("company_name") or "")

        variants: List[str] = []

        def add_variant(text: str) -> None:
            normalized = self._normalize_query(text)
            if normalized and normalized not in variants:
                variants.append(normalized)

        add_variant(query)
        if industry and location:
            add_variant(f"{industry} companies in {location}")
            add_variant(f"{industry} companies in {location} founders")
            add_variant(f"{industry} companies in {location} contact")
        if company_name:
            add_variant(f"{company_name} official website")
            add_variant(f"{company_name} founder contact")
            add_variant(f"{company_name} about team contact")
        return variants[:5]

    @staticmethod
    def _candidate_site(url: str) -> bool:
        host = (urlparse(url).hostname or "").lower()
        return bool(host) and host not in WebResearchTool._excluded_research_domains

    def _extract_candidate_sites(
        self,
        results: List[Dict[str, Any]],
        explicit_website: str = "",
    ) -> List[str]:
        sites: List[str] = []

        def add_site(url: str) -> None:
            parsed = urlparse(url or "")
            if not parsed.scheme or not parsed.netloc:
                return
            normalized = f"{parsed.scheme}://{parsed.netloc}"
            if self._candidate_site(normalized) and normalized not in sites:
                sites.append(normalized)

        if explicit_website:
            add_site(explicit_website)

        for item in results:
            add_site(item.get("url", ""))
            if len(sites) >= 3:
                break

        return sites

    @staticmethod
    def _strip_html(html: str) -> str:
        cleaned = re.sub(r"(?is)<script.*?>.*?</script>", " ", html or "")
        cleaned = re.sub(r"(?is)<style.*?>.*?</style>", " ", cleaned)
        cleaned = re.sub(r"(?is)<[^>]+>", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    def _discover_internal_links(self, base_url: str, html: str) -> List[str]:
        links: List[str] = []
        for href in re.findall(r'href=["\']([^"\']+)["\']', html or "", flags=re.IGNORECASE):
            absolute = urljoin(base_url, href)
            parsed = urlparse(absolute)
            if not parsed.scheme.startswith("http"):
                continue
            if f"{parsed.scheme}://{parsed.netloc}" != base_url:
                continue
            path_lower = parsed.path.lower()
            if any(keyword in path_lower for keyword in self._site_link_keywords):
                normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                if normalized not in links:
                    links.append(normalized)
        return links[:6]

    async def _fetch_page(self, client: httpx.AsyncClient, url: str) -> Dict[str, Any] | None:
        try:
            response = await client.get(url, follow_redirects=True)
            if response.status_code >= 400:
                return None
            html = response.text
            title_match = re.search(r"(?is)<title[^>]*>(.*?)</title>", html)
            text = self._strip_html(html)[:3500]
            if not text:
                return None
            return {
                "url": str(response.url),
                "title": self._strip_html(title_match.group(1))[:200] if title_match else "",
                "content": text,
            }
        except Exception:
            return None

    async def _scan_company_sites(
        self,
        site_urls: List[str],
        site_paths: List[str],
    ) -> List[Dict[str, Any]]:
        if not site_urls:
            return []

        discovered_pages: List[Dict[str, Any]] = []
        seen_urls = set()

        async with httpx.AsyncClient(timeout=12) as client:
            for site_url in site_urls[:3]:
                homepage = await self._fetch_page(client, site_url)
                homepage_html = ""
                if homepage:
                    discovered_pages.append(homepage)
                    seen_urls.add(homepage["url"])
                    try:
                        resp = await client.get(site_url, follow_redirects=True)
                        homepage_html = resp.text
                    except Exception:
                        homepage_html = ""

                candidate_links = [urljoin(site_url, path) for path in site_paths]
                candidate_links.extend(self._discover_internal_links(site_url, homepage_html))

                for candidate_url in candidate_links:
                    normalized = candidate_url.rstrip("/")
                    if normalized in seen_urls:
                        continue
                    page = await self._fetch_page(client, candidate_url)
                    if not page:
                        continue
                    seen_urls.add(normalized)
                    discovered_pages.append(page)
                    if len(discovered_pages) >= 10:
                        return discovered_pages

        return discovered_pages

    def _build_synthesis_prompt(
        self,
        query: str,
        results: List[Dict[str, Any]],
        site_pages: List[Dict[str, Any]],
        output_schema: Dict[str, Any],
    ) -> str:
        packed = json.dumps(results[:8], ensure_ascii=False)
        site_pack = json.dumps(site_pages[:10], ensure_ascii=False)
        schema_text = json.dumps(output_schema, ensure_ascii=False, indent=2)
        return (
            "You are a research synthesizer. Build a factual, source-cited report.\n"
            "Use only the provided web results and website extracts. If evidence is weak, lower confidence.\n"
            "Prioritize person-level contact clues and details found on about, team, leadership, staff, and contact pages.\n"
            "Return JSON only and match this schema shape:\n"
            f"{schema_text}\n\n"
            f"Research Query: {query}\n"
            f"Web Results JSON:\n{packed}\n\n"
            f"Website Page Extracts JSON:\n{site_pack}\n\n"
            "Citation rule: Include source URLs in `sources` and reference only real URLs from input."
        )

    async def run(self, arguments: Dict[str, Any], context: ToolContext) -> Dict[str, Any]:
        query = arguments.get("query", "").strip()
        if not query:
            raise ToolExecutionError("query is required", code="invalid_arguments")

        query_variants = self._build_query_variants(arguments, context)
        max_results = int(arguments.get("max_results", 8))
        search_depth = arguments.get("search_depth", "advanced")
        include_domains = arguments.get("include_domains") or []
        exclude_domains = arguments.get("exclude_domains") or []
        research_payload_base = {
            "max_results": max_results,
            "search_depth": search_depth,
        }
        search_payload_base = {
            "max_results": max_results,
            "search_depth": search_depth,
            "include_domains": include_domains,
            "exclude_domains": exclude_domains,
        }

        client = TavilyClient()
        combined_results: List[Dict[str, Any]] = []
        for variant in query_variants:
            raw = None
            try:
                raw = await client.research({"query": variant, **research_payload_base})
            except Exception:
                raw = await client.search({"query": variant, **search_payload_base})
            variant_results = self._simplify_results(raw.get("results", []))
            variant_results = self._filter_results_by_domains(
                variant_results,
                include_domains=include_domains,
                exclude_domains=exclude_domains,
            )
            combined_results.extend(variant_results)

        deduped_results: List[Dict[str, Any]] = []
        seen_urls = set()
        for item in sorted(combined_results, key=lambda result: result.get("score") or 0, reverse=True):
            url = (item.get("url") or "").strip().lower()
            if url and url in seen_urls:
                continue
            if url:
                seen_urls.add(url)
            deduped_results.append(item)
            if len(deduped_results) >= max_results:
                break

        site_paths = arguments.get("site_paths") or list(self._default_site_paths)
        explicit_website = arguments.get("website_url") or context.state.get("company_website") or ""
        site_urls = self._extract_candidate_sites(deduped_results, explicit_website=str(explicit_website))
        deep_site_scan = bool(arguments.get("deep_site_scan", True))
        site_pages = await self._scan_company_sites(site_urls, site_paths) if deep_site_scan else []

        output_schema = arguments.get("output_schema") or DEFAULT_RESEARCH_SCHEMA
        adapter = context.resources.get("adapter")

        if not adapter:
            return {
                "report": {
                    "summary": "Research completed without synthesis model.",
                    "key_points": [],
                    "sources": [{"title": r.get("title", ""), "url": r.get("url", ""), "relevance": "search_result"} for r in deduped_results[:5]],
                    "confidence": "medium" if deduped_results else "low",
                    "gaps": [],
                },
                "results": deduped_results,
                "site_pages": site_pages,
                "candidate_websites": site_urls,
                "count": len(deduped_results),
            }

        report = await adapter.generate_json(
            prompt=self._build_synthesis_prompt(query, deduped_results, site_pages, output_schema),
            temperature=0.2,
        )
        return {
            "report": report,
            "results": deduped_results,
            "site_pages": site_pages,
            "candidate_websites": site_urls,
            "count": len(deduped_results),
            "query": query,
            "queries_used": query_variants,
        }
