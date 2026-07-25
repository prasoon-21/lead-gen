from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from core.services.vento.discovery import extract_social_handles, normalize_social_handle
from core.services.vento.identity_quality import canonicalize_creator_identity, social_url_scope
from core.services.vento.verification import parse_follower_count


_EMAIL_RE = re.compile(r"[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
_FOLLOWER_RE = re.compile(r"(?P<count>\d[\d,.]*\s*[KMB]?)\s+(?:followers?|fans?)\b", re.I)
_CACHE_FIELDS = {
    "email", "email_source_url", "email_source_type", "instagram_handle", "instagram_url", "tiktok_handle",
    "tiktok_url", "follower_count", "follower_count_status", "follower_status", "follower_source_url",
    "follower_observed_at", "location_evidence", "location_source_url", "enriched_at",
    "profile_enrichment_evidence", "enrichment_evidence_urls",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


class AdaptiveCreatorEnricher:
    """Reuse one high-intent search to fill several missing creator fields."""

    def __init__(self, *, tavily_client: Any = None, cache_path: Optional[Path] = None) -> None:
        self.tavily = tavily_client
        self.cache_path = Path(cache_path) if cache_path is not None else Path("data/vento_creator_cache.json")
        self.cache = self._load_cache()
        self.stats: Dict[str, int] = {
            "candidates_considered": 0,
            "searches_used": 0,
            "searches_avoided_by_cache": 0,
            "fields_recovered_without_extra_search": 0,
            "cross_platform_matches": 0,
            "emails_found": 0,
            "followers_found": 0,
            "locations_found": 0,
        }

    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        try:
            value = json.loads(self.cache_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError, TypeError):
            return {}

    def _save_cache(self) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
            temp.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8")
            temp.replace(self.cache_path)
        except OSError:
            return

    @staticmethod
    def _cache_key(lead: Dict[str, Any]) -> str:
        for platform, field in (("instagram", "instagram_handle"), ("tiktok", "tiktok_handle")):
            handle = normalize_social_handle(str(lead.get(field) or ""), platform)
            if handle:
                return f"{platform}:{handle}"
        return ""

    @staticmethod
    def _missing_fields(
        lead: Dict[str, Any],
        *,
        enrich_cross_platform: bool = True,
        enrich_email: bool = True,
        enrich_followers: bool = True,
    ) -> list[str]:
        missing: list[str] = []
        if enrich_cross_platform and not (lead.get("instagram_handle") and lead.get("tiktok_handle")):
            missing.append("second_platform")
        if enrich_email and not lead.get("email"):
            missing.append("email")
        if enrich_followers and lead.get("follower_count") in (None, ""):
            missing.append("followers")
        if not lead.get("location") and not lead.get("location_evidence"):
            missing.append("location")
        return missing

    @staticmethod
    def _merge_cached(lead: Dict[str, Any], cached: Dict[str, Any]) -> int:
        recovered = 0
        for field in _CACHE_FIELDS:
            value = cached.get(field)
            if value not in (None, "", [], {}) and lead.get(field) in (None, "", [], {}):
                lead[field] = value
                recovered += 1
        return recovered

    def _query(self, lead: Dict[str, Any], missing: Iterable[str], target_location: str) -> str:
        handle = str(lead.get("instagram_handle") or lead.get("tiktok_handle") or "").strip()
        name = _clean(lead.get("creator_name"))
        wants = " ".join(missing)
        target_hint = f' "{target_location}"' if target_location else ""
        return f'"{name or handle}" "{handle}" Instagram TikTok business email followers based location {wants}{target_hint}'[:390]

    async def _search(self, query: str) -> Dict[str, Any]:
        if not self.tavily:
            return {"results": []}
        try:
            return await self.tavily.search(
                {"query": query, "search_depth": "basic", "max_results": 10, "include_raw_content": False}
            )
        except Exception:
            return {"results": []}

    @staticmethod
    def _identity_matches(lead: Dict[str, Any], text: str) -> bool:
        lower = text.lower()
        handles = [
            normalize_social_handle(str(lead.get("instagram_handle") or ""), "instagram"),
            normalize_social_handle(str(lead.get("tiktok_handle") or ""), "tiktok"),
        ]
        if any(handle and handle in lower for handle in handles):
            return True
        name = _clean(lead.get("creator_name")).lower()
        return bool(name and len(name) >= 4 and name in lower)

    def _apply_results(
        self,
        lead: Dict[str, Any],
        results: Iterable[Dict[str, Any]],
        *,
        target_location: str,
        enrich_cross_platform: bool,
        enrich_email: bool,
        enrich_followers: bool,
    ) -> None:
        combined_parts: list[str] = []
        profile_parts: list[str] = []
        evidence_urls: list[str] = []
        for item in results:
            url = _clean(item.get("url"))
            text = " ".join((_clean(item.get("title")), _clean(item.get("content")), url))
            if not self._identity_matches(lead, text):
                continue
            combined_parts.append(text)
            if social_url_scope(url) == "profile":
                profile_parts.append(text)
            if url:
                evidence_urls.append(url)
        if not combined_parts:
            return
        combined = " ".join(combined_parts)
        if profile_parts:
            lead["profile_enrichment_evidence"] = " ".join(profile_parts)[:3_000]
        lead["enrichment_evidence_urls"] = list(dict.fromkeys(evidence_urls))
        handles = extract_social_handles(combined, *evidence_urls)

        before_instagram = bool(lead.get("instagram_handle"))
        before_tiktok = bool(lead.get("tiktok_handle"))
        if enrich_cross_platform and not before_instagram and handles.get("instagram"):
            lead["instagram_handle"] = f"@{handles['instagram'][0]}"
        if enrich_cross_platform and not before_tiktok and handles.get("tiktok"):
            lead["tiktok_handle"] = f"@{handles['tiktok'][0]}"
        lead.update(canonicalize_creator_identity(lead))
        if (not before_instagram and lead.get("instagram_handle")) or (not before_tiktok and lead.get("tiktok_handle")):
            self.stats["cross_platform_matches"] += 1

        if enrich_email and not lead.get("email"):
            email_match = _EMAIL_RE.search(combined)
            if email_match and "****" not in email_match.group(0):
                lead["email"] = email_match.group(0).lower()
                lead["email_source_url"] = evidence_urls[0] if evidence_urls else ""
                lead["email_source_type"] = "search_observed"
                self.stats["emails_found"] += 1

        if enrich_followers and lead.get("follower_count") in (None, ""):
            follower_match = _FOLLOWER_RE.search(combined)
            if follower_match:
                count = parse_follower_count(follower_match.group("count"))
                if count is not None:
                    lead["follower_count"] = count
                    lead["follower_count_status"] = "search_observed"
                    lead["follower_status"] = "search_observed"
                    lead["follower_source_url"] = evidence_urls[0] if evidence_urls else ""
                    lead["follower_observed_at"] = _now()
                    self.stats["followers_found"] += 1

        if not lead.get("location") and not lead.get("location_evidence"):
            target_terms = [part.strip() for part in re.split(r"[,/]", target_location) if len(part.strip()) >= 2]
            if any(term.lower() in combined.lower() for term in target_terms):
                lead["location_evidence"] = combined[:800]
                lead["location_source_url"] = evidence_urls[0] if evidence_urls else ""
                self.stats["locations_found"] += 1

    async def enrich(
        self,
        leads: Iterable[Dict[str, Any]],
        *,
        target_location: str = "",
        enrich_cross_platform: bool = True,
        enrich_email: bool = True,
        enrich_followers: bool = True,
    ) -> list[Dict[str, Any]]:
        results: list[Dict[str, Any]] = []
        cache_changed = False
        for raw in leads:
            lead = canonicalize_creator_identity(dict(raw))
            self.stats["candidates_considered"] += 1
            key = self._cache_key(lead)
            if key and key in self.cache:
                recovered = self._merge_cached(lead, self.cache[key])
                if recovered:
                    self.stats["fields_recovered_without_extra_search"] += recovered

            missing = self._missing_fields(
                lead,
                enrich_cross_platform=enrich_cross_platform,
                enrich_email=enrich_email,
                enrich_followers=enrich_followers,
            )
            if key and key in self.cache and not missing:
                self.stats["searches_avoided_by_cache"] += 1
            promising = (
                bool(lead.get("canonical_social_present"))
                and bool(lead.get("individual_creator"))
                and int(lead.get("product_fit_score") or 0) >= 55
            )
            if missing and promising and self.tavily:
                query = self._query(lead, missing, target_location)
                data = await self._search(query)
                self.stats["searches_used"] += 1
                self._apply_results(
                    lead,
                    data.get("results") or [],
                    target_location=target_location,
                    enrich_cross_platform=enrich_cross_platform,
                    enrich_email=enrich_email,
                    enrich_followers=enrich_followers,
                )

            lead["enriched_at"] = _now()
            if key:
                cached = {field: lead.get(field) for field in _CACHE_FIELDS if lead.get(field) not in (None, "", [], {})}
                if cached:
                    self.cache[key] = cached
                    cache_changed = True
            results.append(lead)
        if cache_changed:
            self._save_cache()
        return results
