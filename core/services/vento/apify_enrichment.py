from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import httpx

from core.services.vento.discovery import is_social_or_directory_url, normalize_social_handle
from core.services.vento.verification import parse_follower_count


_EMAIL_RE = re.compile(r"[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
_BAD_EMAIL_LOCALS = frozenset(
    {
        "admin",
        "ads",
        "advertise",
        "advertising",
        "circulation",
        "corrections",
        "editor",
        "editorial",
        "help",
        "legal",
        "news",
        "newsroom",
        "no-reply",
        "noreply",
        "press",
        "privacy",
        "subscribe",
        "subscriptions",
        "support",
        "tips",
        "webmaster",
    }
)
_BAD_EMAIL_DOMAINS = frozenset(
    {
        "abc15.com",
        "apify.com",
        "cbs.com",
        "cbsnews.com",
        "collabstr.com",
        "denverlifemagazine.com",
        "denverpost.com",
        "facebook.com",
        "favikon.com",
        "feedspot.com",
        "greenwichtime.com",
        "heepsy.com",
        "influencers.club",
        "instagram.com",
        "insense.pro",
        "meta.com",
        "modash.io",
        "patch.com",
        "petfinder.com",
        "popularpays.com",
        "scni.com",
        "sfgate.com",
        "socialcat.com",
        "tiktok.com",
    }
)
_GENERIC_DISPLAY_NAMES = frozenset({"instagram", "tiktok", "user", "profile", "account", "creator", "influencer"})


@dataclass
class ApifyProfileRecord:
    platform: str
    handle: str
    profile_url: str = ""
    display_name: str = ""
    biography: str = ""
    email: str = ""
    website: str = ""
    follower_count: Optional[int] = None
    platform_follower_count: Optional[int] = None
    following_count: Optional[int] = None
    post_count: Optional[int] = None
    verified: Optional[bool] = None
    observed_at: str = ""
    source_actor: str = ""
    source_status: str = "unknown"
    raw: Dict[str, Any] = field(default_factory=dict)


class ApifyCreatorEnricher:
    """Enrich Vento creator leads with Apify Instagram/TikTok profile data.

    The adapter is deliberately optional: without a token it returns the input
    leads unchanged. Actor-specific response shapes are normalized here so the
    Plan 3 pipeline can stay provider-agnostic.
    """

    def __init__(
        self,
        *,
        client: Optional[httpx.AsyncClient] = None,
        token: Optional[str] = None,
        instagram_actor: Optional[str] = None,
        tiktok_actor: Optional[str] = None,
        cache_path: Optional[Path | str] = None,
        cache_ttl_days: Optional[int] = None,
        batch_size: Optional[int] = None,
        max_profiles_per_run: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
    ) -> None:
        self.token = (token if token is not None else os.getenv("APIFY_API_TOKEN") or os.getenv("APIFY_TOKEN") or "").strip()
        self.instagram_actor = (
            instagram_actor
            or os.getenv("APIFY_INSTAGRAM_PROFILE_ACTOR", "").strip()
            or "apify/instagram-scraper"
        )
        self.tiktok_actor = (
            tiktok_actor
            or os.getenv("APIFY_TIKTOK_PROFILE_ACTOR", "").strip()
            or "clockworks/tiktok-scraper"
        )
        default_cache = Path("data/vento/apify_profile_cache.json")
        self.cache_path = Path(cache_path or os.getenv("VENTO_APIFY_CACHE_PATH", "").strip() or default_cache)
        self.cache_ttl_days = self._clamp_int(
            cache_ttl_days if cache_ttl_days is not None else os.getenv("VENTO_APIFY_CACHE_TTL_DAYS", "7"),
            default=7,
            minimum=0,
            maximum=90,
        )
        self.batch_size = self._clamp_int(
            batch_size if batch_size is not None else os.getenv("VENTO_APIFY_BATCH_SIZE", "8"),
            default=8,
            minimum=1,
            maximum=100,
        )
        self.max_profiles_per_run = self._clamp_int(
            max_profiles_per_run if max_profiles_per_run is not None else os.getenv("VENTO_APIFY_MAX_PROFILES_PER_RUN", "15"),
            default=15,
            minimum=1,
            maximum=1_000,
        )
        self.timeout_seconds = self._clamp_float(
            timeout_seconds if timeout_seconds is not None else os.getenv("VENTO_APIFY_TIMEOUT_SECONDS", "45"),
            default=45.0,
            minimum=5.0,
            maximum=180.0,
        )
        self._client = client

    @property
    def configured(self) -> bool:
        return bool(self.token)

    async def enrich_profiles(
        self,
        leads: Sequence[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        stats = self._empty_stats()
        copied = [dict(lead) for lead in leads]
        if not copied:
            return copied, stats
        if not self.configured:
            return copied, stats

        stats["apify_enabled"] = 1
        cache = self._load_cache()
        targets_by_platform = self._collect_targets(copied)
        records: Dict[Tuple[str, str], ApifyProfileRecord] = {}
        fetch_targets: Dict[str, List[str]] = {"instagram": [], "tiktok": []}

        for platform, handles in targets_by_platform.items():
            for handle in handles:
                key = self._cache_key(platform, handle)
                cached = self._record_from_cache(cache.get(key))
                if cached and self._is_cache_fresh(cached):
                    stats["apify_cache_hits"] += 1
                    records[(platform, handle)] = cached
                else:
                    stats["apify_cache_misses"] += 1
                    fetch_targets[platform].append(handle)

        remaining_budget = self.max_profiles_per_run
        owned_client = self._client is None
        timeout = httpx.Timeout(self.timeout_seconds, connect=min(10.0, self.timeout_seconds))
        client = self._client or httpx.AsyncClient(timeout=timeout)
        try:
            for platform in ("instagram", "tiktok"):
                actor_id = self.instagram_actor if platform == "instagram" else self.tiktok_actor
                handles = fetch_targets[platform]
                if remaining_budget <= 0:
                    stats["apify_profiles_skipped_by_budget"] += len(handles)
                    continue
                selected = handles[:remaining_budget]
                stats["apify_profiles_skipped_by_budget"] += max(0, len(handles) - len(selected))
                remaining_budget -= len(selected)
                platform_records, platform_stats = await self._fetch_platform_records(
                    client,
                    platform=platform,
                    actor_id=actor_id,
                    handles=selected,
                )
                self._merge_stats(stats, platform_stats)
                for record in platform_records:
                    if not record.handle:
                        continue
                    records[(record.platform, record.handle)] = record
                    cache[self._cache_key(record.platform, record.handle)] = asdict(record)

                found_handles = {record.handle for record in platform_records}
                for handle in selected:
                    if handle in found_handles:
                        continue
                    not_found = ApifyProfileRecord(
                        platform=platform,
                        handle=handle,
                        observed_at=self._now(),
                        source_actor=actor_id,
                        source_status="not_found",
                    )
                    records[(platform, handle)] = not_found
                    cache[self._cache_key(platform, handle)] = asdict(not_found)
        finally:
            if owned_client:
                await client.aclose()

        self._save_cache(cache)
        enriched = [self._apply_records_to_lead(lead, records, stats) for lead in copied]
        return enriched, stats

    @classmethod
    def _empty_stats(cls) -> Dict[str, int]:
        return {
            "apify_enabled": 0,
            "apify_profile_requests": 0,
            "apify_profiles_enriched": 0,
            "apify_cache_hits": 0,
            "apify_cache_misses": 0,
            "apify_profiles_skipped_by_budget": 0,
            "apify_followers_found": 0,
            "apify_emails_found": 0,
            "apify_platform_mismatches_dropped": 0,
            "apify_failures": 0,
        }

    @staticmethod
    def _clamp_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
        try:
            return max(minimum, min(maximum, int(value)))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _clamp_float(value: Any, *, default: float, minimum: float, maximum: float) -> float:
        try:
            return max(minimum, min(maximum, float(value)))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @classmethod
    def _merge_stats(cls, target: Dict[str, int], source: Dict[str, int]) -> None:
        for key, value in source.items():
            if isinstance(value, int):
                target[key] = int(target.get(key, 0)) + value

    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        try:
            if not self.cache_path.is_file():
                return {}
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_cache(self, cache: Dict[str, Dict[str, Any]]) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        except Exception:
            pass

    @staticmethod
    def _cache_key(platform: str, handle: str) -> str:
        return f"{platform}:{handle.lower().lstrip('@')}"

    @classmethod
    def _record_from_cache(cls, item: Any) -> Optional[ApifyProfileRecord]:
        if not isinstance(item, dict):
            return None
        try:
            return ApifyProfileRecord(**{key: value for key, value in item.items() if key in ApifyProfileRecord.__dataclass_fields__})
        except TypeError:
            return None

    def _is_cache_fresh(self, record: ApifyProfileRecord) -> bool:
        if self.cache_ttl_days <= 0:
            return False
        try:
            observed_at = datetime.fromisoformat(str(record.observed_at or "").replace("Z", "+00:00"))
            if observed_at.tzinfo is None:
                observed_at = observed_at.replace(tzinfo=timezone.utc)
        except ValueError:
            return False
        return datetime.now(timezone.utc) - observed_at <= timedelta(days=self.cache_ttl_days)

    @classmethod
    def _collect_targets(cls, leads: Sequence[Dict[str, Any]]) -> Dict[str, List[str]]:
        output: Dict[str, List[str]] = {"instagram": [], "tiktok": []}
        seen: set[Tuple[str, str]] = set()
        for lead in leads:
            for platform in ("instagram", "tiktok"):
                handle = cls._lead_handle(lead, platform)
                if not handle:
                    continue
                key = (platform, handle)
                if key in seen:
                    continue
                seen.add(key)
                output[platform].append(handle)
        return output

    @staticmethod
    def _lead_handle(lead: Dict[str, Any], platform: str) -> str:
        raw = (
            lead.get(f"{platform}_handle")
            or lead.get(f"{platform}_url")
            or lead.get(platform)
            or lead.get(platform.title())
            or ""
        )
        return normalize_social_handle(str(raw), platform)

    async def _fetch_platform_records(
        self,
        client: httpx.AsyncClient,
        *,
        platform: str,
        actor_id: str,
        handles: Sequence[str],
    ) -> Tuple[List[ApifyProfileRecord], Dict[str, int]]:
        stats = self._empty_stats()
        records: List[ApifyProfileRecord] = []
        if not handles:
            return records, stats

        for batch in self._chunks(list(handles), self.batch_size):
            payload = self._actor_payload(platform, batch)
            response_items: List[Dict[str, Any]] = []
            for attempt in range(2):
                try:
                    stats["apify_profile_requests"] += 1
                    response = await client.post(
                        self._actor_url(actor_id),
                        params={"token": self.token},
                        json=payload,
                    )
                    if response.status_code == 429:
                        stats["apify_failures"] += 1
                        return records, stats
                    response.raise_for_status()
                    response_items = self._response_items(response.json())
                    break
                except Exception:
                    stats["apify_failures"] += 1
                    if attempt == 1:
                        response_items = []
            normalized = self._normalize_actor_items(
                response_items,
                platform=platform,
                actor_id=actor_id,
                requested_handles=batch,
            )
            records.extend(normalized)
        return records, stats

    @staticmethod
    def _chunks(items: Sequence[str], size: int) -> Iterable[List[str]]:
        for index in range(0, len(items), max(1, size)):
            yield list(items[index:index + size])

    @staticmethod
    def _actor_url(actor_id: str) -> str:
        safe_actor_id = str(actor_id or "").strip().replace("/", "~")
        return f"https://api.apify.com/v2/acts/{safe_actor_id}/run-sync-get-dataset-items"

    @staticmethod
    def _actor_payload(platform: str, handles: Sequence[str]) -> Dict[str, Any]:
        if platform == "instagram":
            return {
                "directUrls": [f"https://www.instagram.com/{handle}/" for handle in handles],
                "resultsType": "details",
                "resultsLimit": len(handles),
                "searchType": "user",
            }
        return {
            "profiles": list(handles),
            "resultsPerPage": 1,
            "shouldDownloadVideos": False,
        }

    @staticmethod
    def _response_items(data: Any) -> List[Dict[str, Any]]:
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        if isinstance(data, dict):
            for key in ("items", "results", "data", "records"):
                value = data.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
        return []

    def _normalize_actor_items(
        self,
        items: Sequence[Dict[str, Any]],
        *,
        platform: str,
        actor_id: str,
        requested_handles: Sequence[str],
    ) -> List[ApifyProfileRecord]:
        records: List[ApifyProfileRecord] = []
        single_requested = requested_handles[0] if len(requested_handles) == 1 else ""
        for item in items:
            record = self._normalize_item(item, platform=platform, actor_id=actor_id)
            if not record.handle and single_requested:
                record.handle = single_requested
                record.profile_url = record.profile_url or self._profile_url(platform, single_requested)
            if not record.handle:
                continue
            record.handle = normalize_social_handle(record.handle, platform)
            if not record.handle:
                continue
            records.append(record)
        return records

    def _normalize_item(self, item: Dict[str, Any], *, platform: str, actor_id: str) -> ApifyProfileRecord:
        handle = self._extract_handle(item, platform)
        follower_count = parse_follower_count(
            self._first_path(
                item,
                (
                    "followersCount", "followers", "followerCount", "edge_followed_by.count",
                    "authorMeta.fans", "authorMeta.followers", "stats.followerCount",
                    "stats.followers", "user.stats.followerCount",
                ),
            )
        )
        email = self._clean_email(
            self._first_path(item, ("email", "businessEmail", "publicEmail", "contactEmail", "emails.0"))
        )
        if not email:
            email = self._first_good_email(
                " ".join(str(self._first_path(item, path) or "") for path in ("biography", "bio", "signature", "authorMeta.signature", "user.signature"))
            )
        website = self._clean_website(self._first_path(item, ("externalUrl", "website", "bioLink", "link", "url")))
        profile_url = self._profile_url(platform, handle)
        explicit_url = str(self._first_path(item, ("profileUrl", "webVideoUrl", "url")) or "").strip()
        if explicit_url and (("instagram.com" in explicit_url and platform == "instagram") or ("tiktok.com" in explicit_url and platform == "tiktok")):
            profile_url = explicit_url
        return ApifyProfileRecord(
            platform=platform,
            handle=handle,
            profile_url=profile_url,
            display_name=self._clean_display_name(
                self._first_path(
                    item,
                    ("fullName", "full_name", "name", "nickname", "nickName", "authorMeta.nickName", "authorMeta.name", "user.nickname"),
                )
            ),
            biography=str(self._first_path(item, ("biography", "bio", "signature", "authorMeta.signature", "user.signature")) or "").strip(),
            email=email,
            website=website,
            follower_count=follower_count,
            platform_follower_count=follower_count,
            following_count=parse_follower_count(self._first_path(item, ("followsCount", "following", "authorMeta.following"))),
            post_count=parse_follower_count(self._first_path(item, ("postsCount", "posts", "videoCount", "authorMeta.video"))),
            verified=self._bool_or_none(self._first_path(item, ("isVerified", "verified", "authorMeta.verified", "user.verified"))),
            observed_at=self._now(),
            source_actor=actor_id,
            source_status="apify_verified" if follower_count is not None else "apify_profile_found",
            raw=item,
        )

    @classmethod
    def _extract_handle(cls, item: Dict[str, Any], platform: str) -> str:
        candidates = (
            "username", "userName", "handle", "shortcode", "name", "authorMeta.name",
            "uniqueId", "user.uniqueId", "profileUrl", "url", "webVideoUrl",
        )
        for path in candidates:
            value = cls._first_path(item, (path,))
            if value in (None, ""):
                continue
            handle = normalize_social_handle(str(value), platform)
            if handle:
                return handle
        return ""

    @staticmethod
    def _first_path(data: Dict[str, Any], paths: Sequence[str]) -> Any:
        for path in paths:
            current: Any = data
            for part in str(path).split("."):
                if isinstance(current, dict):
                    current = current.get(part)
                elif isinstance(current, list) and part.isdigit():
                    index = int(part)
                    current = current[index] if 0 <= index < len(current) else None
                else:
                    current = None
                if current is None:
                    break
            if current not in (None, ""):
                return current
        return None

    @staticmethod
    def _profile_url(platform: str, handle: str) -> str:
        clean = str(handle or "").strip().lstrip("@")
        if not clean:
            return ""
        return f"https://www.instagram.com/{clean}/" if platform == "instagram" else f"https://www.tiktok.com/@{clean}"

    @classmethod
    def _clean_display_name(cls, value: Any) -> str:
        text = " ".join(str(value or "").split()).strip(" \t\r\n|-")
        if not text or len(text) > 120:
            return ""
        if cls._name_key(text) in _GENERIC_DISPLAY_NAMES:
            return ""
        return text

    @classmethod
    def _clean_email(cls, value: Any) -> str:
        if isinstance(value, list):
            for item in value:
                email = cls._clean_email(item)
                if email:
                    return email
            return ""
        match = _EMAIL_RE.search(str(value or ""))
        if not match:
            return ""
        email = match.group(0).lower().strip(".,;:()[]{}")
        local, domain = email.rsplit("@", 1)
        local_key = re.sub(r"[^a-z0-9]", "", local)
        if (
            any(domain == bad or domain.endswith(f".{bad}") for bad in _BAD_EMAIL_DOMAINS)
            or local in _BAD_EMAIL_LOCALS
            or local_key in _BAD_EMAIL_LOCALS
            or local_key.startswith(("donotreply", "noreply", "noresponse"))
        ):
            return ""
        return email

    @classmethod
    def _first_good_email(cls, text: str) -> str:
        for match in _EMAIL_RE.findall(str(text or "")):
            email = cls._clean_email(match)
            if email:
                return email
        return ""

    @staticmethod
    def _clean_website(value: Any) -> str:
        text = str(value or "").strip()
        if not text or "@" in text:
            return ""
        if not re.match(r"https?://", text, re.I):
            text = f"https://{text}"
        host = urlparse(text).netloc.lower().removeprefix("www.")
        if not host or is_social_or_directory_url(text):
            return ""
        return text

    @staticmethod
    def _bool_or_none(value: Any) -> Optional[bool]:
        if isinstance(value, bool):
            return value
        if str(value).strip().lower() in {"true", "1", "yes"}:
            return True
        if str(value).strip().lower() in {"false", "0", "no"}:
            return False
        return None

    def _apply_records_to_lead(
        self,
        lead: Dict[str, Any],
        records: Dict[Tuple[str, str], ApifyProfileRecord],
        stats: Dict[str, int],
    ) -> Dict[str, Any]:
        output = dict(lead)
        ig_handle = self._lead_handle(output, "instagram")
        tt_handle = self._lead_handle(output, "tiktok")
        instagram = records.get(("instagram", ig_handle)) if ig_handle else None
        tiktok = records.get(("tiktok", tt_handle)) if tt_handle else None
        instagram = instagram if instagram and instagram.source_status != "not_found" else None
        tiktok = tiktok if tiktok and tiktok.source_status != "not_found" else None

        if instagram and tiktok and not self._same_creator_socials(output, instagram, tiktok):
            keep = self._preferred_platform_for_mismatch(output, instagram, tiktok)
            if keep == "instagram":
                output["tiktok_handle"] = ""
                output["tiktok_url"] = ""
                output["platform_mismatch_dropped"] = "tiktok"
                tiktok = None
            else:
                output["instagram_handle"] = ""
                output["instagram_url"] = ""
                output["platform_mismatch_dropped"] = "instagram"
                instagram = None
            stats["apify_platform_mismatches_dropped"] += 1

        applied = False
        for record in (instagram, tiktok):
            if not record:
                continue
            if self._apply_record(output, record, stats):
                applied = True
        if applied:
            self._sync_best_follower_count(output)
            stats["apify_profiles_enriched"] += 1
        return output

    def _apply_record(self, lead: Dict[str, Any], record: ApifyProfileRecord, stats: Dict[str, int]) -> bool:
        changed = False
        platform = record.platform
        formatted_handle = f"@{record.handle}" if record.handle else ""
        if formatted_handle and not lead.get(f"{platform}_handle"):
            lead[f"{platform}_handle"] = formatted_handle
            changed = True
        if record.profile_url and not lead.get(f"{platform}_url"):
            lead[f"{platform}_url"] = record.profile_url
            changed = True
        if record.platform_follower_count is not None:
            previous = parse_follower_count(lead.get(f"{platform}_follower_count"))
            if previous != record.platform_follower_count:
                lead[f"{platform}_follower_count"] = record.platform_follower_count
                changed = True
            stats["apify_followers_found"] += 1
        if record.email and (not lead.get("email") or not self._clean_email(lead.get("email"))):
            lead["email"] = record.email
            lead["email_source_type"] = "apify_profile"
            lead["email_source_url"] = record.profile_url
            stats["apify_emails_found"] += 1
            changed = True
        if record.website and not lead.get("website"):
            lead["website"] = record.website
            changed = True
        if record.display_name and self._better_display_name(record.display_name, lead):
            lead["creator_name"] = record.display_name
            lead["creator_name_source"] = f"{platform}_apify_profile"
            lead["creator_name_confidence"] = "high"
            changed = True
        if record.biography and "apify_bio" not in lead and len(record.biography) <= 240:
            lead["apify_bio"] = record.biography
            changed = True
        return changed

    @classmethod
    def _sync_best_follower_count(cls, lead: Dict[str, Any]) -> None:
        platform_counts = []
        for platform in ("instagram", "tiktok"):
            count = parse_follower_count(lead.get(f"{platform}_follower_count"))
            if count is not None:
                platform_counts.append((count, platform))
        if not platform_counts:
            return
        count, platform = max(platform_counts, key=lambda item: item[0])
        lead["follower_count"] = count
        lead["follower_count_status"] = "apify_verified"
        lead["follower_status"] = "apify_verified"
        lead["follower_source_platform"] = platform
        lead["follower_source_url"] = lead.get(f"{platform}_url") or cls._profile_url(platform, lead.get(f"{platform}_handle", ""))
        lead["follower_observed_at"] = datetime.now(timezone.utc).isoformat()

    @classmethod
    def _better_display_name(cls, candidate: str, lead: Dict[str, Any]) -> bool:
        current = str(lead.get("creator_name") or lead.get("name") or "").strip()
        if not candidate or cls._looks_like_handle(candidate):
            return False
        if not current or cls._looks_like_handle(current):
            return True
        current_key = cls._name_key(current)
        candidate_key = cls._name_key(candidate)
        if current_key in _GENERIC_DISPLAY_NAMES:
            return True
        if current_key and current_key in candidate_key and " " in candidate:
            return True
        return len(candidate) > len(current) + 4

    @classmethod
    def _same_creator_socials(
        cls,
        lead: Dict[str, Any],
        instagram: ApifyProfileRecord,
        tiktok: ApifyProfileRecord,
    ) -> bool:
        if cls._similar_key(instagram.handle, tiktok.handle):
            return True
        if cls._similar_key(instagram.display_name, tiktok.display_name):
            return True
        if cls._record_matches_lead(lead, instagram) and cls._record_matches_lead(lead, tiktok):
            return True
        combined = f"{instagram.biography} {tiktok.biography}".lower()
        return bool(instagram.handle and tiktok.handle and instagram.handle in combined and tiktok.handle in combined)

    @classmethod
    def _preferred_platform_for_mismatch(
        cls,
        lead: Dict[str, Any],
        instagram: ApifyProfileRecord,
        tiktok: ApifyProfileRecord,
    ) -> str:
        instagram_matches = cls._record_matches_lead(lead, instagram)
        tiktok_matches = cls._record_matches_lead(lead, tiktok)
        if instagram_matches and not tiktok_matches:
            return "instagram"
        if tiktok_matches and not instagram_matches:
            return "tiktok"
        instagram_count = instagram.platform_follower_count or -1
        tiktok_count = tiktok.platform_follower_count or -1
        return "instagram" if instagram_count >= tiktok_count else "tiktok"

    @classmethod
    def _record_matches_lead(cls, lead: Dict[str, Any], record: ApifyProfileRecord) -> bool:
        name = str(lead.get("creator_name") or lead.get("name") or "").strip()
        if cls._similar_key(name, record.display_name):
            return True
        if cls._similar_key(name, record.handle):
            return True
        return cls._token_overlap(name, record.display_name) >= 0.6

    @classmethod
    def _similar_key(cls, left: Any, right: Any) -> bool:
        a = cls._name_key(left)
        b = cls._name_key(right)
        if not a or not b:
            return False
        return a == b or (min(len(a), len(b)) >= 5 and (a in b or b in a))

    @staticmethod
    def _name_key(value: Any) -> str:
        return re.sub(r"[^a-z0-9]", "", str(value or "").lower())

    @classmethod
    def _looks_like_handle(cls, value: Any) -> bool:
        text = str(value or "").strip()
        return text.startswith("@") or cls._name_key(text) in _GENERIC_DISPLAY_NAMES

    @staticmethod
    def _token_overlap(left: Any, right: Any) -> float:
        left_tokens = {token for token in re.findall(r"[a-z0-9]+", str(left or "").lower()) if len(token) >= 3}
        right_tokens = {token for token in re.findall(r"[a-z0-9]+", str(right or "").lower()) if len(token) >= 3}
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / max(1, min(len(left_tokens), len(right_tokens)))
