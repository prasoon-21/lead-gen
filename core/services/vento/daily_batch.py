from __future__ import annotations

import asyncio
import inspect
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from core.services.vento.qualification import creator_identity_keys
from core.services.vento.vento_lead_pipeline import VentoLeadPipeline


class VentoDailyBatchRunner:
    """Run and optionally persist one idempotent daily Vento lead batch."""

    def __init__(self, *, pipeline: Optional[VentoLeadPipeline] = None, todo_store: Any = None) -> None:
        self.pipeline = pipeline or VentoLeadPipeline()
        self.todo_store = todo_store

    @staticmethod
    async def _call(method: Any, *args: Any) -> Any:
        if inspect.iscoroutinefunction(method):
            return await method(*args)
        return await asyncio.to_thread(method, *args)

    async def _existing_identity_keys(self) -> Set[str]:
        if not self.todo_store or not hasattr(self.todo_store, "list_vento_leads"):
            return set()
        try:
            rows = await self._call(self.todo_store.list_vento_leads)
        except Exception:
            return set()
        keys: Set[str] = set()
        for row in rows or []:
            normalized = {
                "creator_id": row.get("Creator ID") or row.get("creator_id"),
                "instagram_handle": row.get("Instagram Handle") or row.get("instagram_handle"),
                "tiktok_handle": row.get("TikTok Handle") or row.get("tiktok_handle"),
                "youtube_channel": row.get("YouTube Channel") or row.get("youtube_channel"),
                "email": row.get("Email") or row.get("email"),
                "website": row.get("Website") or row.get("website"),
                "creator_name": row.get("Creator Name") or row.get("creator_name"),
                "do_not_contact": row.get("Do Not Contact") or row.get("do_not_contact"),
            }
            keys.update(creator_identity_keys(normalized))
        return keys

    async def _load_idempotent_result(self, idempotency_key: str) -> Optional[Dict[str, Any]]:
        if not idempotency_key or not self.todo_store or not hasattr(self.todo_store, "get_vento_daily_run"):
            return None
        try:
            existing = await self._call(self.todo_store.get_vento_daily_run, idempotency_key)
        except Exception:
            return None
        if not existing:
            return None
        return {
            "batch_id": idempotency_key,
            "status": "already_completed",
            "accepted_leads": [],
            "review_leads": [],
            "daily_target": int(existing.get("Daily Usable Target") or 0),
            "accepted_count": int(existing.get("Accepted Count") or 0),
            "shortfall": int(existing.get("Shortfall") or 0),
            "metrics": existing,
            "idempotent_replay": True,
        }

    async def run(
        self,
        *,
        location: str,
        daily_target: int = 100,
        raw_target: Optional[int] = None,
        influencer_type: str = "all",
        follower_tier: str = "all",
        strict_us_only: bool = True,
        bulk_data: Optional[str] = None,
        bulk_file_path: Optional[str] = None,
        idempotency_key: str = "",
        persist: bool = True,
        quality_mode: str = "strict",
        minimum_product_fit: str = "P3",
        include_paid_ugc: bool = True,
        enrich_cross_platform: bool = True,
        enrich_email: bool = True,
        enrich_followers: bool = True,
        require_email: bool = False,
        require_followers: bool = False,
    ) -> Dict[str, Any]:
        daily_target = max(1, min(int(daily_target or 100), 1000))
        raw_target = max(daily_target, min(int(raw_target), 1000)) if raw_target is not None else None
        replay = await self._load_idempotent_result(idempotency_key)
        if replay:
            return replay

        started = time.perf_counter()
        historical_keys = await self._existing_identity_keys()
        pipeline_result = await self.pipeline.run(
            location=location,
            influencer_type=influencer_type,
            follower_tier=follower_tier,
            target_count=daily_target,
            raw_target=raw_target,
            bulk_data=bulk_data,
            bulk_file_path=bulk_file_path,
            strict_us_only=strict_us_only,
            historical_identity_keys=historical_keys,
            quality_mode=quality_mode,
            minimum_product_fit=minimum_product_fit,
            include_paid_ugc=include_paid_ugc,
            enrich_cross_platform=enrich_cross_platform,
            enrich_email=enrich_email,
            enrich_followers=enrich_followers,
            require_email=require_email,
            require_followers=require_followers,
        )
        accepted = list(pipeline_result.get("leads") or [])[:daily_target]
        review_limit = max(0, int(os.getenv("VENTO_DAILY_MAX_REVIEW_LEADS", "50")))
        review = list(pipeline_result.get("review_leads") or [])[:review_limit]
        rejected_count = len(pipeline_result.get("rejected_leads") or [])
        paid_ugc = list(pipeline_result.get("paid_ugc_leads") or [])
        batch_id = idempotency_key or str(pipeline_result.get("batch_id") or "")
        for lead in accepted + review:
            lead["import_batch_id"] = batch_id

        shortfall = max(0, daily_target - len(accepted))
        status = "completed" if shortfall == 0 else "completed_with_shortfall"
        metadata = pipeline_result.get("metadata") or {}
        duration = round(time.perf_counter() - started, 2)
        metrics = {
            "raw_discovered": int(metadata.get("total_raw_ingested") or 0),
            "unique_candidates": int(metadata.get("total_unique") or 0),
            "profile_verified": sum(1 for lead in accepted + review if lead.get("profile_verified")),
            "email_ready": sum(1 for lead in accepted if lead.get("email_verification_status") == "valid"),
            "dm_ready": sum(1 for lead in accepted if lead.get("preferred_contact_method") != "email"),
            "level_a": sum(1 for lead in accepted if lead.get("quality_level") == "A"),
            "level_b": sum(1 for lead in accepted if lead.get("quality_level") == "B"),
            "level_c": len(review),
            "rejected": rejected_count,
            "duplicates": sum((metadata.get("deduplication") or {}).get(key, 0) for key in ("same_run_duplicates", "historical_duplicates")),
            "duration_seconds": duration,
            "tavily_queries_used": int((metadata.get("tavily") or {}).get("queries_used") or 0),
            "tavily_raw_candidates": int((metadata.get("tavily") or {}).get("raw_candidates") or 0),
            "instagram_candidates": int((metadata.get("tavily") or {}).get("instagram_candidates") or 0),
            "tiktok_candidates": int((metadata.get("tavily") or {}).get("tiktok_candidates") or 0),
            "both_platforms": int((metadata.get("tavily") or {}).get("both_platforms") or 0),
            "paid_ugc": len(paid_ugc),
            "known_followers": int(metadata.get("known_followers") or 0),
            "location_confirmed": int(metadata.get("location_confirmed") or 0),
            "canonical_social_profiles": int(metadata.get("canonical_social_profiles") or 0),
            "product_fit_p1_p2": int(metadata.get("product_fit_p1_p2") or 0),
            "raw_to_usable_yield": float(metadata.get("raw_to_usable_yield") or 0),
            "usable_leads_per_tavily_search": float(metadata.get("usable_leads_per_tavily_search") or 0),
            "tavily_searches_avoided_by_cache": int((metadata.get("enrichment") or {}).get("searches_avoided_by_cache") or 0),
            "fields_recovered_without_extra_search": int((metadata.get("enrichment") or {}).get("fields_recovered_without_extra_search") or 0),
        }
        report_row = {
            "batch_id": batch_id,
            "run_date": datetime.now(timezone.utc).isoformat(),
            "target_location": location,
            "raw_target": raw_target or daily_target,
            "daily_usable_target": daily_target,
            "raw_discovered": metrics["raw_discovered"],
            "duplicates": metrics["duplicates"],
            "accepted_count": len(accepted),
            "level_a": metrics["level_a"],
            "level_b": metrics["level_b"],
            "level_c": metrics["level_c"],
            "rejected_count": rejected_count,
            "email_ready": metrics["email_ready"],
            "dm_ready": metrics["dm_ready"],
            "shortfall": shortfall,
            "source_usage": metadata.get("source_breakdown") or {},
            "duration_seconds": duration,
            "status": status,
            "error_summary": "",
        }

        storage_status = "not_configured"
        storage_error = ""
        if persist and self.todo_store:
            try:
                if hasattr(self.todo_store, "ensure_vento_store"):
                    await self._call(self.todo_store.ensure_vento_store)
                if accepted:
                    if hasattr(self.todo_store, "save_vento_leads"):
                        await self._call(self.todo_store.save_vento_leads, accepted)
                    else:
                        for lead in accepted:
                            await self._call(self.todo_store.save_vento_lead, lead)
                if review and hasattr(self.todo_store, "save_vento_review_leads"):
                    await self._call(self.todo_store.save_vento_review_leads, review)
                if hasattr(self.todo_store, "save_vento_daily_run"):
                    await self._call(self.todo_store.save_vento_daily_run, report_row)
                storage_status = "saved"
            except Exception as exc:
                storage_status = "failed"
                storage_error = f"{type(exc).__name__}: {exc}"

        return {
            "batch_id": batch_id,
            "status": status,
            "accepted_leads": accepted,
            "review_leads": review,
            "paid_ugc_leads": paid_ugc,
            "rejected_count": rejected_count,
            "duplicate_count": metrics["duplicates"],
            "daily_target": daily_target,
            "accepted_count": len(accepted),
            "shortfall": shortfall,
            "metrics": metrics,
            "steps": pipeline_result.get("steps") or [],
            "storage_status": storage_status,
            "storage_error": storage_error,
            "report": report_row,
        }
