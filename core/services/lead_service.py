import asyncio
import inspect
import os
from typing import Any, Dict, List, Set
from urllib.parse import urlparse

from core.services.lead_quality_service import rank_and_enrich_leads
from core.services.lead_quality_gate import apply_quality_gate
from core.services.todo_sheet_store import TodoSheetStore
from core.services.vento.verification import parse_follower_count


class LeadService:
    def __init__(self, todo_store: TodoSheetStore):
        self.todo_store = todo_store

    @staticmethod
    def _clean(value: Any) -> str:
        cleaned = " ".join(str(value or "").split()).strip()
        if cleaned.lower() in {"-", "—", "n/a", "na", "none", "null"}:
            return ""
        return cleaned

    @staticmethod
    def _domain_from_url(url: str) -> str:
        return (urlparse(url or "").hostname or "").lower().strip()

    @staticmethod
    def _domain_from_email(email: str) -> str:
        parts = (email or "").split("@", 1)
        return parts[1].lower().strip() if len(parts) == 2 else ""

    @staticmethod
    async def _call_store(method, *args):
        if inspect.iscoroutinefunction(method):
            return await method(*args)
        return await asyncio.to_thread(method, *args)

    @classmethod
    def _company_key(cls, lead: Dict[str, Any]) -> str:
        return cls._clean(lead.get("company_name") or lead.get("Company Name")).lower()

    @classmethod
    def _email_key(cls, lead: Dict[str, Any]) -> str:
        return cls._clean(lead.get("contact_email") or lead.get("email") or lead.get("Email")).lower()

    @classmethod
    def _website_key(cls, lead: Dict[str, Any]) -> str:
        return cls._clean(lead.get("company_website") or lead.get("website") or lead.get("Website"))

    @classmethod
    def _domain_key(cls, lead: Dict[str, Any]) -> str:
        domain = cls._domain_from_url(cls._website_key(lead))
        if domain:
            return domain
        return cls._domain_from_email(cls._email_key(lead))

    @classmethod
    def _linkedin_key(cls, lead: Dict[str, Any]) -> str:
        metadata = lead.get("metadata", {}) if isinstance(lead.get("metadata"), dict) else {}
        return cls._clean(
            lead.get("linkedin_url")
            or metadata.get("linkedin_url")
        ).lower()

    @classmethod
    def _person_key(cls, lead: Dict[str, Any]) -> str:
        metadata = lead.get("metadata", {}) if isinstance(lead.get("metadata"), dict) else {}
        return cls._clean(
            lead.get("contact_person_name")
            or lead.get("founder_name")
            or lead.get("contact_name")
            or lead.get("Contact Name")
            or metadata.get("contact_person_name")
        ).lower()

    @classmethod
    def _person_company_key(cls, lead: Dict[str, Any]) -> str:
        person = cls._person_key(lead)
        company = cls._company_key(lead)
        if person and company:
            return f"{person}|{company}"
        return ""

    @classmethod
    def _is_person_level_lead(cls, lead: Dict[str, Any]) -> bool:
        return bool(
            cls._person_key(lead)
            or cls._email_key(lead)
            or cls._linkedin_key(lead)
        )

    @classmethod
    def _dedup_keys(cls, lead: Dict[str, Any]) -> Dict[str, str]:
        keys = {
            "email": cls._email_key(lead),
            "linkedin": cls._linkedin_key(lead),
            "person_company": cls._person_company_key(lead),
            "company": cls._company_key(lead),
            "domain": cls._domain_key(lead),
        }

        return keys

    @staticmethod
    def _empty_key_registry() -> Dict[str, Set[str]]:
        return {
            "email": set(),
            "linkedin": set(),
            "person_company": set(),
            "company": set(),
            "domain": set(),
        }

    @classmethod
    def _register_keys(cls, registry: Dict[str, Set[str]], lead: Dict[str, Any]) -> None:
        for key_type, value in cls._dedup_keys(lead).items():
            if value:
                registry[key_type].add(value)

    @classmethod
    def _find_duplicate_reason(cls, registry: Dict[str, Set[str]], lead: Dict[str, Any]) -> str:
        for key_type, value in cls._dedup_keys(lead).items():
            if value and value in registry[key_type]:
                return key_type
        return ""

    @classmethod
    def summarize_leads(cls, leads: List[Dict[str, Any]]) -> Dict[str, Any]:
        confidence_buckets = {"high": 0, "medium": 0, "low": 0, "unknown": 0}
        source_buckets: Dict[str, int] = {}

        total = len(leads)
        usable = 0
        rejected = 0
        verified = 0
        with_email = 0
        with_phone = 0
        with_linkedin = 0

        for lead in leads:
            if lead.get("usable"):
                usable += 1
            if lead.get("rejected"):
                rejected += 1

            verification_status = cls._clean(lead.get("verification_status")).lower()
            if verification_status in {"valid", "risky"}:
                verified += 1

            if cls._email_key(lead):
                with_email += 1
            if cls._clean(lead.get("contact_phone")):
                with_phone += 1
            if cls._linkedin_key(lead):
                with_linkedin += 1

            confidence = cls._clean(lead.get("confidence")).lower() or "unknown"
            if confidence not in confidence_buckets:
                confidence = "unknown"
            confidence_buckets[confidence] += 1

            source = lead.get("source")
            if isinstance(source, list):
                sources = [cls._clean(item).lower() for item in source if cls._clean(item)]
            else:
                sources = [cls._clean(source).lower()] if cls._clean(source) else []

            if not sources:
                sources = ["unknown"]

            for bucket in sources:
                source_buckets[bucket] = source_buckets.get(bucket, 0) + 1

        return {
            "total_leads": total,
            "usable_leads": usable,
            "rejected_leads": rejected,
            "verified_leads": verified,
            "with_email": with_email,
            "with_phone": with_phone,
            "with_linkedin": with_linkedin,
            "confidence_breakdown": confidence_buckets,
            "source_breakdown": dict(sorted(source_buckets.items())),
        }

    async def export_leads(self, leads: List[Dict[str, Any]], mode: str = "generic") -> Dict[str, Any]:
        """
        Export strong, deduplicated leads to the dedicated worksheet.

        Deduplication now checks:
        - company name
        - email
        - website/email domain
        - LinkedIn URL
        - same person + same company
        """
        if str(mode or "").strip().lower() == "vento":
            return await self._export_vento_leads(leads)

        await self._call_store(self.todo_store.ensure_store)

        existing_rows = await self._call_store(self.todo_store.list_leads)
        existing_registry = self._empty_key_registry()
        for row in existing_rows:
            self._register_keys(existing_registry, row)

        gated_leads, gate_stats = apply_quality_gate(leads, drop_rejected=True)
        ranked_leads, quality_stats = rank_and_enrich_leads(gated_leads)
        batch_registry = self._empty_key_registry()

        added_count = 0
        duplicate_count = 0
        duplicate_reasons: Dict[str, int] = {}
        exported_leads: List[Dict[str, Any]] = []

        for lead in ranked_leads:
            duplicate_reason = self._find_duplicate_reason(existing_registry, lead)
            if not duplicate_reason:
                duplicate_reason = self._find_duplicate_reason(batch_registry, lead)

            if duplicate_reason:
                duplicate_count += 1
                duplicate_reasons[duplicate_reason] = duplicate_reasons.get(duplicate_reason, 0) + 1
                continue

            await self._call_store(self.todo_store.save_lead, lead)
            added_count += 1
            exported_leads.append(lead)
            self._register_keys(existing_registry, lead)
            self._register_keys(batch_registry, lead)

        return {
            "added": added_count,
            "duplicates_skipped": duplicate_count,
            "duplicate_reasons": duplicate_reasons,
            "total_processed": len(leads),
            "quality": {**quality_stats, "quality_gate": gate_stats},
            "report": self.summarize_leads(ranked_leads),
            "export_report": self.summarize_leads(exported_leads),
            "exported_preview": exported_leads[:5],
        }

    async def _export_vento_leads(self, leads: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Save usable Plan 3 leads with simple name/contact deduplication."""
        if not hasattr(self.todo_store, "ensure_vento_store"):
            raise RuntimeError("Vento Sheets storage is not available")
        await self._call_store(self.todo_store.ensure_vento_store)
        existing_rows = await self._call_store(self.todo_store.list_vento_leads)

        def keys_for(lead: Dict[str, Any]) -> Set[str]:
            output: Set[str] = set()
            name = self._clean(
                lead.get("creator_name") or lead.get("business_name") or lead.get("name")
                or lead.get("Name") or lead.get("Creator Name")
            ).lower()
            email = self._clean(lead.get("email") or lead.get("Email")).lower()
            instagram = self._clean(
                lead.get("instagram_handle") or lead.get("instagram")
                or lead.get("Instagram") or lead.get("Instagram Handle")
            ).lower().lstrip("@")
            website = self._clean(lead.get("website") or lead.get("Website"))
            domain = self._domain_from_url(website if "://" in website else f"https://{website}")
            if name:
                output.add(f"name:{name}")
            if email:
                output.add(f"email:{email}")
            if instagram:
                output.add(f"instagram:{instagram}")
            if domain and domain not in {"instagram.com", "www.instagram.com", "tiktok.com", "www.tiktok.com", "linktr.ee"}:
                output.add(f"domain:{domain.removeprefix('www.')}")
            return output

        existing_keys: Set[str] = set()
        for row in existing_rows or []:
            existing_keys.update(keys_for(row))

        batch_keys: Set[str] = set()
        exported: List[Dict[str, Any]] = []
        duplicates = 0
        rejected = 0
        min_social_followers = self._vento_min_social_followers()
        max_social_followers = self._vento_max_social_followers()
        require_social_followers = self._vento_require_social_followers()
        allow_above_max_followers = self._vento_allow_above_max_social_followers()
        for raw in leads or []:
            lead = dict(raw)
            quality_level = self._clean(lead.get("quality_level") or lead.get("lead_level") or lead.get("level")).upper()
            if lead.get("usable") is False or lead.get("rejected") is True or quality_level == "C":
                rejected += 1
                continue
            if self._is_vento_creator_lead(lead):
                follower_count = parse_follower_count(lead.get("follower_count"))
                if (require_social_followers and follower_count is None) or (
                    follower_count is not None and follower_count < min_social_followers
                ) or (
                    follower_count is not None
                    and max_social_followers > 0
                    and follower_count > max_social_followers
                    and not allow_above_max_followers
                ):
                    rejected += 1
                    continue
            keys = keys_for(lead)
            if keys.intersection(existing_keys) or keys.intersection(batch_keys):
                duplicates += 1
                continue
            exported.append(lead)
            existing_keys.update(keys)
            batch_keys.update(keys)

        if exported:
            await self._call_store(self.todo_store.save_vento_leads, exported)
        return {
            "added": len(exported),
            "duplicates_skipped": duplicates,
            "rejected_skipped": rejected,
            "total_processed": len(leads or []),
            "report": {
                "total_leads": len(leads or []),
                "usable_leads": len(exported),
                "with_email": sum(1 for lead in exported if lead.get("email")),
                "with_phone": sum(1 for lead in exported if lead.get("phone")),
                "with_instagram": sum(1 for lead in exported if lead.get("instagram_handle")),
                "with_tiktok": sum(1 for lead in exported if lead.get("tiktok_handle")),
                "with_website": sum(1 for lead in exported if lead.get("website")),
            },
            "exported_preview": exported[:5],
        }

    @staticmethod
    def _vento_min_social_followers() -> int:
        try:
            return max(0, min(10_000_000, int(os.getenv("VENTO_MIN_SOCIAL_FOLLOWERS", "50000"))))
        except (TypeError, ValueError):
            return 50_000

    @staticmethod
    def _vento_max_social_followers() -> int:
        try:
            return max(0, min(100_000_000, int(os.getenv("VENTO_MAX_SOCIAL_FOLLOWERS", "500000"))))
        except (TypeError, ValueError):
            return 500_000

    @staticmethod
    def _vento_require_social_followers() -> bool:
        return os.getenv("VENTO_REQUIRE_SOCIAL_FOLLOWERS", "false").strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _vento_allow_above_max_social_followers() -> bool:
        return os.getenv("VENTO_ALLOW_ABOVE_MAX_SOCIAL_FOLLOWERS", "true").strip().lower() in {"1", "true", "yes", "on"}

    @classmethod
    def _is_vento_creator_lead(cls, lead: Dict[str, Any]) -> bool:
        return cls._clean(lead.get("category")).lower() == "dog_parent_influencers"

    async def get_existing_companies(self, mode: str = "generic") -> List[str]:
        """
        Fetch existing company names from the leads worksheet for duplicate-checking.
        """
        if str(mode or "").strip().lower() == "vento" and hasattr(self.todo_store, "list_vento_leads"):
            if hasattr(self.todo_store, "ensure_vento_store"):
                await self._call_store(self.todo_store.ensure_vento_store)
            rows = await self._call_store(self.todo_store.list_vento_leads)
            return [
                self._clean(row.get("Name") or row.get("Creator Name"))
                for row in rows
                if self._clean(row.get("Name") or row.get("Creator Name"))
            ]

        await self._call_store(self.todo_store.ensure_store)
        rows = await self._call_store(self.todo_store.list_leads)
        companies: List[str] = []
        seen = set()
        for row in rows:
            company = self._clean(row.get("Company Name") or row.get("company_name"))
            if company and company.lower() not in seen:
                seen.add(company.lower())
                companies.append(company)
        return companies
