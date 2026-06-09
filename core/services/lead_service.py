import asyncio
from typing import Any, Dict, List, Set
from urllib.parse import urlparse

from core.services.lead_quality_service import rank_and_enrich_leads
from core.services.todo_sheet_store import TodoSheetStore


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
        }

        # Only use coarse company/domain dedup for company-level fallback leads.
        # Person-level leads should be allowed to coexist at the same company
        # as long as they represent different people/identities.
        if not cls._is_person_level_lead(lead):
            keys["company"] = cls._company_key(lead)
            keys["domain"] = cls._domain_key(lead)

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
        await asyncio.to_thread(self.todo_store.ensure_store)

        existing_rows = await asyncio.to_thread(self.todo_store.list_leads)
        existing_registry = self._empty_key_registry()
        for row in existing_rows:
            self._register_keys(existing_registry, row)

        ranked_leads, quality_stats = rank_and_enrich_leads(leads)
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

            await asyncio.to_thread(self.todo_store.save_lead, lead)
            added_count += 1
            exported_leads.append(lead)
            self._register_keys(existing_registry, lead)
            self._register_keys(batch_registry, lead)

        return {
            "added": added_count,
            "duplicates_skipped": duplicate_count,
            "duplicate_reasons": duplicate_reasons,
            "total_processed": len(leads),
            "quality": quality_stats,
            "report": self.summarize_leads(ranked_leads),
            "export_report": self.summarize_leads(exported_leads),
            "exported_preview": exported_leads[:5],
        }

    async def get_existing_companies(self, mode: str = "generic") -> List[str]:
        """
        Fetch existing company names from the leads worksheet for duplicate-checking.
        """
        await asyncio.to_thread(self.todo_store.ensure_store)
        rows = await asyncio.to_thread(self.todo_store.list_leads)
        companies: List[str] = []
        seen = set()
        for row in rows:
            company = self._clean(row.get("Company Name") or row.get("company_name"))
            if company and company.lower() not in seen:
                seen.add(company.lower())
                companies.append(company)
        return companies
