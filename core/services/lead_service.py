import asyncio
from typing import List, Dict, Any
from core.services.todo_sheet_store import TodoSheetStore


class LeadService:
    def __init__(self, todo_store: TodoSheetStore):
        self.todo_store = todo_store

    async def export_leads(self, leads: List[Dict[str, Any]], mode: str = "generic") -> Dict[str, Any]:
        """
        Exports leads to the dedicated 'leads' worksheet in Google Sheets.
        Columns: Company Name | Specialty | Contact Name | Email | Phone |
                 Website | City | State | Zip Code | Status | Notes
        Deduplicates on Company Name (case-insensitive).
        """
        # Ensure store and leads worksheet are ready
        await asyncio.to_thread(self.todo_store.ensure_store)

        # Fetch existing company names from the leads sheet for deduplication
        existing_names = await asyncio.to_thread(self.todo_store.get_lead_company_names)
        existing_companies = {name.lower() for name in existing_names}

        added_count = 0
        duplicate_count = 0

        for lead in leads:
            company_name = lead.get("company_name", "").strip()
            company_key = company_name.lower()

            if company_key and company_key in existing_companies:
                duplicate_count += 1
                continue

            await asyncio.to_thread(self.todo_store.save_lead, lead)
            added_count += 1
            if company_key:
                existing_companies.add(company_key)

        return {
            "added": added_count,
            "duplicates_skipped": duplicate_count,
            "total_processed": len(leads),
        }

    async def get_existing_companies(self, mode: str = "generic") -> List[str]:
        """
        Fetches existing company names from the leads worksheet for duplicate-checking.
        """
        await asyncio.to_thread(self.todo_store.ensure_store)
        return await asyncio.to_thread(self.todo_store.get_lead_company_names)

