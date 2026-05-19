import asyncio
from typing import List, Dict, Any
from core.services.todo_sheet_store import TodoSheetStore

class LeadService:
    def __init__(self, todo_store: TodoSheetStore):
        self.todo_store = todo_store

    async def export_leads(self, leads: List[Dict[str, Any]], mode: str = "generic") -> Dict[str, Any]:
        """
        Exports leads to the Google Sheet with deduplication based on Company Name.
        """
        # Ensure store is ready
        await asyncio.to_thread(self.todo_store.ensure_store)
        
        # Get existing leads to check for duplicates
        existing_tasks = await asyncio.to_thread(self.todo_store.list_tasks, {"include_done": True})
        
        # Build a set of existing company names (case-insensitive, stripped)
        existing_companies = set()
        for task in existing_tasks:
            name = str(task.get("title", "")).strip().lower()
            if name:
                existing_companies.add(name)

        added_count = 0
        duplicate_count = 0
        
        # Determine project name based on mode
        project_name = "VelitCamping Outreach" if mode == "velit" else "Lead Generation"
        
        for lead in leads:
            company_name = lead.get("company_name", "").strip()
            company_key = company_name.lower()
            
            if company_key and company_key in existing_companies:
                duplicate_count += 1
                continue
            
            # Map lead fields to Task fields
            payload = {
                "title": company_name or "Unknown Company",
                "description": lead.get("value_proposition", ""),
                "category": "Lead",
                "project": project_name,
                "notes": f"Email: {lead.get('contact_email')}\nPhone: {lead.get('contact_phone')}\nFounder: {lead.get('founder_name')}\nLinkedIn: {lead.get('linkedin_url')}\nWebsite: {lead.get('company_website')}\nTech Stack: {lead.get('tech_stack')}\nSize: {lead.get('company_size')}",
                "status": "pending",
                "priority": "medium"
            }
            
            await asyncio.to_thread(self.todo_store.create_task, payload)
            added_count += 1
            if company_key:
                existing_companies.add(company_key)

        return {
            "added": added_count,
            "duplicates_skipped": duplicate_count,
            "total_processed": len(leads)
        }

    async def get_existing_companies(self, mode: str = "generic") -> List[str]:
        """
        Fetches a list of existing company names to prevent duplicates.
        """
        await asyncio.to_thread(self.todo_store.ensure_store)
        existing_tasks = await asyncio.to_thread(self.todo_store.list_tasks, {"include_done": True})
        
        existing_companies = set()
        for task in existing_tasks:
            name = str(task.get("title", "")).strip()
            if name:
                existing_companies.add(name)
        return list(existing_companies)
