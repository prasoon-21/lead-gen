import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock
from fastapi import HTTPException
from api.routes.leads import get_lead_details
from api.state import AppState, get_state
from core.services.lead_service import LeadService
from core.services.todo_sheet_store import TodoSheetStore


# Mock get_state to return our test state
def mock_get_state_for_test(mock_todo_store):
    state = AppState()
    state.todo_store = mock_todo_store
    return state


class TestLeadServiceSources(unittest.TestCase):
    def setUp(self):
        # Mock TodoSheetStore
        self.mock_todo_store = MagicMock(spec=TodoSheetStore)
        self.mock_todo_store.ensure_store = AsyncMock()
        self.mock_todo_store.list_leads = AsyncMock(return_value=[])  # No existing leads by default
        self.mock_todo_store.save_lead = AsyncMock()

        self.lead_service = LeadService(self.mock_todo_store)

    def test_export_leads_with_list_source(self):
        leads = [
            {
                "company_name": "Test Company",
                "company_website": "https://testcompany.com",
                "contact_person_name": "John Doe",
                "contact_email": "john.doe@testcompany.com",
                "source": ["https://linkedin.com/in/johndoe", "https://testcompany.com/about"],
                "confidence": "high",
            }
        ]

        result = asyncio.run(self.lead_service.export_leads(leads))

        self.assertEqual(result["added"], 1)
        self.assertEqual(result["duplicates_skipped"], 0)
        self.mock_todo_store.save_lead.assert_called_once()

        # Extract the lead saved by save_lead
        saved_lead_arg = self.mock_todo_store.save_lead.call_args[0][0]

        # In LeadService, the lead dictionary is passed as-is to save_lead.
        # todo_sheet_store.py's _build_lead_metadata then processes it.
        # We need to verify that the 'source' field in the passed lead is indeed a list.
        self.assertIn("source", saved_lead_arg)
        self.assertIsInstance(saved_lead_arg["source"], list)
        self.assertEqual(saved_lead_arg["source"], ["https://linkedin.com/in/johndoe", "https://testcompany.com/about"])

    def test_summarize_leads_with_list_source(self):
        leads = [
            {
                "company_name": "Comp1",
                "source": ["web_search", "linkedin"],
                "confidence": "high"
            },
            {
                "company_name": "Comp2",
                "source": ["linkedin", "contact_page"],
                "confidence": "medium"
            },
            {
                "company_name": "Comp3",
                "source": "web_search", # Test mixed input
                "confidence": "low"
            },
            {
                "company_name": "Comp4",
                "confidence": "unknown"
            }
        ]

        summary = self.lead_service.summarize_leads(leads)

        self.assertIn("source_breakdown", summary)
        self.assertEqual(summary["source_breakdown"]["web_search"], 2) # From Comp1 and Comp3
        self.assertEqual(summary["source_breakdown"]["linkedin"], 2) # From Comp1 and Comp2
        self.assertEqual(summary["source_breakdown"]["contact_page"], 1) # From Comp2
        self.assertEqual(summary["source_breakdown"]["unknown"], 1) # From Comp4 (no source)
        self.assertEqual(summary["total_leads"], 4)

    def test_get_lead_details_endpoint(self):
        # Sample lead data that list_leads would return
        sample_leads_from_store = [
            {
                "Company Name": "Endpoint Test Co",
                "Email": "info@endpointtest.com",
                "Notes": "[AGENTIC_META]{\"source\": [\"https://endpointtest.com/about\", \"https://linkedin.com/company/endpointtest\"]}",
                "metadata": {
                    "source": ["https://endpointtest.com/about", "https://linkedin.com/company/endpointtest"],
                    "confidence": "high"
                }
            },
            {
                "Company Name": "Another Lead Inc",
                "Email": "contact@anotherlead.com",
                "Notes": "[AGENTIC_META]{\"source\": [\"https://anotherlead.com/contact\"]}",
                "metadata": {
                    "source": ["https://anotherlead.com/contact"],
                    "confidence": "medium"
                }
            }
        ]
        self.mock_todo_store.list_leads.return_value = sample_leads_from_store
        
        # Patch get_state to return our mocked state
        with unittest.mock.patch('api.routes.leads.get_state', return_value=mock_get_state_for_test(self.mock_todo_store)):
            leads = asyncio.run(get_lead_details())
            
            self.assertEqual(len(leads), 2)
            
            # Check first lead
            self.assertEqual(leads[0]["Company Name"], "Endpoint Test Co")
            self.assertIn("metadata", leads[0])
            self.assertIn("source", leads[0]["metadata"])
            self.assertIsInstance(leads[0]["metadata"]["source"], list)
            self.assertEqual(leads[0]["metadata"]["source"], ["https://endpointtest.com/about", "https://linkedin.com/company/endpointtest"])
            
            # Check second lead
            self.assertEqual(leads[1]["Company Name"], "Another Lead Inc")
            self.assertIn("metadata", leads[1])
            self.assertIn("source", leads[1]["metadata"])
            self.assertIsInstance(leads[1]["metadata"]["source"], list)
            self.assertEqual(leads[1]["metadata"]["source"], ["https://anotherlead.com/contact"])

    def test_get_lead_details_endpoint_no_store(self):
        # Patch get_state to return a state with no todo_store configured
        state_no_store = AppState()
        state_no_store.todo_store = None
        with unittest.mock.patch('api.routes.leads.get_state', return_value=state_no_store):
            with self.assertRaises(HTTPException) as cm:
                asyncio.run(get_lead_details())
            self.assertEqual(cm.exception.status_code, 500)
            self.assertEqual(cm.exception.detail, "Google Sheets storage not configured")

    def test_build_lead_metadata_preserves_source_json(self):
        lead = {
            "source": ["https://example.com/directory", "https://linkedin.com/in/example"],
            "source_details": [
                {"stage": "directory_discovery", "type": "search_result", "url": "https://example.com/directory"},
                {"stage": "linkedin_enrichment", "type": "linkedin_research", "url": "https://linkedin.com/in/example"},
            ],
        }
        metadata = TodoSheetStore._build_lead_metadata(lead)
        self.assertIsInstance(metadata["source"], list)
        self.assertEqual(len(metadata["source_details"]), 2)
        self.assertEqual(metadata["source_details"][0]["stage"], "directory_discovery")
