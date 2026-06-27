import unittest

from core.services.lead_quality_gate import apply_quality_gate, normalize_and_score_lead


class TestLeadQualityGate(unittest.TestCase):
    def test_trusted_directory_source_does_not_reject_official_site(self):
        lead = {
            "company_name": "Ready Set Van",
            "company_website": "https://readysetvan.com/",
            "industry": "Upfitter",
            "value_proposition": "Custom camper van and commercial vehicle upfitting services.",
            "contact_email": "hello@readysetvan.com",
            "contact_phone": "(609) 878-8822",
            "source": "trusted_directory",
            "directory_url": "https://example-directory.test/ready-set-van",
        }

        normalized = normalize_and_score_lead(lead)

        self.assertNotEqual(normalized["pipeline_disposition"], "Rejected")
        self.assertEqual(normalized["rejection_reason_code"], "NULL")

    def test_banned_final_website_is_still_rejected(self):
        lead = {
            "company_name": "Directory Row",
            "company_website": "https://www.yelp.com/biz/directory-row",
            "industry": "Upfitter",
            "contact_email": "sales@example.com",
        }

        normalized = normalize_and_score_lead(lead)

        self.assertEqual(normalized["pipeline_disposition"], "Rejected")
        self.assertEqual(normalized["rejection_reason_code"], "ERR_BANNED_INDUSTRY")

    def test_generic_company_email_survives_as_needs_review_alternative(self):
        lead = {
            "company_name": "Brooklyn Campervans",
            "company_website": "https://brooklyncampervans.com/",
            "industry": "Upfitter",
            "value_proposition": "Camper van conversion and upfitting services.",
            "contact_email": "sales@brooklyncampervans.com",
        }

        normalized = normalize_and_score_lead(lead)

        self.assertEqual(normalized["pipeline_disposition"], "Needs_Review")
        self.assertEqual(normalized["contact_email"], "")
        self.assertIn("sales@brooklyncampervans.com", normalized["alternative_contacts"]["generic_emails"])

    def test_apply_quality_gate_keeps_reviewable_generic_email_lead(self):
        leads, stats = apply_quality_gate(
            [
                {
                    "company_name": "Highland Vans",
                    "company_website": "https://highlandvans.com/",
                    "industry": "Upfitter",
                    "value_proposition": "Custom van conversion and upfitter services.",
                    "contact_email": "info@highlandvans.com",
                    "source": "trusted_directory",
                }
            ]
        )

        self.assertEqual(len(leads), 1)
        self.assertEqual(leads[0]["pipeline_disposition"], "Needs_Review")
        self.assertEqual(stats["rejected_count"], 0)


if __name__ == "__main__":
    unittest.main()
