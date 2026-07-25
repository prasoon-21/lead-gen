import asyncio

from core.services.vento.daily_batch import VentoDailyBatchRunner


def accepted_lead(index, level="A"):
    return {
        "creator_id": f"creator-{index}",
        "creator_name": f"Creator {index}",
        "instagram_handle": f"@creator{index}",
        "quality_level": level,
        "email_verification_status": "valid" if level == "A" else "not_available",
        "preferred_contact_method": "email" if level == "A" else "instagram_dm",
        "profile_verified": True,
    }


class FakePipeline:
    async def run(self, **kwargs):
        leads = [accepted_lead(index, "A" if index < 2 else "B") for index in range(3)]
        return {
            "batch_id": "generated-batch",
            "leads": leads,
            "review_leads": [{**accepted_lead(9), "quality_level": "C"}],
            "rejected_leads": [{}],
            "steps": [],
            "metadata": {"total_raw_ingested": 10, "total_unique": 7, "deduplication": {"same_run_duplicates": 2, "historical_duplicates": 1}, "source_breakdown": {"bulk_ingest": 10}},
        }


class FakeStore:
    def __init__(self):
        self.saved = []
        self.review = []
        self.runs = []

    def list_vento_leads(self):
        return []

    def ensure_vento_store(self):
        return {}

    def save_vento_leads(self, leads):
        self.saved.extend(leads)

    def save_vento_review_leads(self, leads):
        self.review.extend(leads)

    def save_vento_daily_run(self, report):
        self.runs.append(report)

    def get_vento_daily_run(self, batch_id):
        return None


def test_daily_batch_returns_honest_shortfall_and_persists():
    store = FakeStore()
    result = asyncio.run(
        VentoDailyBatchRunner(pipeline=FakePipeline(), todo_store=store).run(
            location="Seattle, WA",
            daily_target=5,
            raw_target=10,
            idempotency_key="daily-1",
        )
    )
    assert result["accepted_count"] == 3
    assert result["shortfall"] == 2
    assert result["status"] == "completed_with_shortfall"
    assert result["storage_status"] == "saved"
    assert len(store.saved) == 3
    assert len(store.review) == 1
    assert store.runs[0]["batch_id"] == "daily-1"
