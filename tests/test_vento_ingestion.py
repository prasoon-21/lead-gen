import json

import pytest
from openpyxl import Workbook

from core.services.vento.vento_lead_pipeline import VentoLeadPipeline


class DisabledTavily:
    async def search(self, payload):
        return {"results": []}


class DisabledProvider:
    configured = False


def pipeline():
    item = VentoLeadPipeline(tavily_client=DisabledTavily(), social_provider=DisabledProvider(), gemini_api_key="disabled")
    item._gemini_client = None
    return item


def test_csv_and_json_ingestion_map_flexible_headers():
    service = pipeline()
    csv_rows = service._parse_raw_text(
        'Full Name,Business Email,Instagram URL,Followers,City State\n'
        '"Jane, Creator",jane@example.com,https://instagram.com/jane.dog,12.5K,"Seattle, WA"'
    )
    assert csv_rows[0]["creator_name"] == "Jane, Creator"
    assert csv_rows[0]["email"] == "jane@example.com"
    assert csv_rows[0]["follower_count"] == "12.5K"

    json_rows = service._parse_raw_text(json.dumps([{"creator": "Alex", "ig": "@alexpet"}]))
    assert json_rows[0]["creator_name"] == "Alex"
    assert json_rows[0]["instagram_handle"] == "@alexpet"


def test_xlsx_ingestion(tmp_path):
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Name", "Email", "TikTok", "Location"])
    sheet.append(["Taylor", "taylor@example.com", "@taylortok", "Portland, OR"])
    path = tmp_path / "creators.xlsx"
    workbook.save(path)

    rows = pipeline()._parse_file(str(path))
    assert len(rows) == 1
    assert rows[0]["creator_name"] == "Taylor"
    assert rows[0]["tiktok_handle"] == "@taylortok"


def test_corrupt_xlsx_raises_clear_error(tmp_path):
    path = tmp_path / "broken.xlsx"
    path.write_bytes(b"not an excel workbook")
    with pytest.raises(ValueError, match="Unable to parse XLSX"):
        pipeline()._parse_file(str(path))


def test_legacy_xls_is_explicitly_rejected(tmp_path):
    path = tmp_path / "old.xls"
    path.write_bytes(b"legacy")
    with pytest.raises(ValueError, match="Legacy .xls"):
        pipeline()._parse_file(str(path))
