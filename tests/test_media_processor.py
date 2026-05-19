import asyncio
import zipfile
from io import BytesIO

from core.media.processor import MediaProcessor


def _zip_bytes(files):
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path, content in files.items():
            archive.writestr(path, content)
    return buffer.getvalue()


def _minimal_xlsx() -> bytes:
    return _zip_bytes(
        {
            "[Content_Types].xml": """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="xml" ContentType="application/xml"/>
</Types>""",
            "xl/sharedStrings.xml": """<?xml version="1.0" encoding="UTF-8"?>
<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <si><t>Venue</t></si>
  <si><t>Aurika</t></si>
  <si><t>Revenue</t></si>
</sst>""",
            "xl/worksheets/sheet1.xml": """<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
    <row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>2</v></c></row>
    <row r="2"><c r="A2" t="s"><v>1</v></c><c r="B2"><v>1200</v></c></row>
  </sheetData>
</worksheet>""",
        }
    )


def _minimal_pptx() -> bytes:
    return _zip_bytes(
        {
            "[Content_Types].xml": """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="xml" ContentType="application/xml"/>
</Types>""",
            "ppt/slides/slide1.xml": """<?xml version="1.0" encoding="UTF-8"?>
<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
       xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
  <p:cSld><p:spTree>
    <p:sp><p:txBody><a:p><a:r><a:t>Quarterly Plan</a:t></a:r></a:p></p:txBody></p:sp>
    <p:sp><p:txBody><a:p><a:r><a:t>Launch agent uploads</a:t></a:r></a:p></p:txBody></p:sp>
  </p:spTree></p:cSld>
</p:sld>""",
        }
    )


def test_media_processor_extracts_xlsx_text():
    result = asyncio.run(
        MediaProcessor().process(
            [
                (
                    "sales.xlsx",
                    _minimal_xlsx(),
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            ]
        )
    )

    message = result.build_message("Analyze this spreadsheet")

    assert "Venue" in message
    assert "Aurika" in message
    assert "Revenue" in message
    assert "1200" in message
    assert result.summaries[0].kind == "spreadsheet"


def test_media_processor_extracts_pptx_text():
    result = asyncio.run(
        MediaProcessor().process(
            [
                (
                    "roadmap.pptx",
                    _minimal_pptx(),
                    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                )
            ]
        )
    )

    message = result.build_message("Summarize this deck")

    assert "Quarterly Plan" in message
    assert "Launch agent uploads" in message
    assert result.summaries[0].kind == "presentation"
