"""
MediaProcessor — converts uploaded files into agent-consumable content.

Usage:
    processor = MediaProcessor(adapter=state.adapter)
    result = await processor.process(files)   # files: list of (filename, bytes, mime_type)
    augmented_message = result.build_message(original_message)
    # → original message + transcriptions + doc extracts + image descriptions

Supported formats
-----------------
Images   : jpg, jpeg, png, gif, webp        → Gemini Vision analysis
Audio    : mp3, wav, ogg, m4a, webm, flac   → OpenAI Whisper transcription
PDF      : .pdf                              → pypdf text extraction
Word     : .docx                             → python-docx text extraction
Excel    : .xlsx, .xlsm                      → lightweight XML text extraction
Slides   : .pptx                             → lightweight XML text extraction
Plain    : .txt, .csv, .md, .json, .xml     → decoded UTF-8
"""

import io
import os
import base64
import logging
import re
import zipfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

logger = logging.getLogger("agent.media")

# ── MIME type maps ────────────────────────────────────────────────────────────

IMAGE_MIMES = {
    "image/jpeg", "image/jpg", "image/png",
    "image/gif", "image/webp", "image/svg+xml",
}
AUDIO_MIMES = {
    "audio/mpeg", "audio/mp3", "audio/wav", "audio/wave",
    "audio/ogg", "audio/mp4", "audio/m4a", "audio/webm",
    "audio/flac", "audio/x-flac", "audio/x-wav",
    "video/webm",          # browser MediaRecorder often sends this for voice
}
PDF_MIMES   = {"application/pdf"}
DOCX_MIMES  = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
}
TEXT_MIMES  = {
    "text/plain", "text/csv", "text/markdown",
    "text/xml", "application/xml", "text/html",
    "application/json", "application/csv",
}
SPREADSHEET_MIMES = {
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel.sheet.macroenabled.12",
}
LEGACY_SPREADSHEET_MIMES = {"application/vnd.ms-excel"}
PRESENTATION_MIMES = {
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}
LEGACY_PRESENTATION_MIMES = {"application/vnd.ms-powerpoint"}

EXTENSION_MIME: Dict[str, str] = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".png": "image/png",  ".gif": "image/gif",
    ".webp": "image/webp",
    ".mp3": "audio/mpeg", ".wav": "audio/wav",
    ".ogg": "audio/ogg",  ".m4a": "audio/m4a",
    ".webm": "audio/webm",".flac": "audio/flac",
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc":  "application/msword",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xlsm": "application/vnd.ms-excel.sheet.macroenabled.12",
    ".xls": "application/vnd.ms-excel",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".ppt": "application/vnd.ms-powerpoint",
    ".txt": "text/plain", ".csv": "text/csv",
    ".md":  "text/markdown", ".json": "application/json",
    ".xml": "text/xml", ".html": "text/html", ".htm": "text/html",
}

DEFAULT_MAX_EXTRACTED_CHARS = 120_000


def _infer_mime(filename: str, declared: Optional[str] = None) -> str:
    """Best-effort MIME: use declared if sensible, else infer from extension."""
    if declared and declared != "application/octet-stream":
        return declared.split(";")[0].strip().lower()
    ext = os.path.splitext(filename.lower())[1]
    return EXTENSION_MIME.get(ext, "application/octet-stream")


# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class AttachmentSummary:
    filename: str
    mime_type: str
    kind: str          # "image" | "audio" | "document" | "spreadsheet" | "presentation" | "text" | "unsupported"
    chars_extracted: int = 0
    note: str = ""


@dataclass
class ProcessedAttachments:
    injected_blocks: List[str] = field(default_factory=list)   # text blocks to append
    summaries: List[AttachmentSummary] = field(default_factory=list)

    def build_message(self, original_message: str) -> str:
        """Merge original message with all extracted content blocks."""
        if not self.injected_blocks:
            return original_message
        blocks_text = "\n\n".join(self.injected_blocks)
        return (
            f"{original_message}\n\n"
            "--- Attached content ---\n"
            f"{blocks_text}\n"
            "--- End of attached content ---"
        )

    def to_metadata(self) -> List[Dict[str, Any]]:
        return [
            {
                "filename": s.filename,
                "kind": s.kind,
                "mime_type": s.mime_type,
                "chars_extracted": s.chars_extracted,
                "note": s.note,
            }
            for s in self.summaries
        ]


# ── MediaProcessor ────────────────────────────────────────────────────────────

class MediaProcessor:
    """
    Process a list of (filename, bytes, mime_type) tuples.
    adapter  — BaseLLMAdapter with generate_with_vision (used for images)
    """

    def __init__(self, adapter=None, max_extracted_chars: Optional[int] = None):
        self._adapter = adapter
        self._max_extracted_chars = max_extracted_chars or _env_int(
            "AGENT_ATTACHMENT_MAX_EXTRACTED_CHARS",
            DEFAULT_MAX_EXTRACTED_CHARS,
        )

    async def process(
        self, files: List[Tuple[str, bytes, Optional[str]]]
    ) -> ProcessedAttachments:
        result = ProcessedAttachments()
        for filename, file_bytes, declared_mime in files:
            mime = _infer_mime(filename, declared_mime)
            try:
                if mime in IMAGE_MIMES:
                    await self._handle_image(filename, file_bytes, mime, result)
                elif mime in AUDIO_MIMES:
                    await self._handle_audio(filename, file_bytes, mime, result)
                elif mime in PDF_MIMES:
                    self._handle_pdf(filename, file_bytes, result)
                elif mime in DOCX_MIMES:
                    self._handle_docx(filename, file_bytes, result)
                elif mime in SPREADSHEET_MIMES:
                    self._handle_xlsx(filename, file_bytes, mime, result)
                elif mime in LEGACY_SPREADSHEET_MIMES:
                    result.summaries.append(AttachmentSummary(
                        filename=filename,
                        mime_type=mime,
                        kind="unsupported",
                        note="Legacy .xls files are not supported yet; upload .xlsx or .csv",
                    ))
                elif mime in PRESENTATION_MIMES:
                    self._handle_pptx(filename, file_bytes, mime, result)
                elif mime in LEGACY_PRESENTATION_MIMES:
                    result.summaries.append(AttachmentSummary(
                        filename=filename,
                        mime_type=mime,
                        kind="unsupported",
                        note="Legacy .ppt files are not supported yet; upload .pptx",
                    ))
                elif mime in TEXT_MIMES:
                    self._handle_text(filename, file_bytes, mime, result)
                else:
                    result.summaries.append(AttachmentSummary(
                        filename=filename, mime_type=mime,
                        kind="unsupported",
                        note=f"Unsupported file type '{mime}' — skipped",
                    ))
            except Exception as exc:
                logger.warning("media_processor_error filename=%s error=%s", filename, exc)
                result.summaries.append(AttachmentSummary(
                    filename=filename, mime_type=mime,
                    kind="error", note=str(exc),
                ))
        return result

    # ── Image ─────────────────────────────────────────────────────────────────

    async def _handle_image(
        self,
        filename: str,
        data: bytes,
        mime: str,
        result: ProcessedAttachments,
    ) -> None:
        if not self._adapter or not hasattr(self._adapter, "generate_with_vision"):
            # Fallback: base64-encode and note it
            result.summaries.append(AttachmentSummary(
                filename=filename, mime_type=mime, kind="image",
                note="Vision adapter not available — image skipped",
            ))
            return

        b64 = base64.b64encode(data).decode()
        images = [{"base64": b64, "mimeType": mime, "filename": filename}]
        llm_result = await self._adapter.generate_with_vision(
            prompt=(
                "Carefully describe the content of this image. "
                "Extract any text, tables, charts, diagrams, or key information visible. "
                "Be thorough and specific — this description will be the agent's only reference to the image."
            ),
            images=images,
        )
        description = llm_result.text.strip()
        block = f"[Image: {filename}]\n{description}"
        result.injected_blocks.append(block)
        result.summaries.append(AttachmentSummary(
            filename=filename, mime_type=mime, kind="image",
            chars_extracted=len(description),
            note="Analyzed via vision model",
        ))

    # ── Audio ─────────────────────────────────────────────────────────────────

    async def _handle_audio(
        self,
        filename: str,
        data: bytes,
        mime: str,
        result: ProcessedAttachments,
    ) -> None:
        transcript = await _transcribe_audio(filename, data, mime)
        block = f"[Audio transcription: {filename}]\n{transcript}"
        result.injected_blocks.append(block)
        result.summaries.append(AttachmentSummary(
            filename=filename, mime_type=mime, kind="audio",
            chars_extracted=len(transcript),
            note="Transcribed via Whisper",
        ))

    # ── PDF ───────────────────────────────────────────────────────────────────

    def _handle_pdf(
        self,
        filename: str,
        data: bytes,
        result: ProcessedAttachments,
    ) -> None:
        text = _extract_pdf_text(data)
        if not text.strip():
            result.summaries.append(AttachmentSummary(
                filename=filename, mime_type="application/pdf",
                kind="document", note="PDF contained no extractable text (may be scanned)",
            ))
            return
        block = f"[Document: {filename}]\n{text}"
        block, chars_extracted, note = self._limited_block(block, "Text extracted from PDF")
        result.injected_blocks.append(block)
        result.summaries.append(AttachmentSummary(
            filename=filename, mime_type="application/pdf",
            kind="document", chars_extracted=chars_extracted,
            note=note,
        ))

    # ── DOCX ──────────────────────────────────────────────────────────────────

    def _handle_docx(
        self,
        filename: str,
        data: bytes,
        result: ProcessedAttachments,
    ) -> None:
        text = _extract_docx_text(data)
        block = f"[Document: {filename}]\n{text}"
        block, chars_extracted, note = self._limited_block(block, "Text extracted from Word document")
        result.injected_blocks.append(block)
        result.summaries.append(AttachmentSummary(
            filename=filename,
            mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            kind="document", chars_extracted=chars_extracted,
            note=note,
        ))

    # ── Excel / Sheets export ─────────────────────────────────────────────────

    def _handle_xlsx(
        self,
        filename: str,
        data: bytes,
        mime: str,
        result: ProcessedAttachments,
    ) -> None:
        text = _extract_xlsx_text(data)
        if not text.strip():
            result.summaries.append(AttachmentSummary(
                filename=filename, mime_type=mime,
                kind="spreadsheet", note="Spreadsheet contained no extractable text",
            ))
            return
        block = f"[Spreadsheet: {filename}]\n{text}"
        block, chars_extracted, note = self._limited_block(block, "Text extracted from spreadsheet")
        result.injected_blocks.append(block)
        result.summaries.append(AttachmentSummary(
            filename=filename, mime_type=mime,
            kind="spreadsheet", chars_extracted=chars_extracted,
            note=note,
        ))

    # ── PowerPoint ────────────────────────────────────────────────────────────

    def _handle_pptx(
        self,
        filename: str,
        data: bytes,
        mime: str,
        result: ProcessedAttachments,
    ) -> None:
        text = _extract_pptx_text(data)
        if not text.strip():
            result.summaries.append(AttachmentSummary(
                filename=filename, mime_type=mime,
                kind="presentation", note="Presentation contained no extractable text",
            ))
            return
        block = f"[Presentation: {filename}]\n{text}"
        block, chars_extracted, note = self._limited_block(block, "Text extracted from presentation")
        result.injected_blocks.append(block)
        result.summaries.append(AttachmentSummary(
            filename=filename, mime_type=mime,
            kind="presentation", chars_extracted=chars_extracted,
            note=note,
        ))

    # ── Plain text ────────────────────────────────────────────────────────────

    def _handle_text(
        self,
        filename: str,
        data: bytes,
        mime: str,
        result: ProcessedAttachments,
    ) -> None:
        text = data.decode("utf-8", errors="replace").strip()
        block = f"[File: {filename}]\n{text}"
        block, chars_extracted, note = self._limited_block(block, "Text decoded as UTF-8")
        result.injected_blocks.append(block)
        result.summaries.append(AttachmentSummary(
            filename=filename, mime_type=mime,
            kind="text", chars_extracted=chars_extracted,
            note=note,
        ))

    def _limited_block(self, block: str, note: str) -> tuple[str, int, str]:
        if len(block) <= self._max_extracted_chars:
            return block, len(block), note
        clipped = block[: self._max_extracted_chars].rstrip()
        return (
            f"{clipped}\n\n[Attachment text truncated at {self._max_extracted_chars} characters]",
            len(clipped),
            f"{note}; truncated at {self._max_extracted_chars} characters",
        )


# ── Standalone helpers ────────────────────────────────────────────────────────

async def _transcribe_audio(filename: str, data: bytes, mime: str) -> str:
    """Transcribe audio using OpenAI Whisper. Falls back to a placeholder if unavailable."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "[Audio transcription unavailable — OPENAI_API_KEY not set]"
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key)
        # Whisper accepts the file as a tuple: (filename, bytes, mime_type)
        transcript = await client.audio.transcriptions.create(
            model="whisper-1",
            file=(filename, io.BytesIO(data), mime),
            response_format="text",
        )
        return str(transcript).strip()
    except Exception as exc:
        logger.warning("whisper_transcription_failed filename=%s error=%s", filename, exc)
        return f"[Audio transcription failed: {exc}]"


def _extract_pdf_text(data: bytes) -> str:
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(data))
        pages = []
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                pages.append(extracted)
        return "\n\n".join(pages)
    except ImportError:
        return "[PDF text extraction unavailable — install pypdf: pip3 install pypdf]"
    except Exception as exc:
        return f"[PDF extraction error: {exc}]"


def _extract_docx_text(data: bytes) -> str:
    try:
        import docx
        doc = docx.Document(io.BytesIO(data))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)
    except ImportError:
        return "[DOCX text extraction unavailable — install python-docx: pip3 install python-docx]"
    except Exception as exc:
        return f"[DOCX extraction error: {exc}]"


def _extract_xlsx_text(data: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            shared_strings = _read_xlsx_shared_strings(archive)
            sheet_paths = sorted(
                (
                    name for name in archive.namelist()
                    if name.startswith("xl/worksheets/") and name.endswith(".xml")
                ),
                key=_natural_key,
            )
            sheets = []
            for sheet_number, path in enumerate(sheet_paths, start=1):
                xml = archive.read(path)
                root = ET.fromstring(xml)
                rows = []
                for row in root.findall(".//{*}row"):
                    values = []
                    for cell in row.findall("{*}c"):
                        value = _xlsx_cell_value(cell, shared_strings)
                        if value:
                            values.append(value)
                    if values:
                        rows.append(" | ".join(values))
                if rows:
                    sheets.append(f"Sheet {sheet_number}:\n" + "\n".join(rows))
            return "\n\n".join(sheets)
    except Exception as exc:
        return f"[XLSX extraction error: {exc}]"


def _read_xlsx_shared_strings(archive: zipfile.ZipFile) -> List[str]:
    try:
        xml = archive.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ET.fromstring(xml)
    strings = []
    for item in root.findall("{*}si"):
        parts = [text_node.text or "" for text_node in item.findall(".//{*}t")]
        strings.append("".join(parts))
    return strings


def _xlsx_cell_value(cell: ET.Element, shared_strings: List[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return " ".join(text_node.text or "" for text_node in cell.findall(".//{*}t")).strip()

    value_node = cell.find("{*}v")
    if value_node is None or value_node.text is None:
        return ""

    value = value_node.text.strip()
    if cell_type == "s":
        try:
            return shared_strings[int(value)].strip()
        except (ValueError, IndexError):
            return value
    return value


def _extract_pptx_text(data: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            slide_paths = sorted(
                (
                    name for name in archive.namelist()
                    if name.startswith("ppt/slides/slide") and name.endswith(".xml")
                ),
                key=_natural_key,
            )
            slides = []
            for slide_number, path in enumerate(slide_paths, start=1):
                root = ET.fromstring(archive.read(path))
                texts = [
                    node.text.strip()
                    for node in root.findall(".//{*}t")
                    if node.text and node.text.strip()
                ]
                if texts:
                    slides.append(f"Slide {slide_number}:\n" + "\n".join(texts))
            return "\n\n".join(slides)
    except Exception as exc:
        return f"[PPTX extraction error: {exc}]"


def _natural_key(value: str) -> List[Any]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if not raw_value:
        return default
    try:
        value = int(raw_value)
    except ValueError:
        return default
    return value if value > 0 else default
