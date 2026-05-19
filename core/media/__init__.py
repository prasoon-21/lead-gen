"""
core.media — pre-processing layer for file attachments.

Converts any uploaded file into text/image data the AgentKernel can consume:
    Images (jpg, png, gif, webp)         → vision analysis → injected text
    Audio  (mp3, wav, ogg, m4a, webm, flac) → Whisper transcript → injected text
    PDF                                  → extracted text → injected text
    DOCX                                 → extracted text → injected text
    XLSX / XLSM                          → spreadsheet text → injected text
    PPTX                                 → slide text → injected text
    TXT / CSV / plain text               → read directly → injected text
"""
from core.media.processor import MediaProcessor, ProcessedAttachments, AttachmentSummary

__all__ = ["MediaProcessor", "ProcessedAttachments", "AttachmentSummary"]
