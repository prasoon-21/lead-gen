import base64

import pytest
from fastapi import HTTPException

from api.routes.agent_kernel import _files_from_document_payloads, _validate_upload_sizes


def test_files_from_document_payloads_decodes_base64_documents():
    encoded = base64.b64encode(b"# Notes\nUse uploaded context.").decode("ascii")

    files = _files_from_document_payloads(
        [
            {
                "filename": "notes.md",
                "mimeType": "text/markdown",
                "base64": encoded,
            }
        ]
    )

    assert files == [("notes.md", b"# Notes\nUse uploaded context.", "text/markdown")]


def test_files_from_document_payloads_rejects_invalid_base64():
    with pytest.raises(HTTPException) as exc_info:
        _files_from_document_payloads(
            [{"filename": "bad.txt", "mimeType": "text/plain", "base64": "not-valid***"}]
        )

    assert exc_info.value.status_code == 422
    assert "valid base64" in exc_info.value.detail


def test_validate_upload_sizes_rejects_file_over_limit():
    with pytest.raises(HTTPException) as exc_info:
        _validate_upload_sizes([("large.txt", b"12345", "text/plain")], max_file_bytes=4, max_total_bytes=10)

    assert exc_info.value.status_code == 413
    assert "large.txt" in exc_info.value.detail


def test_validate_upload_sizes_rejects_total_over_limit():
    with pytest.raises(HTTPException) as exc_info:
        _validate_upload_sizes(
            [("a.txt", b"123", "text/plain"), ("b.txt", b"456", "text/plain")],
            max_file_bytes=10,
            max_total_bytes=5,
        )

    assert exc_info.value.status_code == 413
    assert "combined upload size" in exc_info.value.detail
