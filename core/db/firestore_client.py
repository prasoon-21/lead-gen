"""
FirestoreClient — singleton Firebase Admin SDK + async Firestore access.

Credentials are resolved in this order:
  1. Individual ENV vars (recommended — no file needed):
       FIREBASE_PROJECT_ID + FIREBASE_CLIENT_EMAIL + FIREBASE_PRIVATE_KEY
  2. FIREBASE_CREDENTIALS_PATH  → path to a service account JSON file
  3. GOOGLE_SERVICE_ACCOUNT_JSON → base64-encoded service account JSON

All collections are scoped under businesses/{business_id}/ so the same
Firestore project can serve multiple businesses safely.
"""

import asyncio
import base64
import json
import os
from typing import Any, Dict, List, Optional

_firebase_app = None  # singleton


def _init_firebase() -> None:
    global _firebase_app
    if _firebase_app is not None:
        return

    try:
        import firebase_admin
        from firebase_admin import credentials as fb_creds

        if firebase_admin._apps:
            _firebase_app = firebase_admin.get_app()
            return

        cred = _resolve_credentials(fb_creds)
        _firebase_app = firebase_admin.initialize_app(cred)

    except ImportError:
        raise RuntimeError(
            "firebase-admin is not installed. Run: pip install firebase-admin>=6.3.0"
        )


def _resolve_credentials(fb_creds):
    """Try credential sources in priority order."""

    # 1. Individual ENV vars — FIREBASE_PROJECT_ID + FIREBASE_CLIENT_EMAIL + FIREBASE_PRIVATE_KEY
    project_id = os.getenv("FIREBASE_PROJECT_ID")
    client_email = os.getenv("FIREBASE_CLIENT_EMAIL")
    private_key = os.getenv("FIREBASE_PRIVATE_KEY", "").replace("\\n", "\n")

    if project_id and client_email and private_key:
        info = {
            "type": "service_account",
            "project_id": project_id,
            "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID", ""),
            "private_key": private_key,
            "client_email": client_email,
            "client_id": os.getenv("FIREBASE_CLIENT_ID", ""),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
        return fb_creds.Certificate(info)

    # 2. Path to a JSON credentials file
    cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH", "firebase-credentials.json")
    if os.path.exists(cred_path):
        return fb_creds.Certificate(cred_path)

    # 3. Base64-encoded Google service account JSON (same one used for Sheets)
    b64 = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON_BASE64") or os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if b64:
        try:
            # Try base64 decode first; if it fails assume it's already raw JSON
            try:
                raw = base64.b64decode(b64).decode("utf-8")
            except Exception:
                raw = b64
            info = json.loads(raw)
            return fb_creds.Certificate(info)
        except Exception:
            pass

    raise RuntimeError(
        "No Firebase credentials found. Set FIREBASE_PROJECT_ID + FIREBASE_CLIENT_EMAIL + "
        "FIREBASE_PRIVATE_KEY in your .env file."
    )


class FirestoreClient:
    """Thin async wrapper around the synchronous Firestore client.

    All I/O is offloaded to a thread pool via asyncio.to_thread so it
    plays nicely with FastAPI's async event loop.
    """

    def __init__(self, business_id: Optional[str] = None):
        _init_firebase()
        from firebase_admin import firestore as fb_firestore
        self._db = fb_firestore.client()
        self.business_id = business_id or os.getenv("FIRESTORE_BUSINESS_ID", "aurika-agentic-core")

    def _col(self, collection: str):
        """Return a CollectionReference scoped to this business."""
        return (
            self._db
            .collection("businesses")
            .document(self.business_id)
            .collection(collection)
        )

    # ── Low-level async helpers ───────────────────────────────────────────────

    async def get(self, collection: str, doc_id: str) -> Optional[Dict[str, Any]]:
        def _read():
            doc = self._col(collection).document(doc_id).get()
            return doc.to_dict() if doc.exists else None
        return await asyncio.to_thread(_read)

    async def set(self, collection: str, doc_id: str, data: Dict[str, Any]) -> None:
        def _write():
            from google.cloud.firestore_v1 import SERVER_TIMESTAMP
            payload = {**data, "updated_at": SERVER_TIMESTAMP}
            self._col(collection).document(doc_id).set(payload, merge=True)
        await asyncio.to_thread(_write)

    async def delete(self, collection: str, doc_id: str) -> None:
        await asyncio.to_thread(
            lambda: self._col(collection).document(doc_id).delete()
        )

    async def list_all(self, collection: str) -> List[Dict[str, Any]]:
        def _list():
            return [doc.to_dict() for doc in self._col(collection).stream() if doc.exists]
        return await asyncio.to_thread(_list)

    async def query(
        self,
        collection: str,
        filters: List[tuple],              # [(field, op, value), ...]
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Run a filtered query against a collection."""
        def _query():
            ref = self._col(collection)
            for field, op, value in filters:
                ref = ref.where(field, op, value)
            if order_by:
                ref = ref.order_by(order_by)
            if limit:
                ref = ref.limit(limit)
            docs = ref.stream()
            results = [doc.to_dict() for doc in docs if doc.exists]
            return results[offset:] if offset else results
        return await asyncio.to_thread(_query)

    async def exists(self, collection: str, doc_id: str) -> bool:
        def _check():
            return self._col(collection).document(doc_id).get().exists
        return await asyncio.to_thread(_check)
