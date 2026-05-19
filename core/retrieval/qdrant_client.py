import re
import hashlib
import time
from typing import Dict, Any, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from config.schemas import FilterConfig
from core.retrieval.embeddings import EmbeddingService
from core.retrieval.filter_builder import build_qdrant_filter


class QdrantRetriever:
    
    EXACT_QUERY_PATTERNS = {
        "exercise": r'exercise\s*(\d+\.?\d*)',
        "example": r'example\s*(\d+\.?\d*)',
        "theorem": r'theorem\s*(\d+\.?\d*)',
        "activity": r'activity\s*(\d+\.?\d*)',
    }
    
    def __init__(
        self,
        url: str,
        api_key: Optional[str],
        collection_name: str,
        embedding_service: EmbeddingService,
        top_k: int = 7,
        score_threshold: float = 0.5
    ):
        use_https = url.lower().startswith("https") if url else True
        self.client = QdrantClient(
            url=url,
            api_key=api_key,
            prefer_grpc=False,
            https=use_https,
            timeout=30
        )
        self.collection_name = collection_name
        self.embedding_service = embedding_service
        self.top_k = top_k
        self.score_threshold = score_threshold
    
    def classify_query(self, query: str) -> Dict[str, Any]:
        query_lower = query.lower().strip()
        
        for query_type, pattern in self.EXACT_QUERY_PATTERNS.items():
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                number = match.group(1)
                heading_format = {
                    "exercise": f"EXERCISE {number}",
                    "example": f"Example {number}",
                    "theorem": f"Theorem {number}",
                    "activity": f"Activity {number}",
                }
                return {
                    "type": "exact",
                    "query_type": query_type,
                    "number": number,
                    "heading": heading_format.get(query_type, ""),
                    "content_type": query_type
                }
        
        if any(word in query_lower for word in ["diagram", "figure", "picture", "image", "show"]):
            return {"type": "visual", "query_type": "visual", "number": None, "heading": None}
        
        return {"type": "conceptual", "query_type": "conceptual", "number": None, "heading": None}
    
    async def search(
        self,
        query: str,
        context: Dict[str, Any],
        top_k: Optional[int] = None,
        prioritize_images: bool = False,
        collection_name: Optional[str] = None,
        filter_config: Optional[FilterConfig] = None,
    ) -> List[Dict[str, Any]]:
        collection = self._require_collection(collection_name)
        return await self._vector_search(
            query,
            context,
            top_k,
            prioritize_images,
            collection_name=collection,
            filter_config=filter_config,
        )
    
    async def _vector_search(
        self,
        query: str,
        context: Dict[str, Any],
        top_k: Optional[int] = None,
        prioritize_images: bool = False,
        collection_name: Optional[str] = None,
        filter_config: Optional[FilterConfig] = None,
    ) -> List[Dict[str, Any]]:
        query_vector = await self.embedding_service.embed(query)
        search_filter = build_qdrant_filter(filter_config)

        results = self._search_points(
            query_vector=query_vector,
            query_filter=search_filter,
            limit=top_k or self.top_k,
            with_payload=True,
            collection_name=collection_name,
        )
        
        formatted = self._format_results(results)
        return self._score_chunks(formatted, query, context)

    def _search_points(self, query_vector, query_filter, limit, with_payload, collection_name: str):
        if hasattr(self.client, "search"):
            return self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=with_payload,
            )

        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=with_payload,
            )
            return getattr(response, "points", response)

        raise AttributeError("QdrantClient has no compatible search/query_points method")
    
    def _require_collection(self, collection_name: Optional[str]) -> str:
        if not collection_name:
            raise ValueError("collection_name is required for vector search")
        return collection_name
    
    def _score_chunks(
        self,
        chunks: List[Dict[str, Any]],
        query: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        query_info = self.classify_query(query)
        
        scored = []
        for chunk in chunks:
            score = chunk.get("score", 0.5)
            m = chunk.get("metadata", {})
            
            content_type = query_info.get("content_type")
            if content_type and m.get("content_type") == content_type:
                score += 0.3
            
            heading = m.get("heading", "").lower()
            if query_info.get("number") and query_info["number"] in heading:
                score += 0.5
            
            if context.get("chapter") and m.get("chapter_name") == context.get("chapter"):
                score += 0.2
            
            if "visual" in query_info.get("type", "") and m.get("has_images") == "true":
                score += 0.3
            
            chunk["score"] = score
            scored.append(chunk)
        
        scored.sort(reverse=True, key=lambda x: x.get("score", 0))
        return scored[:5]
    
    def _format_scroll_results(self, results, match_type: str = "scroll") -> List[Dict[str, Any]]:
        formatted = []
        seen_content = set()
        
        for r in results:
            payload = r.payload or {}
            content = (
                payload.get("pageContent") or
                payload.get("content") or
                payload.get("text") or
                ""
            )
            
            content_hash = hashlib.md5(content.encode()).hexdigest()[:16]
            if content_hash in seen_content:
                continue
            seen_content.add(content_hash)
            
            metadata = payload.get("metadata", payload)
            
            formatted.append({
                "content": content,
                "metadata": metadata,
                "score": 1.0,
                "match_type": match_type,
                "level": metadata.get("level", ""),
                "subject": metadata.get("subject", ""),
                "board": metadata.get("board", ""),
                "medium": metadata.get("medium", ""),
                "course_type": metadata.get("course_type", ""),
                "chapter_number": metadata.get("chapter_number", ""),
                "topic_name": metadata.get("topic_name", ""),
                "subtopic_name": metadata.get("subtopic_name", ""),
                "content_type": metadata.get("content_type", ""),
                "has_images": metadata.get("has_images", "false"),
                "source_file": metadata.get("source_file", ""),
                "image_url": metadata.get("image_url", ""),
                "image_caption": metadata.get("image_caption", ""),
            })
        
        return formatted
    
    def _format_results(self, results) -> List[Dict[str, Any]]:
        formatted = []
        seen_content = set()
        
        for r in results:
            payload = r.payload or {}
            content = (
                payload.get("pageContent") or
                payload.get("content") or
                payload.get("text") or
                ""
            )
            
            content_hash = hashlib.md5(content.encode()).hexdigest()[:16]
            if content_hash in seen_content:
                continue
            seen_content.add(content_hash)
            
            metadata = payload.get("metadata", payload)
            
            formatted.append({
                "content": content,
                "metadata": metadata,
                "score": r.score,
                "match_type": "vector",
                "level": metadata.get("level", ""),
                "subject": metadata.get("subject", ""),
                "board": metadata.get("board", ""),
                "medium": metadata.get("medium", ""),
                "course_type": metadata.get("course_type", ""),
                "chapter_number": metadata.get("chapter_number", ""),
                "topic_name": metadata.get("topic_name", ""),
                "subtopic_name": metadata.get("subtopic_name", ""),
                "content_type": metadata.get("content_type", ""),
                "has_images": metadata.get("has_images", "false"),
                "source_file": metadata.get("source_file", ""),
                "image_url": metadata.get("image_url", ""),
                "image_caption": metadata.get("image_caption", ""),
            })
        
        return formatted
    
    async def upsert_image(self, metadata: Dict[str, Any]) -> bool:
        try:
            content = " ".join([
                f"Description: {metadata.get('image_description', '')}",
                f"Caption: {metadata.get('image_caption', '')}",
                f"Prompt: {metadata.get('image_prompt', '')}"
            ])
            
            print(f"   📝 Embedding image metadata...")
            vector = await self.embedding_service.embed(content)
            
            point_id = int(time.time() * 1000)
            
            payload = {
                "content": content,
                "metadata": {
                    "level": metadata.get("level", "Class_10"),
                    "subject": metadata.get("subject", ""),
                    "board": metadata.get("board", "CBSE"),
                    "medium": metadata.get("medium", "English"),
                    "course_type": metadata.get("course_type", "School"),
                    "chapter_number": str(metadata.get("chapter_number", "")),
                    "topic_name": metadata.get("topic_name", "") or metadata.get("topic", ""),
                    "subtopic_name": metadata.get("subtopic_name", "") or metadata.get("subtopic", ""),
                    "content_type": "image",
                    "image_caption": metadata.get("image_caption", ""),
                    "image_url": metadata.get("image_url", ""),
                }
            }
            
            print(f"   📤 Upserting to Qdrant (id: {point_id})...")
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload
                    )
                ]
            )
            
            print(f"   ✅ Image stored in Qdrant for future retrieval")
            return True
            
        except Exception as e:
            print(f"   ⚠️ Qdrant upsert failed: {e}")
            return False
