"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    try:
        from underthesea import word_tokenize
        return word_tokenize(text, format="text")
    except ImportError:
        print("⚠️ Thư viện underthesea chưa được cài đặt. Fallback về text gốc.")
        return text


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        from rank_bm25 import BM25Okapi
        self.documents = chunks
        self.corpus_tokens = []
        for chunk in chunks:
            # Segment and lowercase before splitting
            tokenized = segment_vietnamese(chunk["text"]).lower().split()
            self.corpus_tokens.append(tokenized)
            
        if self.corpus_tokens:
            self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if not self.bm25:
            return []
            
        segmented_query = segment_vietnamese(query).lower()
        tokenized_query = segmented_query.split()
        
        # Cải thiện cho short queries: Nếu query có nhiều từ rời rạc, thử ghép lại thành cụm từ
        if len(tokenized_query) > 1 and "_" not in segmented_query:
            tokenized_query.append("_".join(tokenized_query))
            
        scores = self.bm25.get_scores(tokenized_query)
        
        # Sort indices by score descending
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for i in top_indices:
            # Lấy tất cả kết quả có score > 0
            if scores[i] > 0:
                doc = self.documents[i]
                results.append(SearchResult(
                    text=doc["text"],
                    score=float(scores[i]),
                    metadata=doc.get("metadata", {}),
                    method="bm25"
                ))
        return results


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        try:
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=3.0)
            self.client.get_collections() # Test connection
        except Exception:
            print("⚠️ Qdrant server không khả dụng, chuyển sang dùng in-memory Qdrant.")
            self.client = QdrantClient(location=":memory:")
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        from qdrant_client.models import Distance, VectorParams, PointStruct
        
        # Tạo lại collection (force_recreate=True)
        self.client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
        )
        
        if not chunks:
            return
            
        texts = [c["text"] for c in chunks]
        vectors = self._get_encoder().encode(texts, show_progress_bar=False)
        
        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            payload = {**chunk.get("metadata", {}), "text": chunk["text"]}
            points.append(PointStruct(id=i, vector=vector.tolist(), payload=payload))
            
        self.client.upsert(collection_name=collection, points=points)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        query_vector = self._get_encoder().encode(query).tolist()
        hits = self.client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=top_k
        )
        
        results = []
        for hit in hits:
            results.append(SearchResult(
                text=hit.payload.get("text", ""),
                score=float(hit.score),
                metadata=hit.payload,
                method="dense"
            ))
        return results


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    rrf_scores = {}  # Dictionary lưu trữ score gộp
    
    for result_list in results_list:
        for rank, result in enumerate(result_list):
            if result.text not in rrf_scores:
                rrf_scores[result.text] = {
                    "score": 0.0,
                    "result": SearchResult(
                        text=result.text,
                        score=0.0,
                        metadata=result.metadata,
                        method="hybrid"
                    )
                }
            # Cộng dồn điểm RRF
            rrf_scores[result.text]["score"] += 1.0 / (k + rank + 1)
            
    # Gán lại điểm rrf và chuẩn bị list kết quả
    merged_results = []
    for data in rrf_scores.values():
        data["result"].score = data["score"]
        merged_results.append(data["result"])
        
    # Sắp xếp theo score giảm dần
    merged_results.sort(key=lambda x: x.score, reverse=True)
    
    return merged_results[:top_k]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
