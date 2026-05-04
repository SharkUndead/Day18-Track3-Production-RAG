"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents:
            return []
            
        model = self._load_model()
        # Tạo danh sách các cặp [query, text]
        pairs = [[query, doc["text"]] for doc in documents]
        
        # Compute scores
        scores = model.predict(pairs)
        if isinstance(scores, float):
            scores = [scores]
            
        # Gom kết quả lại cùng original document
        combined = []
        for score, doc in zip(scores, documents):
            combined.append((score, doc))
            
        # Sắp xếp theo score giảm dần
        combined.sort(key=lambda x: x[0], reverse=True)
        
        # Trả về top_k dưới định dạng RerankResult
        results = []
        for rank, (score, doc) in enumerate(combined[:top_k], start=1):
            results.append(RerankResult(
                text=doc["text"],
                original_score=doc.get("score", 0.0),
                rerank_score=float(score),
                metadata=doc.get("metadata", {}),
                rank=rank
            ))
            
        return results


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        try:
            from flashrank import Ranker, RerankRequest
            if self._model is None:
                self._model = Ranker()
                
            # Đổi format cho flashrank
            passages = [
                {"id": i, "text": d["text"], "meta": d.get("metadata", {}), "original_score": d.get("score", 0.0)} 
                for i, d in enumerate(documents)
            ]
            
            request = RerankRequest(query=query, passages=passages)
            results = self._model.rerank(request)[:top_k]
            
            # Map kết quả ngược lại thành RerankResult
            rerank_results = []
            for rank, res in enumerate(results, start=1):
                rerank_results.append(RerankResult(
                    text=res["text"],
                    original_score=res.get("original_score", 0.0),
                    rerank_score=float(res["score"]),
                    metadata=res.get("meta", {}),
                    rank=rank
                ))
            return rerank_results
            
        except ImportError:
            print("⚠️ Thư viện flashrank chưa được cài đặt. Bỏ qua Reranking.")
            return []


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs."""
    import time
    
    # Warm up để load model vào RAM/GPU
    reranker.rerank(query, documents)
    
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        times.append((time.perf_counter() - start) * 1000)  # Tính theo milliseconds
        
    if not times:
        return {"avg_ms": 0, "min_ms": 0, "max_ms": 0}
        
    return {
        "avg_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times)
    }


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
