"""
Production RAG Pipeline — End-to-End
Ghép nối M1, M2, M3, M4, M5 thành hệ thống hoàn chỉnh.
"""

import os, sys, time, json

# Fix encoding cho Windows console
if sys.platform == "win32":
    if not hasattr(sys.stdout, 'reconfigured'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
            sys.stdout.reconfigured = True
        except (AttributeError, Exception):
            pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import OPENAI_API_KEY, OPENAI_MODEL_GEN, DATA_DIR

REPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
REPORT_PATH = os.path.join(REPORT_DIR, "ragas_report.json")


class ProductionRAG:
    def __init__(self):
        print("[Init] Khởi tạo ProductionRAG...")
        print("[Init]   Tạo HybridSearch...")
        self.search = HybridSearch()
        print("[Init]   HybridSearch OK")
        print("[Init]   Tạo CrossEncoderReranker...")
        self.reranker = CrossEncoderReranker()
        print("[Init]   CrossEncoderReranker OK")
        self.llm_model = OPENAI_MODEL_GEN
        print(f"[Init] LLM model: {self.llm_model}")

    def index_documents(self):
        """M1 -> M5 -> M2."""
        # ── M1 ────────────────────────────────────────────────
        print("\n[1/3] Loading & Chunking documents (M1)...")
        try:
            docs = load_documents(DATA_DIR)
        except Exception as e:
            print(f"  [ERR] load_documents thất bại: {e}")
            return

        if not docs:
            print("  [WARN] Không tìm thấy tài liệu trong data/.")
            return
        print(f"  Loaded {len(docs)} documents.")

        all_children = []
        for doc in docs:
            try:
                _, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
                all_children.extend(children)
            except Exception as e:
                print(f"  [ERR] chunk_hierarchical: {e}")
        print(f"  Tạo được {len(all_children)} child chunks.")

        # ── M5 ────────────────────────────────────────────────
        print("\n[2/3] Enriching chunks (M5)...")
        try:
            chunks_to_enrich = [{"text": c.text, "metadata": c.metadata} for c in all_children]
            enriched = enrich_chunks(chunks_to_enrich, methods=["contextual"])
            final_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
            print(f"  Enrichment xong: {len(final_chunks)} chunks.")
        except Exception as e:
            print(f"  [WARN] Enrichment thất bại ({e}). Dùng raw chunks.")
            final_chunks = [{"text": c.text, "metadata": c.metadata} for c in all_children]

        # ── M2 ────────────────────────────────────────────────
        print("\n[3/3] Indexing into Hybrid Search (M2)...")
        try:
            print("  [3a] BM25 indexing...")
            self.search.bm25.index(final_chunks)
            print("  [3a] BM25 done.")
        except Exception as e:
            print(f"  [ERR] BM25 indexing: {e}")

        try:
            print("  [3b] Dense (BGE-M3) indexing — có thể mất 1-2 phút...")
            self.search.dense.index(final_chunks)
            print("  [3b] Dense done.")
        except MemoryError as e:
            print(f"  [ERR-OOM] Không đủ RAM: {e}. Chỉ dùng BM25.")
        except Exception as e:
            print(f"  [ERR] Dense indexing: {type(e).__name__}: {e}. Chỉ dùng BM25.")

        print("  Indexing hoàn tất.")

    def query(self, question: str, top_k_retrieval: int = 10, top_k_rerank: int = 3) -> tuple[str, list[str]]:
        """Retrieve -> Rerank -> Generate."""
        # 1. Retrieve
        print(f"    [Query] Hybrid search (top_k={top_k_retrieval})...", flush=True)
        try:
            raw_results = self.search.search(question, top_k=top_k_retrieval)
            print(f"    [Query] Retrieved {len(raw_results)} results.")
        except Exception as e:
            print(f"    [ERR] search: {e}")
            return "⚠️ Search thất bại.", []

        if not raw_results:
            return "Không tìm thấy thông tin liên quan.", []

        # 2. Rerank
        print(f"    [Query] Reranking (top_k={top_k_rerank})...", flush=True)
        try:
            docs_for_rerank = [{"text": r.text, "metadata": r.metadata} for r in raw_results]
            reranked = self.reranker.rerank(question, docs_for_rerank, top_k=top_k_rerank)
            contexts = [r.text for r in reranked] if reranked else [r.text for r in raw_results[:top_k_rerank]]
            print(f"    [Query] Reranked to {len(contexts)} contexts.")
        except Exception as e:
            print(f"    [WARN] Rerank thất bại ({e}). Dùng kết quả thô.")
            contexts = [r.text for r in raw_results[:top_k_rerank]]

        # 3. Generate
        print(f"    [Query] Generating answer (model={self.llm_model})...", flush=True)
        answer = self._generate_answer(question, "\n\n".join(contexts))
        print(f"    [Query] Answer: {answer[:80]}...")
        return answer, contexts

    def _generate_answer(self, query: str, context: str) -> str:
        """Sử dụng OpenAI để sinh câu trả lời."""
        if not OPENAI_API_KEY:
            return "⚠️ Thiếu OPENAI_API_KEY."
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            resp = client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "Bạn là trợ lý AI. Trả lời CHỈ dựa trên context được cung cấp. Nếu không có thông tin, hãy nói 'Không tìm thấy thông tin.'"},
                    {"role": "user", "content": f"Context:\n{context}\n\nCâu hỏi: {query}"},
                ],
                max_tokens=512,
                temperature=0.1,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"⚠️ Lỗi Generation: {str(e)}"

    def run_pipeline_and_evaluate(self):
        """Evaluation End-to-End (M4)."""
        print("\n" + "=" * 60)
        print("RUNNING END-TO-END EVALUATION")
        print("=" * 60)

        try:
            test_set = load_test_set()
        except Exception as e:
            print(f"  [ERR] Không đọc được test_set.json: {e}")
            return

        if not test_set:
            print("  [WARN] test_set.json rỗng.")
            return
        print(f"  Loaded {len(test_set)} test questions.\n")

        questions, answers, all_contexts, ground_truths = [], [], [], []

        for i, item in enumerate(test_set):
            q = item["question"]
            gt = item.get("ground_truth", "")
            print(f"  [{i+1}/{len(test_set)}] {q[:70]}")
            try:
                ans, ctxs = self.query(q)
            except Exception as e:
                print(f"    [ERR] query() crash: {e}")
                ans, ctxs = f"ERROR: {e}", []

            questions.append(q)
            answers.append(ans)
            all_contexts.append(ctxs if ctxs else [""])
            ground_truths.append(gt)

        # M4: RAGAS
        print("\n[Eval] Tính RAGAS scores...")
        try:
            results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
        except Exception as e:
            print(f"  [ERR] evaluate_ragas crash: {e}")
            results = {"faithfulness": 0.0, "answer_relevancy": 0.0,
                       "context_precision": 0.0, "context_recall": 0.0, "per_question": []}

        # M4: Failure Analysis
        print("[Eval] Phân tích lỗi...")
        try:
            failures = failure_analysis(results.get("per_question", []))
        except Exception as e:
            print(f"  [WARN] failure_analysis: {e}")
            failures = []

        # Lưu báo cáo vào reports/
        os.makedirs(REPORT_DIR, exist_ok=True)
        try:
            save_report(results, failures, path=REPORT_PATH)
        except Exception as e:
            print(f"  [WARN] save_report: {e}")

        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"  Faithfulness      : {results.get('faithfulness', 0):.4f}")
        print(f"  Answer Relevancy  : {results.get('answer_relevancy', 0):.4f}")
        print(f"  Context Precision : {results.get('context_precision', 0):.4f}")
        print(f"  Context Recall    : {results.get('context_recall', 0):.4f}")
        print(f"\n  Báo cáo: {REPORT_PATH}")


if __name__ == "__main__":
    start = time.time()
    print("=" * 60)
    print("Production RAG Pipeline - Starting")
    print("=" * 60)

    rag = ProductionRAG()
    rag.index_documents()
    rag.run_pipeline_and_evaluate()

    print(f"\nTổng thời gian: {time.time() - start:.1f}s")
