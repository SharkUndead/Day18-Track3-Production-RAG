"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH, OPENAI_API_KEY, OPENAI_MODEL_EVAL, EMBEDDING_MODEL


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    from datasets import Dataset
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        
        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        
        # Cấu hình LLM cho RAGAS sử dụng OpenAI
        llm = None
        if OPENAI_API_KEY:
            try:
                from langchain_openai import ChatOpenAI
                from ragas.llms import LangchainLLMWrapper
                # Dung OpenAI model cho evaluation
                eval_llm = ChatOpenAI(
                    model=OPENAI_MODEL_EVAL,
                    api_key=OPENAI_API_KEY
                )
                llm = LangchainLLMWrapper(eval_llm)
                
                # Dung model nho de tranh OOM (BGE-M3 qua nang khi chay cung reranker)
                from langchain_community.embeddings import HuggingFaceEmbeddings
                _emb_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                embeddings = HuggingFaceEmbeddings(model_name=_emb_model)
                
                # Gán LLM và Embeddings cho từng metric
                for metric in [faithfulness, answer_relevancy, context_precision, context_recall]:
                    metric.llm = llm
                    if hasattr(metric, 'embeddings'):
                        metric.embeddings = embeddings
            except Exception as e:
                print(f"⚠️ Không thể cấu hình OpenAI/Embeddings cho RAGAS: {e}. Sẽ dùng mặc định.")

        try:
            result = evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
            )
            df = result.to_pandas()
            per_question = []
            
            # Vá lỗi KeyError 'question' do Ragas đổi tên cột (user_input, etc.)
            # Sử dụng index để lấy dữ liệu từ các list gốc
            for i, row in df.iterrows():
                per_question.append(EvalResult(
                    question=questions[i],
                    answer=answers[i],
                    contexts=contexts[i],
                    ground_truth=ground_truths[i],
                    faithfulness=row.get("faithfulness", 0.0) or 0.0,
                    answer_relevancy=row.get("answer_relevancy", 0.0) or 0.0,
                    context_precision=row.get("context_precision", 0.0) or 0.0,
                    context_recall=row.get("context_recall", 0.0) or 0.0
                ))
                
            return {
                "faithfulness": result.get("faithfulness", 0.0),
                "answer_relevancy": result.get("answer_relevancy", 0.0),
                "context_precision": result.get("context_precision", 0.0),
                "context_recall": result.get("context_recall", 0.0),
                "per_question": per_question
            }
        except Exception as e:
            print(f"⚠️ RAGAS evaluation failed (có thể lỗi parse hoặc API): {e}")
            
    except ImportError:
        print("⚠️ Thư viện ragas hoặc datasets chưa được cài đặt.")
        
    # MOCK Trả về dummy dict để tránh crash pipeline
    return {"faithfulness": 0.0, "answer_relevancy": 0.0,
            "context_precision": 0.0, "context_recall": 0.0, "per_question": []}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if not eval_results:
        return []
        
    scored_results = []
    for res in eval_results:
        avg_score = (res.faithfulness + res.answer_relevancy + res.context_precision + res.context_recall) / 4.0
        
        metrics = {
            "faithfulness": res.faithfulness,
            "answer_relevancy": res.answer_relevancy,
            "context_precision": res.context_precision,
            "context_recall": res.context_recall
        }
        worst_metric = min(metrics.items(), key=lambda x: x[1])
        
        scored_results.append((avg_score, worst_metric, res))
        
    # Sắp xếp ascending để lấy các câu điểm thấp nhất
    scored_results.sort(key=lambda x: x[0])
    
    failures = []
    for avg_score, (metric_name, metric_score), res in scored_results[:bottom_n]:
        diagnosis = "Unknown issue"
        suggested_fix = "Review manually"
        
        # Diagnostic mapping
        if metric_name == "faithfulness" and metric_score < 0.85:
            diagnosis = "LLM hallucinating"
            suggested_fix = "Tighten prompt, lower temperature"
        elif metric_name == "context_recall" and metric_score < 0.75:
            diagnosis = "Missing relevant chunks"
            suggested_fix = "Improve chunking or add BM25"
        elif metric_name == "context_precision" and metric_score < 0.75:
            diagnosis = "Too many irrelevant chunks"
            suggested_fix = "Add reranking or metadata filter"
        elif metric_name == "answer_relevancy" and metric_score < 0.80:
            diagnosis = "Answer doesn't match question"
            suggested_fix = "Improve prompt template"
            
        failures.append({
            "question": res.question,
            "worst_metric": metric_name,
            "score": metric_score,
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix
        })
        
    return failures


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
