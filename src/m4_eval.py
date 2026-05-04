"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json, math
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
    """Load test set from JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation with manual aggregate calculation."""
    from datasets import Dataset
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        
        # Map đa dạng key để tương thích với mọi phiên bản Ragas
        dataset = Dataset.from_dict({
            "question": questions,
            "user_input": questions,
            "answer": answers,
            "response": answers,
            "contexts": contexts,
            "retrieved_contexts": contexts,
            "ground_truth": ground_truths,
            "reference": ground_truths,
        })
        
        # Cấu hình LLM cho RAGAS
        llm = None
        if OPENAI_API_KEY:
            try:
                from langchain_openai import ChatOpenAI
                from ragas.llms import LangchainLLMWrapper
                eval_llm = ChatOpenAI(
                    model=OPENAI_MODEL_EVAL,
                    api_key=OPENAI_API_KEY,
                    temperature=0
                )
                llm = LangchainLLMWrapper(eval_llm)
                
                try:
                    from langchain_huggingface import HuggingFaceEmbeddings
                except ImportError:
                    from langchain_community.embeddings import HuggingFaceEmbeddings
                
                _emb_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                embeddings = HuggingFaceEmbeddings(model_name=_emb_model)
                
                for metric in [faithfulness, answer_relevancy, context_precision, context_recall]:
                    metric.llm = llm
                    if hasattr(metric, 'embeddings'):
                        metric.embeddings = embeddings
            except Exception as e:
                print(f"⚠️ Warning during RAGAS Setup: {e}")

        try:
            # Thực hiện đánh giá
            result = evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
            )
            
            df = result.to_pandas()
            per_question = []
            
            def safe_score(val):
                """Xử lý NaN, None và ép kiểu float."""
                try:
                    if val is None: return 0.0
                    f_val = float(val)
                    return 0.0 if math.isnan(f_val) else f_val
                except (ValueError, TypeError):
                    return 0.0

            # 1. Trích xuất kết quả chi tiết từng câu
            for i in range(len(questions)):
                row = df.iloc[i] if i < len(df) else {}
                per_question.append(EvalResult(
                    question=questions[i],
                    answer=answers[i],
                    contexts=contexts[i],
                    ground_truth=ground_truths[i],
                    faithfulness=safe_score(row.get("faithfulness", 0.0)),
                    answer_relevancy=safe_score(row.get("answer_relevancy", 0.0)),
                    context_precision=safe_score(row.get("context_precision", 0.0)),
                    context_recall=safe_score(row.get("context_recall", 0.0))
                ))
            
            # 2. Tự tính điểm Aggregate (Chống lỗi NaN hoặc KeyError từ Ragas object)
            num_q = len(per_question)
            if num_q > 0:
                agg_scores = {
                    "faithfulness": sum(q.faithfulness for q in per_question) / num_q,
                    "answer_relevancy": sum(q.answer_relevancy for q in per_question) / num_q,
                    "context_precision": sum(q.context_precision for q in per_question) / num_q,
                    "context_recall": sum(q.context_recall for q in per_question) / num_q,
                }
            else:
                agg_scores = {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0, "context_recall": 0.0}

            return {
                **agg_scores,
                "per_question": per_question
            }
            
        except Exception as e:
            print(f"⚠️ Error during RAGAS Evaluation execution: {e}")
            
    except Exception as e:
        print(f"⚠️ Critical Error in evaluate_ragas: {e}")
        
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
        
    scored_results.sort(key=lambda x: x[0])
    
    failures = []
    for avg_score, (metric_name, metric_score), res in scored_results[:bottom_n]:
        diagnosis = "Unknown issue"
        suggested_fix = "Review manually"
        
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


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON."""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")
