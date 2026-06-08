"""
RAG Evaluation Pipeline — DeepEval (Group Project — Person 2).

Chạy:
    python -m group_project.evaluation.eval_pipeline

Yêu cầu:
    pip install deepeval
    OPENAI_API_KEY phải được set trong .env

Pipeline:
    1. Load golden_dataset.json (17 cặp Q&A)
    2. Chạy RAG pipeline trên từng question với 2 configs
    3. Evaluate với 4 metrics (Faithfulness, AnswerRelevancy, ContextualRecall, ContextualPrecision)
    4. So sánh A/B: Config A (hybrid + reranking) vs Config B (hybrid, no reranking)
    5. Export kết quả ra results.md
"""

import json
import sys
from pathlib import Path

# Thêm root vào PYTHONPATH khi chạy trực tiếp
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"


# =============================================================================
# LLM Judge helpers
# =============================================================================

def _judge(prompt: str) -> float:
    """Call GPT-4o-mini as a judge, returns score 0.0-1.0."""
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an objective evaluator. "
                    "Respond ONLY with a decimal number between 0.0 and 1.0 "
                    "representing the score. No explanation."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=10,
    )
    try:
        return float(response.choices[0].message.content.strip())
    except ValueError:
        return 0.5


def score_faithfulness(answer: str, context: str) -> float:
    """Faithfulness: câu trả lời có bám đúng context không?"""
    prompt = f"""Rate how faithful this answer is to the provided context.
Score 1.0 if every claim in the answer is directly supported by the context.
Score 0.0 if the answer contains information not in the context (hallucination).

Context:
{context[:2000]}

Answer:
{answer[:1000]}

Score (0.0 to 1.0):"""
    return _judge(prompt)


def score_answer_relevance(question: str, answer: str) -> float:
    """Answer Relevance: câu trả lời có đúng câu hỏi không?"""
    prompt = f"""Rate how relevant this answer is to the question.
Score 1.0 if the answer directly and completely addresses the question.
Score 0.0 if the answer is off-topic or does not address the question.

Question: {question}
Answer: {answer[:1000]}

Score (0.0 to 1.0):"""
    return _judge(prompt)


def score_context_recall(question: str, context: str, expected_answer: str) -> float:
    """Context Recall: retriever có lấy đủ evidence không?"""
    prompt = f"""Rate how well the retrieved context supports answering this question,
given the expected answer.
Score 1.0 if the context contains all information needed to derive the expected answer.
Score 0.0 if the context is missing key information needed for the answer.

Question: {question}
Expected answer: {expected_answer}
Retrieved context:
{context[:2000]}

Score (0.0 to 1.0):"""
    return _judge(prompt)


def score_context_precision(question: str, context: str) -> float:
    """Context Precision: bao nhiêu % context thực sự hữu ích?"""
    prompt = f"""Rate what proportion of the retrieved context is actually relevant to answering the question.
Score 1.0 if all retrieved context is directly relevant.
Score 0.0 if all retrieved context is irrelevant noise.

Question: {question}
Retrieved context:
{context[:2000]}

Score (0.0 to 1.0):"""
    return _judge(prompt)


# =============================================================================
# DeepEval Evaluation
# =============================================================================

def evaluate_with_deepeval(
    generate_fn,
    golden_dataset: list[dict],
    use_reranking: bool = True,
    verbose: bool = True,
) -> list:
    """
    Evaluate RAG pipeline sử dụng DeepEval.

    Args:
        generate_fn: Callable nhận (question, use_reranking) → {'answer', 'sources', ...}
        golden_dataset: List của {'question', 'expected_answer', 'expected_context'}
        use_reranking: Config flag truyền vào generate_fn
        verbose: In tiến trình

    Returns:
        List of TestResult objects từ DeepEval.
    """
    from deepeval import evaluate
    from deepeval.metrics import (
        FaithfulnessMetric,
        AnswerRelevancyMetric,
        ContextualRecallMetric,
        ContextualPrecisionMetric,
    )
    from deepeval.test_case import LLMTestCase

    test_cases = []
    for i, item in enumerate(golden_dataset):
        if verbose:
            print(f"  [{i+1}/{len(golden_dataset)}] {item['question'][:60]}...")

        result = generate_fn(item["question"], use_reranking=use_reranking)

        test_case = LLMTestCase(
            input=item["question"],
            actual_output=result["answer"],
            expected_output=item["expected_answer"],
            retrieval_context=[c["content"] for c in result["sources"]],
        )
        test_cases.append(test_case)

    metrics = [
        FaithfulnessMetric(threshold=0.7),
        AnswerRelevancyMetric(threshold=0.7),
        ContextualRecallMetric(threshold=0.7),
        ContextualPrecisionMetric(threshold=0.7),
    ]

    return evaluate(test_cases, metrics)


# =============================================================================
# A/B Comparison
# =============================================================================

def compare_configs(generate_fn, golden_dataset: list[dict]) -> dict:
    """
    So sánh A/B giữa 2 configs:
      - Config A: hybrid search + reranking  (use_reranking=True)
      - Config B: hybrid search, no reranking (use_reranking=False)

    Returns:
        {'config_a': results_a, 'config_b': results_b}
    """
    print("\n=== Config A: Hybrid + Reranking ===")
    results_a = evaluate_with_deepeval(generate_fn, golden_dataset, use_reranking=True)

    print("\n=== Config B: Hybrid, No Reranking ===")
    results_b = evaluate_with_deepeval(generate_fn, golden_dataset, use_reranking=False)

    return {"config_a": results_a, "config_b": results_b}


# =============================================================================
# Parse DeepEval results
# =============================================================================

def _extract_scores(eval_results) -> dict:
    """Trích xuất điểm trung bình từng metric từ DeepEval EvaluationResult."""
    metric_names = [
        "FaithfulnessMetric",
        "AnswerRelevancyMetric",
        "ContextualRecallMetric",
        "ContextualPrecisionMetric",
    ]
    scores = {name: [] for name in metric_names}

    for test_result in eval_results.test_results:
        for metric_data in test_result.metrics_data:
            name = metric_data.name
            if name in scores and metric_data.score is not None:
                scores[name].append(metric_data.score)

    return {
        name: (sum(vals) / len(vals) if vals else 0.0)
        for name, vals in scores.items()
    }


def _find_worst_performers(eval_results, golden_dataset: list[dict], top_n: int = 3) -> list[dict]:
    """Tìm top_n test cases có điểm thấp nhất (avg across metrics)."""
    rows = []
    for i, test_result in enumerate(eval_results.test_results):
        scores = [m.score for m in test_result.metrics_data if m.score is not None]
        avg = sum(scores) / len(scores) if scores else 0.0
        metric_map = {m.name: m.score for m in test_result.metrics_data}
        rows.append({
            "question": golden_dataset[i]["question"],
            "avg_score": avg,
            "faithfulness": metric_map.get("FaithfulnessMetric", 0.0),
            "relevancy": metric_map.get("AnswerRelevancyMetric", 0.0),
            "recall": metric_map.get("ContextualRecallMetric", 0.0),
        })

    rows.sort(key=lambda x: x["avg_score"])
    return rows[:top_n]


# =============================================================================
# Main
# =============================================================================

def export_results(comparison: dict, golden_dataset: list[dict]):
    """Export evaluation results ra results.md."""
    scores_a = _extract_scores(comparison["config_a"])
    scores_b = _extract_scores(comparison["config_b"])
    worst = _find_worst_performers(comparison["config_a"], golden_dataset)

    def avg(scores: dict) -> float:
        vals = list(scores.values())
        return sum(vals) / len(vals) if vals else 0.0

    def fmt(v) -> str:
        return f"{v:.3f}" if v is not None else "N/A"

    def delta(a, b) -> str:
        d = a - b
        return f"+{d:.3f}" if d >= 0 else f"{d:.3f}"

    content = "# RAG Evaluation Results\n\n"
    content += "## Framework sử dụng\n\nDeepEval\n\n---\n\n"

    content += "## Overall Scores\n\n"
    content += "| Metric | Config A (hybrid + rerank) | Config B (no reranking) | Δ |\n"
    content += "|--------|---------------------------|------------------------|---|\n"

    metric_display = {
        "FaithfulnessMetric": "Faithfulness",
        "AnswerRelevancyMetric": "Answer Relevance",
        "ContextualRecallMetric": "Context Recall",
        "ContextualPrecisionMetric": "Context Precision",
    }
    for key, label in metric_display.items():
        a = scores_a.get(key, 0.0)
        b = scores_b.get(key, 0.0)
        content += f"| {label} | {fmt(a)} | {fmt(b)} | {delta(a, b)} |\n"

    content += f"| **Average** | **{fmt(avg(scores_a))}** | **{fmt(avg(scores_b))}** | **{delta(avg(scores_a), avg(scores_b))}** |\n"

    content += "\n---\n\n"
    content += "## A/B Comparison Analysis\n\n"
    content += "**Config A: Hybrid Search + Reranking**\n"
    content += "> Kết hợp semantic search + BM25 lexical search, sau đó rerank bằng cross-encoder. "
    content += "Reranking giúp đưa chunks liên quan nhất lên đầu, giảm noise trong context.\n\n"
    content += "**Config B: Hybrid Search, No Reranking**\n"
    content += "> Kết hợp semantic search + BM25 lexical search, merge bằng RRF nhưng bỏ bước reranking. "
    content += "Nhanh hơn nhưng context có thể kém relevant hơn.\n\n"
    content += "**Kết luận:**\n"
    if avg(scores_a) >= avg(scores_b):
        diff = avg(scores_a) - avg(scores_b)
        content += f"> Config A (hybrid + reranking) tốt hơn Config B {diff:.3f} điểm trung bình. "
        content += "> Reranking cải thiện chất lượng context, giúp LLM trả lời chính xác và faithful hơn.\n"
    else:
        diff = avg(scores_b) - avg(scores_a)
        content += f"> Config B (no reranking) cho kết quả tương đương hoặc nhỉnh hơn {diff:.3f} điểm. "
        content += "> Cần phân tích thêm xem reranker hiện tại có phù hợp với domain pháp luật không.\n"

    content += "\n---\n\n"
    content += "## Worst Performers (Bottom 3 — Config A)\n\n"
    content += "| # | Question | Faithfulness | Relevance | Recall | Root Cause |\n"
    content += "|---|----------|-------------|-----------|--------|------------|\n"
    for i, row in enumerate(worst, 1):
        q = row["question"][:60] + ("..." if len(row["question"]) > 60 else "")
        f = fmt(row["faithfulness"])
        r = fmt(row["relevancy"])
        rc = fmt(row["recall"])
        cause = "Retriever không lấy đúng chunk" if row["recall"] < 0.5 else "LLM hallucinate / không đủ evidence"
        content += f"| {i} | {q} | {f} | {r} | {rc} | {cause} |\n"

    content += "\n---\n\n"
    content += "## Recommendations\n\n"
    content += "### Cải tiến 1: Nâng chất lượng chunking\n"
    content += "**Action:** Thử chunk theo điều/khoản thay vì sliding window — chunk ranh giới điều luật rõ ràng hơn.  \n"
    content += "**Expected impact:** Tăng Context Precision và Faithfulness vì mỗi chunk sẽ chứa một ý pháp lý hoàn chỉnh.\n\n"
    content += "### Cải tiến 2: Fine-tune reranker cho domain pháp luật\n"
    content += "**Action:** Sử dụng reranker được fine-tune trên dữ liệu pháp luật Việt Nam thay vì cross-encoder tổng quát.  \n"
    content += "**Expected impact:** Tăng Context Recall vì reranker hiểu tốt hơn các thuật ngữ pháp lý.\n\n"
    content += "### Cải tiến 3: Tăng số lượng golden dataset\n"
    content += "**Action:** Mở rộng golden dataset lên 30-50 cặp Q&A, bao phủ thêm các điều luật trong Bộ luật Hình sự.  \n"
    content += "**Expected impact:** Kết quả evaluation đáng tin cậy hơn, phát hiện được thêm điểm yếu của pipeline.\n"

    RESULTS_PATH.write_text(content, encoding="utf-8")
    print(f"\nResults exported to {RESULTS_PATH}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")

    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} test cases from golden_dataset.json")

    # Import generation function từ group project
    from src.generation import generate_with_citation

    print("\nStarting A/B evaluation...")
    comparison = compare_configs(generate_with_citation, golden_dataset)

    print("\nExporting results...")
    export_results(comparison, golden_dataset)

    print("\nDone! See group_project/evaluation/results.md")
