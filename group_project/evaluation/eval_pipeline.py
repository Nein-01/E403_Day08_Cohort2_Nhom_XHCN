"""
RAG Evaluation Pipeline — LLM-as-Judge approach.

Sử dụng GPT-4o-mini làm judge để đánh giá 4 metrics:
  1. Faithfulness    — Câu trả lời có bám đúng context không?
  2. Answer Relevance — Câu trả lời có đúng câu hỏi không?
  3. Context Recall  — Retriever có lấy đủ evidence không?
  4. Context Precision — Trong context lấy về, bao nhiêu % thực sự hữu ích?

A/B Comparison:
  Config A: Dense-only (chỉ semantic search, không reranking)
  Config B: Hybrid + RRF Reranking (semantic + lexical + RRF)

Chạy:
    python -m group_project.evaluation.eval_pipeline
"""

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Add project root to path
ROOT = Path(__file__).parent.parent.parent
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
# Single test case evaluation
# =============================================================================

def evaluate_single(
    question: str,
    expected_answer: str,
    rag_fn,
    use_reranking: bool = True,
) -> dict:
    """Evaluate một test case. Returns dict với answer, sources, metrics."""
    result = rag_fn(question, use_reranking=use_reranking)
    answer = result["answer"]
    sources = result["sources"]
    context = "\n\n---\n\n".join(c["content"] for c in sources[:5])

    faithfulness = score_faithfulness(answer, context)
    relevance = score_answer_relevance(question, answer)
    recall = score_context_recall(question, context, expected_answer)
    precision = score_context_precision(question, context)

    return {
        "question": question,
        "answer": answer,
        "expected_answer": expected_answer,
        "context_chunks": len(sources),
        "faithfulness": faithfulness,
        "answer_relevance": relevance,
        "context_recall": recall,
        "context_precision": precision,
        "avg_score": (faithfulness + relevance + recall + precision) / 4,
    }


# =============================================================================
# RAG pipeline wrappers (A/B configs)
# =============================================================================

def rag_hybrid_rerank(question: str, use_reranking: bool = True) -> dict:
    """Config B: Hybrid search + RRF reranking (mặc định)."""
    from src.task9_retrieval_pipeline import retrieve
    from src.task10_generation import generate_with_citation, reorder_for_llm, format_context, SYSTEM_PROMPT, TEMPERATURE, TOP_P
    import os
    from openai import OpenAI

    chunks = retrieve(question, top_k=5, use_reranking=use_reranking)
    if not chunks:
        return {"answer": "Toi khong the xac minh thong tin nay.", "sources": []}

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    user_message = f"Context:\n{context}\n\n---\n\nCau hoi: {question}"

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    return {"answer": response.choices[0].message.content, "sources": chunks}


def rag_dense_only(question: str, use_reranking: bool = False) -> dict:
    """Config A: Dense-only (semantic search, no RRF)."""
    from src.task5_semantic_search import semantic_search
    from src.task10_generation import reorder_for_llm, format_context, SYSTEM_PROMPT, TEMPERATURE, TOP_P
    import os
    from openai import OpenAI

    chunks = semantic_search(question, top_k=5)
    for c in chunks:
        c["source"] = "dense"
    if not chunks:
        return {"answer": "Toi khong the xac minh thong tin nay.", "sources": []}

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    user_message = f"Context:\n{context}\n\n---\n\nCau hoi: {question}"

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    return {"answer": response.choices[0].message.content, "sources": chunks}


# =============================================================================
# Full evaluation
# =============================================================================

def run_evaluation(golden_dataset: list[dict], rag_fn, config_name: str) -> list[dict]:
    """Chạy evaluation trên toàn bộ golden dataset."""
    results = []
    total = len(golden_dataset)

    for i, item in enumerate(golden_dataset, 1):
        print(f"  [{config_name}] {i}/{total}: {item['question'][:60]}...")
        try:
            r = evaluate_single(item["question"], item["expected_answer"], rag_fn)
            results.append(r)
        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({
                "question": item["question"],
                "answer": "",
                "expected_answer": item["expected_answer"],
                "context_chunks": 0,
                "faithfulness": 0.0,
                "answer_relevance": 0.0,
                "context_recall": 0.0,
                "context_precision": 0.0,
                "avg_score": 0.0,
            })
        time.sleep(0.5)  # Tránh rate limit

    return results


def compute_averages(results: list[dict]) -> dict:
    if not results:
        return {}
    n = len(results)
    return {
        "faithfulness": round(sum(r["faithfulness"] for r in results) / n, 3),
        "answer_relevance": round(sum(r["answer_relevance"] for r in results) / n, 3),
        "context_recall": round(sum(r["context_recall"] for r in results) / n, 3),
        "context_precision": round(sum(r["context_precision"] for r in results) / n, 3),
        "avg_score": round(sum(r["avg_score"] for r in results) / n, 3),
    }


def find_worst_performers(results: list[dict], n: int = 5) -> list[dict]:
    return sorted(results, key=lambda x: x["avg_score"])[:n]


# =============================================================================
# Export results.md
# =============================================================================

def export_results(config_a_results: list[dict], config_b_results: list[dict]):
    avg_a = compute_averages(config_a_results)
    avg_b = compute_averages(config_b_results)

    def delta(b, a):
        d = b - a
        return f"+{d:.3f}" if d >= 0 else f"{d:.3f}"

    worst = find_worst_performers(config_b_results, n=5)

    lines = [
        "# RAG Evaluation Results\n",
        f"**Dataset:** {len(config_b_results)} test cases  ",
        f"**Model:** gpt-4o-mini (judge + generation)  ",
        "**Framework:** LLM-as-Judge (custom)\n",
        "---\n",
        "## Overall Scores\n",
        "| Metric | Config A (Dense-only) | Config B (Hybrid+RRF) | Delta |",
        "|--------|----------------------|----------------------|-------|",
        f"| Faithfulness    | {avg_a.get('faithfulness',0):.3f} | {avg_b.get('faithfulness',0):.3f} | {delta(avg_b.get('faithfulness',0), avg_a.get('faithfulness',0))} |",
        f"| Answer Relevance | {avg_a.get('answer_relevance',0):.3f} | {avg_b.get('answer_relevance',0):.3f} | {delta(avg_b.get('answer_relevance',0), avg_a.get('answer_relevance',0))} |",
        f"| Context Recall  | {avg_a.get('context_recall',0):.3f} | {avg_b.get('context_recall',0):.3f} | {delta(avg_b.get('context_recall',0), avg_a.get('context_recall',0))} |",
        f"| Context Precision| {avg_a.get('context_precision',0):.3f} | {avg_b.get('context_precision',0):.3f} | {delta(avg_b.get('context_precision',0), avg_a.get('context_precision',0))} |",
        f"| **Average**     | **{avg_a.get('avg_score',0):.3f}** | **{avg_b.get('avg_score',0):.3f}** | **{delta(avg_b.get('avg_score',0), avg_a.get('avg_score',0))}** |\n",
        "---\n",
        "## A/B Comparison Analysis\n",
        "- **Config A** (Dense-only): Chỉ dùng semantic search với OpenAI embeddings, không reranking.",
        "- **Config B** (Hybrid+RRF): Kết hợp semantic search + BM25 lexical search, merge bằng RRF.\n",
        "**Nhận xét:**",
    ]

    # Analysis
    if avg_b.get("context_recall", 0) > avg_a.get("context_recall", 0):
        lines.append("- Hybrid+RRF cải thiện Context Recall vì BM25 bắt được các keyword chính xác (tên người, điều luật).")
    if avg_b.get("faithfulness", 0) >= avg_a.get("faithfulness", 0):
        lines.append("- Faithfulness tương đương hoặc cao hơn nhờ context đa dạng hơn từ hybrid search.")
    lines.append("- RRF fusion cho phép kết quả từ cả 2 retriever, giảm miss rate.\n")

    lines.extend([
        "---\n",
        f"## Worst Performers (Config B, {len(worst)} câu thấp nhất)\n",
        "| # | Câu hỏi | Avg Score | Vấn đề |",
        "|---|---------|-----------|--------|",
    ])

    for i, r in enumerate(worst, 1):
        issue = "Thiếu context chi tiết" if r["context_recall"] < 0.5 else "Citation không đủ"
        lines.append(f"| {i} | {r['question'][:60]} | {r['avg_score']:.3f} | {issue} |")

    lines.extend([
        "\n---\n",
        "## Detailed Results (Config B)\n",
        "| Câu hỏi | Faith. | Relev. | Recall | Prec. | Trạng thái |",
        "|---------|--------|--------|--------|-------|------------|",
    ])

    for r in config_b_results:
        status = "Dat" if r["avg_score"] >= 0.7 else "Xem lai"
        q = r["question"][:50] + "..." if len(r["question"]) > 50 else r["question"]
        lines.append(
            f"| {q} | {r['faithfulness']:.2f} | {r['answer_relevance']:.2f} | "
            f"{r['context_recall']:.2f} | {r['context_precision']:.2f} | {status} |"
        )

    lines.extend([
        "\n---\n",
        "## Recommendations\n",
        "1. **Them data phap luat**: Hien tai chi co bai bao tin tuc, can them van ban phap luat de tra loi cau hoi ve dieu luat.",
        "2. **Cai thien chunking**: Loc bo cac chunk navigation/menu, chi giu noi dung chinh.",
        "3. **Cross-encoder reranking**: Dung Jina Reranker de cai thien Context Precision.",
        "4. **Conversation memory**: Them follow-up question support cho chatbot.",
    ])

    content = "\n".join(lines)
    RESULTS_PATH.write_text(content, encoding="utf-8")
    print(f"\nResults exported to {RESULTS_PATH}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("RAG Evaluation Pipeline")
    print("=" * 60)

    golden_dataset = json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(golden_dataset)} test cases\n")

    # Dùng subset nhỏ hơn để tiết kiệm API calls (6 câu hỏi)
    # Comment dòng này để chạy đầy đủ
    subset = golden_dataset[:6]

    print("--- Config A: Dense-only ---")
    results_a = run_evaluation(subset, rag_dense_only, "Dense-only")

    print("\n--- Config B: Hybrid + RRF ---")
    results_b = run_evaluation(subset, rag_hybrid_rerank, "Hybrid+RRF")

    avg_a = compute_averages(results_a)
    avg_b = compute_averages(results_b)

    print("\n=== RESULTS SUMMARY ===")
    print(f"{'Metric':<20} {'Config A':>12} {'Config B':>12}")
    print("-" * 46)
    for metric in ["faithfulness", "answer_relevance", "context_recall", "context_precision"]:
        print(f"{metric:<20} {avg_a.get(metric, 0):>12.3f} {avg_b.get(metric, 0):>12.3f}")
    print(f"{'avg_score':<20} {avg_a.get('avg_score', 0):>12.3f} {avg_b.get('avg_score', 0):>12.3f}")

    # Export nếu chạy full dataset
    if len(subset) == len(golden_dataset):
        export_results(results_a, results_b)
    else:
        print(f"\n[INFO] Chay subset {len(subset)}/{len(golden_dataset)} cau. De export results.md, chay full dataset.")
