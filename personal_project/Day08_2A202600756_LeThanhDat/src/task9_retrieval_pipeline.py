"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Pipeline:
    Query
      ├→ Semantic Search (Task 5)  ──┐
      │                               ├→ RRF Merge → Rerank → Results
      ├→ Lexical Search  (Task 6)  ──┘
      │
      └→ Nếu best_score < threshold → Fallback: PageIndex (Task 8)
"""

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search

SCORE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5
RERANK_METHOD = "rrf"  # rrf không cần API key, hoạt động out-of-the-box


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Full retrieval pipeline với fallback logic.

    Args:
        query: Câu truy vấn
        top_k: Số kết quả cuối cùng
        score_threshold: Ngưỡng điểm tối thiểu — dưới ngưỡng → fallback PageIndex
        use_reranking: Có áp dụng reranking không

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict, 'source': str}
        source = 'hybrid' hoặc 'pageindex'
    """
    # Step 1: Chạy semantic + lexical search
    dense = semantic_search(query, top_k=top_k * 2)
    sparse = lexical_search(query, top_k=top_k * 2)

    # Step 2: Merge với RRF (Reciprocal Rank Fusion)
    merged = rerank_rrf([dense, sparse], top_k=top_k * 2)
    for item in merged:
        item.setdefault("source", "hybrid")

    # Step 3: Rerank
    if use_reranking and merged:
        final = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
    else:
        final = merged[:top_k]

    # Step 4: Fallback nếu kết quả quá yếu
    best_score = final[0]["score"] if final else 0.0
    if not final or best_score < score_threshold:
        print(f"  Hybrid score ({best_score:.3f}) < threshold ({score_threshold}) → PageIndex fallback")
        fallback = pageindex_search(query, top_k=top_k)
        if fallback:
            return fallback

    return final


if __name__ == "__main__":
    queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý",
        "Luật phòng chống ma tuý 2021 quy định gì về cai nghiện",
    ]
    for q in queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.4f}] [{r.get('source', '?')}] {r['content'][:80]}...")
