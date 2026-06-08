"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Pipeline:
    Query
      ├→ Semantic Search (task5)  ──┐
      │                              ├→ RRF Merge → top_k kết quả
      ├→ Lexical Search (task6)   ──┘
      │
      └→ Nếu semantic score thấp < threshold
            └→ Fallback: PageIndex (task8, trả [] nếu chưa cấu hình)
"""

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank_rrf
from .task8_pageindex_vectorless import pageindex_search

SCORE_THRESHOLD = 0.25   # ngưỡng cosine similarity — dưới đây thì coi là kém
DEFAULT_TOP_K = 5
RERANK_METHOD = "rrf"


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Args:
        query: Câu truy vấn
        top_k: Số kết quả cuối cùng
        score_threshold: Ngưỡng cosine similarity để quyết định fallback
        use_reranking: Có áp dụng RRF merge không

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    # Step 1: Song song semantic + lexical
    dense_results = semantic_search(query, top_k=top_k * 2)
    sparse_results = lexical_search(query, top_k=top_k * 2)

    best_semantic_score = dense_results[0]["score"] if dense_results else 0.0

    # Step 2: Merge bằng RRF
    if use_reranking and (dense_results or sparse_results):
        lists_to_merge = [l for l in [dense_results, sparse_results] if l]
        merged = rerank_rrf(lists_to_merge, top_k=top_k)
    else:
        merged = dense_results[:top_k]

    for item in merged:
        item["source"] = "hybrid"

    # Step 3: Fallback PageIndex nếu semantic score thấp
    if best_semantic_score < score_threshold:
        print(f"  ⚠ Best semantic score ({best_semantic_score:.3f}) < threshold ({score_threshold}). Trying PageIndex fallback...")
        fallback = pageindex_search(query, top_k=top_k)
        if fallback:
            return fallback

    return merged


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý năm 2024",
        "Rapper Bình Gold bị bắt vì lý do gì",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.4f}] [{r['source']}] {r['content'][:80]}...")
