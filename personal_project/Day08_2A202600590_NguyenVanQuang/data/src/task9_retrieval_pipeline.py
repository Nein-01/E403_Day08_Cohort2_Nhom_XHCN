"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Flow:
  Query
    ├→ Semantic Search (Task 5)  ──┐
    │                               ├→ RRF Merge → Rerank → MMR Dedup → Results
    ├→ Lexical Search (Task 6)  ──┘
    │
    └→ Nếu best_score < threshold → Fallback: PageIndex (Task 8)

Thiết kế tiết kiệm token:
  - Dense  : top 20 candidates
  - Sparse : top 20 candidates
  - RRF    : merge → top 12
  - Rerank : top 8 (semantic re-scoring)
  - MMR    : select 4 diverse chunks → đây là gì LLM nhận được
  - Mỗi chunk tối đa 500 chars → tổng context ~2000 chars ≈ 500 tokens
"""

import numpy as np

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank_rrf, rerank_mmr, rerank
from .task8_pageindex_vectorless import pageindex_search

# ─── CONFIG ───────────────────────────────────────────────────────────────────
SCORE_THRESHOLD = 0.3  # RRF score dưới ngưỡng này → fallback PageIndex
DEFAULT_TOP_K = 5
CANDIDATES_DENSE = 20   # lấy nhiều để RRF có đủ lựa chọn
CANDIDATES_SPARSE = 20
RRF_INTERMEDIATE = 12   # sau RRF trước rerank
RERANK_CANDIDATES = 8   # sau rerank trước MMR
MAX_CHUNK_CHARS = 500   # giới hạn mỗi chunk để tiết kiệm token LLM


def _truncate_chunk(item: dict, max_chars: int = MAX_CHUNK_CHARS) -> dict:
    """Truncate content để tiết kiệm token. Không cắt giữa câu nếu có thể."""
    content = item["content"]
    if len(content) <= max_chars:
        return item
    truncated = content[:max_chars]
    # Cố tìm điểm kết thúc câu
    for sep in [". ", ".\n", "! ", "? "]:
        idx = truncated.rfind(sep)
        if idx > max_chars * 0.7:
            truncated = truncated[: idx + 1]
            break
    item = item.copy()
    item["content"] = truncated.strip()
    return item


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline với hybrid search + RRF + MMR + PageIndex fallback.

    Args:
        query           : câu truy vấn
        top_k           : số kết quả cuối cùng (dùng cho LLM)
        score_threshold : nếu top RRF score < ngưỡng → dùng PageIndex
        use_reranking   : có áp dụng semantic reranking không

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict, 'source': str}
        source = 'hybrid' hoặc 'pageindex'
    """
    # ── Step 1: Chạy dense + sparse search ──────────────────────────────────
    try:
        dense_results = semantic_search(query, top_k=CANDIDATES_DENSE)
    except Exception:
        dense_results = []

    try:
        sparse_results = lexical_search(query, top_k=CANDIDATES_SPARSE)
    except Exception:
        sparse_results = []

    # ── Step 2: RRF merge ───────────────────────────────────────────────────
    merged = rerank_rrf(
        [r for r in [dense_results, sparse_results] if r],
        top_k=RRF_INTERMEDIATE,
    )
    for item in merged:
        item["source"] = "hybrid"

    # ── Step 3: Semantic reranking ──────────────────────────────────────────
    if use_reranking and merged:
        reranked = rerank(query, merged, top_k=RERANK_CANDIDATES)
    else:
        reranked = merged[:RERANK_CANDIDATES]

    # ── Step 4: MMR dedup — chống trùng lặp context ─────────────────────────
    if reranked:
        from .task4_chunking_indexing import get_embedding_model
        model = get_embedding_model()
        q_emb = model.encode(query).tolist()
        final = rerank_mmr(q_emb, reranked, top_k=top_k, lambda_param=0.7)
    else:
        final = []

    # ── Step 5: Fallback PageIndex nếu score thấp ───────────────────────────
    top_score = final[0]["score"] if final else 0.0
    if top_score < score_threshold:
        print(
            f"  ⚠ Hybrid score ({top_score:.3f}) < threshold ({score_threshold})"
            " → Fallback PageIndex"
        )
        try:
            fallback = pageindex_search(query, top_k=top_k)
            if fallback:
                return [_truncate_chunk(r) for r in fallback[:top_k]]
        except Exception:
            pass

    # Truncate để tiết kiệm token
    return [_truncate_chunk(r) for r in final[:top_k]]


if __name__ == "__main__":
    queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ bị bắt vì sử dụng ma tuý",
        "Luật phòng chống ma tuý 2021 quy định về cai nghiện",
    ]
    for q in queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.4f}] [{r['source']}] {r['content'][:80]}...")
