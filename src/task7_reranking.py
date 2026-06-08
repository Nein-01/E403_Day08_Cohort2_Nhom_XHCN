"""
Task 7 — Reranking Module.

Phương pháp chính: RRF (Reciprocal Rank Fusion)
Phương pháp phụ: Cross-encoder (stub, cần API key Jina/Cohere) và MMR.

RRF được chọn làm default vì:
  - Không cần model nặng hay API key
  - Hiệu quả khi gộp kết quả từ dense + sparse search
  - Công thức: RRF(d) = Σ 1/(k + rank_r(d)) với k=60 (Cormack et al. 2009)
  - Rank thấp hơn (kết quả tốt hơn) → điểm cao hơn
  - Tài liệu xuất hiện ở nhiều ranker → được ưu tiên
"""

from typing import Optional


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank bằng cross-encoder model (Jina Reranker API).
    Stub — cần JINA_API_KEY trong .env.
    """
    import os, requests
    api_key = os.getenv("JINA_API_KEY", "")
    if not api_key:
        raise NotImplementedError(
            "JINA_API_KEY chưa set trong .env. "
            "Đăng ký tại https://jina.ai/reranker/"
        )

    response = requests.post(
        "https://api.jina.ai/v1/rerank",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "jina-reranker-v2-base-multilingual",
            "query": query,
            "documents": [c["content"] for c in candidates],
            "top_n": top_k,
        },
        timeout=30,
    )
    response.raise_for_status()
    reranked = response.json()["results"]
    return [
        {**candidates[r["index"]], "score": r["relevance_score"]}
        for r in reranked
    ]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — vừa relevant vừa diverse.
    MMR = λ × sim(query, doc) − (1−λ) × max(sim(doc, selected_docs))
    """
    import numpy as np

    if not candidates:
        return []

    q = np.array(query_embedding, dtype=np.float32)
    q = q / (np.linalg.norm(q) + 1e-9)

    embs = []
    for c in candidates:
        e = np.array(c.get("embedding", [0.0] * len(q)), dtype=np.float32)
        e = e / (np.linalg.norm(e) + 1e-9)
        embs.append(e)
    embs = np.array(embs)

    relevance = embs @ q

    selected: list[int] = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float("-inf")

        for idx in remaining:
            rel = float(relevance[idx])
            if selected:
                sims = [float(embs[idx] @ embs[s]) for s in selected]
                max_sim = max(sims)
            else:
                max_sim = 0.0
            mmr_score = lambda_param * rel - (1 - lambda_param) * max_sim
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        selected.append(best_idx)
        remaining.remove(best_idx)

    return [
        {**candidates[i], "score": float(relevance[i])}
        for i in selected
    ]


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))
    k=60: smoothing constant từ Cormack et al. 2009

    Args:
        ranked_lists: Danh sách các ranked lists (mỗi list từ 1 retriever)
        top_k: Số kết quả cuối cùng
        k: Smoothing constant (default=60)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map[key] = item

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = score
        results.append(item)

    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "rrf",  # default rrf vì không cần API key
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số kết quả sau rerank
        method: "rrf" | "cross_encoder" | "mmr"
    """
    if method == "rrf":
        return rerank_rrf([candidates], top_k=top_k)
    elif method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        raise NotImplementedError("Call rerank_mmr directly with query_embedding")
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dummy = [
        {"content": "Điều 249: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    results = rerank_rrf([dummy[:2], dummy[1:]], top_k=3)
    for r in results:
        print(f"[{r['score']:.4f}] {r['content']}")
