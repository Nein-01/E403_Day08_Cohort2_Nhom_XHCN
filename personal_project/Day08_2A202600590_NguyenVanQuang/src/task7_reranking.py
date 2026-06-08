"""
Task 7 — Reranking Module: RRF + MMR.

Hai kỹ thuật kết hợp:

1. RRF (Reciprocal Rank Fusion) — gộp nhiều ranked lists:
   RRF(d) = Σ 1 / (k + rank_r(d))
   k=60 (Cormack et al. 2009) — smoothing tránh top-1 chiếm hết điểm.
   Ưu điểm: không cần score tuyệt đối, chỉ cần thứ tự; robust với scale khác nhau.

2. MMR (Maximal Marginal Relevance) — chống trùng lặp context:
   MMR(d) = λ × sim(query, d) − (1−λ) × max_{s∈Selected} sim(d, s)
   λ=0.7: ưu tiên relevance 70%, diversity 30%.
   Ưu điểm: tránh LLM nhận 4 đoạn cùng nội dung → tiết kiệm token, tăng chất lượng.

3. rerank() — interface chính: re-score candidates bằng cosine similarity.
   Kết hợp original score (0.4) + semantic similarity (0.6).
"""

import numpy as np


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    Args:
        ranked_lists : List of ranked result lists (mỗi list từ 1 ranker)
        top_k        : Số kết quả cuối cùng
        k            : Smoothing constant (default=60)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            # Dùng 100 ký tự đầu làm key để dedup nội dung gần giống
            key = item["content"][:200].strip()
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in content_map:
                content_map[key] = item

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content_key, score in sorted_items[:top_k]:
        item = content_map[content_key].copy()
        item["score"] = float(score)
        results.append(item)

    return results


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ × sim(query, doc) − (1−λ) × max(sim(doc, selected_docs))

    Args:
        query_embedding : Vector embedding của query
        candidates      : List of {'content': str, 'score': float, 'metadata': dict}
                          Cần có key 'embedding' hoặc sẽ embed on-the-fly.
        top_k           : Số kết quả
        lambda_param    : Trade-off relevance (1.0) vs diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    if not candidates:
        return []

    q_emb = np.array(query_embedding, dtype=np.float32)

    # Embed nếu candidates chưa có embedding
    if "embedding" not in candidates[0]:
        from .task4_chunking_indexing import get_embedding_model
        model = get_embedding_model()
        texts = [c["content"] for c in candidates]
        embeddings = model.encode(texts, show_progress_bar=False)
        for c, emb in zip(candidates, embeddings):
            c["embedding"] = emb
    else:
        embeddings = [np.array(c["embedding"], dtype=np.float32) for c in candidates]

    candidate_embs = [np.array(c["embedding"], dtype=np.float32) for c in candidates]

    selected_indices: list[int] = []
    remaining_indices = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float("-inf")

        for idx in remaining_indices:
            relevance = _cosine_sim(q_emb, candidate_embs[idx])

            max_sim_to_selected = 0.0
            for sel_idx in selected_indices:
                sim = _cosine_sim(candidate_embs[idx], candidate_embs[sel_idx])
                max_sim_to_selected = max(max_sim_to_selected, sim)

            mmr_score = (
                lambda_param * relevance
                - (1.0 - lambda_param) * max_sim_to_selected
            )

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

    return [candidates[i] for i in selected_indices]


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Re-score candidates bằng cosine similarity query↔content.
    Gộp với original score theo trọng số.
    """
    if not candidates:
        return []

    from .task4_chunking_indexing import get_embedding_model
    model = get_embedding_model()

    q_emb = model.encode(query)
    texts = [c["content"] for c in candidates]
    doc_embs = model.encode(texts, show_progress_bar=False)

    scored = []
    for c, d_emb in zip(candidates, doc_embs):
        sem_score = _cosine_sim(q_emb, d_emb)
        # Blend: 40% original score + 60% semantic re-score
        orig = c.get("score", 0.0)
        blended = 0.4 * orig + 0.6 * sem_score
        item = c.copy()
        item["score"] = float(blended)
        scored.append(item)

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query      : câu truy vấn
        candidates : danh sách candidates từ retrieval
        top_k      : số kết quả sau rerank
        method     : "cross_encoder" | "mmr" | "rrf"

    Returns:
        List of top_k reranked candidates.
    """
    if not candidates:
        return []

    if method == "rrf":
        # Treat single list as 1 ranker
        return rerank_rrf([candidates], top_k=top_k)
    elif method == "mmr":
        from .task4_chunking_indexing import get_embedding_model
        model = get_embedding_model()
        q_emb = model.encode(query).tolist()
        return rerank_mmr(q_emb, candidates, top_k=top_k)
    else:
        # default: cross_encoder (semantic re-scoring)
        return rerank_cross_encoder(query, candidates, top_k=top_k)


if __name__ == "__main__":
    dummy = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2 đến 7 năm", "score": 0.6, "metadata": {}},
    ]
    results = rerank("hình phạt tàng trữ ma tuý", dummy, top_k=2)
    for r in results:
        print(f"[{r['score']:.4f}] {r['content']}")
