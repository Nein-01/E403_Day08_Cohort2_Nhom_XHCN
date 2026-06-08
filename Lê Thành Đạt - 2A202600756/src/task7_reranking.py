"""
Task 7 — Reranking Module.

Implements:
- RRF (Reciprocal Rank Fusion): gộp nhiều ranked lists, không cần API — default
- Cross-encoder: Jina Reranker v2 via API (cần JINA_API_KEY)
- MMR (Maximal Marginal Relevance): vừa relevant vừa diverse
"""

import os
from typing import Optional

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

JINA_API_KEY = os.getenv("JINA_API_KEY", "")


# =============================================================================
# Helpers
# =============================================================================

def _cosine_sim(a: list, b: list) -> float:
    a_arr, b_arr = np.array(a, dtype=float), np.array(b, dtype=float)
    norm = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    return float(np.dot(a_arr, b_arr) / norm) if norm > 0 else 0.0


# =============================================================================
# Reranking methods
# =============================================================================

def rerank_rrf(ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60) -> list[dict]:
    """
    Reciprocal Rank Fusion: gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))
    k=60 từ Cormack et al. 2009 — giảm ảnh hưởng của rank rất cao/thấp.
    Không cần score tuyệt đối, chỉ cần thứ hạng → robust với nhiều loại ranker.

    Args:
        ranked_lists: Nhiều danh sách kết quả (mỗi list từ 1 ranker)
        top_k: Số kết quả trả về
        k: Smoothing constant

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


def rerank_cross_encoder(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    Cross-encoder reranking via Jina Reranker v2 API (multilingual, tốt cho tiếng Việt).
    Falls back to score-based sort nếu không có API key.
    """
    if not candidates:
        return []

    if not JINA_API_KEY:
        print("  JINA_API_KEY not set — dùng original scores thay thế")
        return sorted(candidates, key=lambda x: x["score"], reverse=True)[:top_k]

    try:
        response = requests.post(
            "https://api.jina.ai/v1/rerank",
            headers={
                "Authorization": f"Bearer {JINA_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "jina-reranker-v2-base-multilingual",
                "query": query,
                "documents": [c["content"] for c in candidates],
                "top_n": top_k,
            },
            timeout=15,
        )
        response.raise_for_status()
        reranked = response.json()["results"]
        return [
            {**candidates[r["index"]], "score": r["relevance_score"]}
            for r in reranked
        ]
    except Exception as e:
        print(f"  Jina API error: {e} — fallback to original scores")
        return sorted(candidates, key=lambda x: x["score"], reverse=True)[:top_k]


def rerank_mmr(
    query_embedding: list,
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))
    λ=0.7: ưu tiên relevance (70%) hơn diversity (30%).
    Tốt khi muốn tránh trả về nhiều chunks trùng nội dung.
    """
    if not candidates:
        return []

    selected_idx: list[int] = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx, best_score = None, float("-inf")
        for idx in remaining:
            emb = candidates[idx].get("embedding", [])
            relevance = (
                _cosine_sim(query_embedding, emb) if emb
                else candidates[idx]["score"]
            )
            max_sim = max(
                (
                    _cosine_sim(emb, candidates[s].get("embedding", []))
                    for s in selected_idx
                    if candidates[s].get("embedding")
                ),
                default=0.0,
            )
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
            if mmr_score > best_score:
                best_score, best_idx = mmr_score, idx

        if best_idx is None:
            break
        selected_idx.append(best_idx)
        remaining.remove(best_idx)

    return [candidates[i] for i in selected_idx]


# =============================================================================
# Unified interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "rrf",
) -> list[dict]:
    """
    Unified reranking interface.

    method="rrf"          — default, không cần API key
    method="cross_encoder"— Jina API (cần JINA_API_KEY)
    method="mmr"          — cần embedding trong candidates
    """
    if not candidates:
        return []
    if method == "rrf":
        return rerank_rrf([candidates], top_k=top_k)
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    if method == "mmr":
        query_emb = candidates[0].get("embedding", [])
        return rerank_mmr(query_emb, candidates, top_k)
    raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dummy = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    results = rerank("hình phạt tàng trữ ma tuý", dummy, top_k=2)
    for r in results:
        print(f"[{r['score']:.4f}] {r['content']}")
