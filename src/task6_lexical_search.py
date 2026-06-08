"""
Task 6 — Lexical Search Module (BM25).

Sử dụng BM25Okapi với tokenization word-level (split theo khoảng trắng).
Corpus được load từ cùng data/index/chunks.pkl với task 5.

Cơ chế BM25:
  - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
  - Inverse Document Frequency (IDF): từ hiếm (ít doc chứa) → quan trọng hơn
  - Length normalization: document dài không bị ưu tiên quá mức
  - Formula: score(q,d) = Σ IDF(qi) × (tf × (k1+1)) / (tf + k1×(1-b+b×|d|/avgdl))
    với k1=1.5 (saturation), b=0.75 (length norm factor)

Bonus: BM25 bổ sung rất tốt cho semantic search — nắm bắt exact keyword match
mà dense retrieval đôi khi bỏ qua (tên người, điều luật cụ thể như "Điều 249").
"""

import pickle
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

INDEX_DIR = Path(__file__).parent.parent / "data" / "index"

# Lazy-loaded globals
_bm25: BM25Okapi | None = None
_chunks: list[dict] | None = None


def _load_corpus() -> list[dict]:
    global _chunks
    if _chunks is None:
        chunks_path = INDEX_DIR / "chunks.pkl"
        if not chunks_path.exists():
            from .task4_chunking_indexing import run_pipeline
            print("⚠ Index chưa tồn tại. Đang build index...")
            run_pipeline()
        with open(chunks_path, "rb") as f:
            _chunks = pickle.load(f)
    return _chunks


def build_bm25_index(corpus: list[dict]) -> BM25Okapi:
    """Xây dựng BM25 index từ corpus."""
    tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
    return BM25Okapi(tokenized_corpus)


def _get_bm25() -> BM25Okapi:
    global _bm25
    if _bm25 is None:
        _bm25 = build_bm25_index(_load_corpus())
    return _bm25


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
        Sorted by score descending.
    """
    chunks = _load_corpus()
    bm25 = _get_bm25()

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "content": chunks[idx]["content"],
                "score": float(scores[idx]),
                "metadata": chunks[idx]["metadata"],
            })
    return results


def reload_index():
    """Reset cached index."""
    global _bm25, _chunks
    _bm25 = None
    _chunks = None


if __name__ == "__main__":
    results = lexical_search("Điều 249 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
