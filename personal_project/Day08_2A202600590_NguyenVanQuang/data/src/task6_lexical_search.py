"""
Task 6 — Lexical Search Module (BM25 Okapi).

BM25 hoạt động như thế nào:
  score(q, d) = Σ IDF(t) × tf(t,d)×(k1+1) / (tf(t,d) + k1×(1 - b + b×|d|/avgdl))

  - TF  : tần suất từ trong document — từ xuất hiện nhiều → điểm cao hơn (nhưng bão hoà)
  - IDF : từ hiếm trong corpus → quan trọng hơn (IDF cao)
  - k1  : điều chỉnh bão hoà TF (k1=1.5 → bão hoà sớm, phù hợp văn bản pháp lý)
  - b   : chuẩn hoá độ dài (b=0.75 → document dài không bị lợi thế quá mức)

Khác với TF-IDF thuần tuý: BM25 có chuẩn hoá độ dài document và bão hoà TF,
nên hiệu quả hơn cho văn bản dài như pháp luật.

Cache: data/bm25_cache.json (tạo bởi Task 4).
"""

import json
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

_BM25_CACHE = Path(__file__).parent.parent / "data" / "bm25_cache.json"

# Module-level singletons — build 1 lần, tái dùng
_corpus: list[dict] | None = None
_bm25: BM25Okapi | None = None


def _load_corpus() -> list[dict]:
    global _corpus
    if _corpus is None:
        if not _BM25_CACHE.exists():
            raise FileNotFoundError(
                f"BM25 cache chưa có tại {_BM25_CACHE}. Chạy Task 4 trước."
            )
        _corpus = json.loads(_BM25_CACHE.read_text(encoding="utf-8"))
    return _corpus


def _get_bm25() -> BM25Okapi:
    global _bm25
    if _bm25 is None:
        corpus = _load_corpus()
        tokenized = [doc["content"].lower().split() for doc in corpus]
        _bm25 = BM25Okapi(tokenized, k1=1.5, b=0.75)
    return _bm25


def build_bm25_index(corpus: list[dict]) -> BM25Okapi:
    """Xây BM25 index từ corpus (list of {'content': str, 'metadata': dict})."""
    tokenized = [doc["content"].lower().split() for doc in corpus]
    return BM25Okapi(tokenized, k1=1.5, b=0.75)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    BM25 keyword search.

    Args:
        query  : câu truy vấn
        top_k  : số kết quả tối đa

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
        Sorted by BM25 score descending, chỉ trả về results có score > 0.
    """
    corpus = _load_corpus()
    bm25 = _get_bm25()

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if float(scores[idx]) > 0:
            results.append({
                "content": corpus[idx]["content"],
                "score": float(scores[idx]),
                "metadata": corpus[idx]["metadata"],
            })

    return results


if __name__ == "__main__":
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
