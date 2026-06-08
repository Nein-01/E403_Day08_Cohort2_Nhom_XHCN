"""
Task 6 — Lexical Search Module (BM25).

BM25 (Best Match 25) hoạt động như thế nào:
- TF (Term Frequency): từ xuất hiện nhiều trong document → điểm cao hơn
- IDF (Inverse Document Frequency): từ hiếm trong toàn corpus → quan trọng hơn
- Length normalization: document dài không bị ưu tiên quá mức
- Formula: Σ IDF(qi) * tf(qi,d)*(k1+1) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
  với k1=1.5 (term saturation), b=0.75 (length norm) — tốt hơn TF-IDF thuần

Bonus: giải thích cơ chế BM25 trong demo → +5 điểm
"""

from pathlib import Path
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

_corpus: list[dict] = []
_bm25: Optional[BM25Okapi] = None


def _load_corpus() -> list[dict]:
    corpus = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in str(md_file) else "news"
        corpus.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type},
        })
    return corpus


def _get_bm25() -> tuple:
    global _corpus, _bm25
    if _bm25 is None:
        _corpus = _load_corpus()
        if not _corpus:
            raise RuntimeError(f"No documents in {STANDARDIZED_DIR}. Run task3 first.")
        tokenized = [doc["content"].lower().split() for doc in _corpus]
        _bm25 = BM25Okapi(tokenized)
    return _bm25, _corpus


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    BM25 lexical search — tìm kiếm theo từ khóa.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
        Sorted by score descending.
    """
    bm25, corpus = _get_bm25()
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
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
