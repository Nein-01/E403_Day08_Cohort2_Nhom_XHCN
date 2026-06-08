"""
Task 5 — Semantic Search Module.

Sử dụng OpenAI text-embedding-3-small + cosine similarity trên numpy array.
Index được load lazy từ data/index/ (tạo bởi task4).
"""

import os
import pickle
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

INDEX_DIR = Path(__file__).parent.parent / "data" / "index"

# Lazy-loaded globals để tránh load lại mỗi query
_embeddings: np.ndarray | None = None
_chunks: list[dict] | None = None


def _ensure_index():
    global _embeddings, _chunks
    if _embeddings is not None:
        return

    emb_path = INDEX_DIR / "embeddings.npy"
    chunks_path = INDEX_DIR / "chunks.pkl"

    if not emb_path.exists() or not chunks_path.exists():
        from .task4_chunking_indexing import run_pipeline
        print("⚠ Index chưa tồn tại. Đang build index (lần đầu chạy)...")
        run_pipeline()

    _embeddings = np.load(emb_path)
    with open(chunks_path, "rb") as f:
        _chunks = pickle.load(f)


def _embed_query(query: str) -> np.ndarray:
    """Embed query bằng OpenAI, normalize về unit vector."""
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(model="text-embedding-3-small", input=[query])
    emb = np.array(response.data[0].embedding, dtype=np.float32)
    return emb / (np.linalg.norm(emb) + 1e-9)


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa bằng cosine similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
        Sorted by score descending.
    """
    _ensure_index()

    query_emb = _embed_query(query)
    # Dot product = cosine similarity (vì đã normalize ở task4)
    scores = _embeddings @ query_emb

    top_indices = np.argsort(scores)[::-1][:top_k]

    return [
        {
            "content": _chunks[idx]["content"],
            "score": float(scores[idx]),
            "metadata": _chunks[idx]["metadata"],
        }
        for idx in top_indices
    ]


def reload_index():
    """Reset cached index (dùng sau khi rebuild)."""
    global _embeddings, _chunks
    _embeddings = None
    _chunks = None


if __name__ == "__main__":
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
