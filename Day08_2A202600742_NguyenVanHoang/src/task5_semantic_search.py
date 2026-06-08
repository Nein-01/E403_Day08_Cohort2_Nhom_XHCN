"""
Task 5 — Semantic Search Module (Dense Retrieval).

Dùng ChromaDB + BAAI/bge-m3 embedding.
Score = 1 - cosine_distance (ChromaDB trả distance, ta convert sang similarity).
"""


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Dense retrieval bằng cosine similarity trên ChromaDB.

    Args:
        query  : câu truy vấn
        top_k  : số kết quả tối đa

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
        Sorted by score descending.
    """
    from .task4_chunking_indexing import get_collection, get_embedding_model

    collection = get_collection()
    count = collection.count()
    if count == 0:
        return []

    model = get_embedding_model()
    query_embedding = model.encode(query).tolist()

    n = min(top_k, count)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # ChromaDB cosine distance ∈ [0, 2]; similarity = 1 - distance
        score = float(1.0 - dist)
        output.append({"content": doc, "score": score, "metadata": meta})

    output.sort(key=lambda x: x["score"], reverse=True)
    return output


if __name__ == "__main__":
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
