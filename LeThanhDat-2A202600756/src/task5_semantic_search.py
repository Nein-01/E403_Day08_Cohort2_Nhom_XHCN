"""
Task 5 — Semantic Search Module (dense retrieval via ChromaDB).
"""

from .task4_chunking_indexing import (
    COLLECTION_NAME,
    get_chroma_client,
    get_embedding_model,
)


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng cosine similarity trên ChromaDB.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
        Sorted by score descending.
    """
    model = get_embedding_model()
    query_embedding = model.encode(query).tolist()

    client = get_chroma_client()
    collection = client.get_collection(COLLECTION_NAME)

    n = min(top_k, collection.count())
    if n == 0:
        return []

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
        output.append({
            "content": doc,
            "score": float(1.0 - dist),  # cosine distance → similarity
            "metadata": meta,
        })

    return sorted(output, key=lambda x: x["score"], reverse=True)


if __name__ == "__main__":
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
