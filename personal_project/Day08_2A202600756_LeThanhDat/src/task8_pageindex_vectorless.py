"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tại: https://pageindex.ai/
SDK: https://github.com/VectifyAI/PageIndex

PageIndex dùng structural understanding của document thay vì vector embedding
— không cần vector store, tốt cho document có cấu trúc rõ (luật, điều khoản).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def upload_documents():
    """Upload toàn bộ markdown documents lên PageIndex."""
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("PAGEINDEX_API_KEY không có trong .env — đăng ký tại https://pageindex.ai/")

    from pageindex import PageIndex

    pi = PageIndex(api_key=PAGEINDEX_API_KEY)
    uploaded = 0

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        pi.upload(
            content=content,
            metadata={"filename": md_file.name, "type": md_file.parent.name},
        )
        print(f"  Uploaded: {md_file.name}")
        uploaded += 1

    print(f"Done: {uploaded} documents uploaded to PageIndex")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval using PageIndex.
    Dùng làm fallback khi hybrid search score < threshold.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict, 'source': 'pageindex'}
    """
    if not PAGEINDEX_API_KEY:
        print("  PAGEINDEX_API_KEY not set — returning empty results")
        return []

    try:
        from pageindex import PageIndex

        pi = PageIndex(api_key=PAGEINDEX_API_KEY)
        results = pi.query(query=query, top_k=top_k)
        return [
            {
                "content": r.text,
                "score": r.score,
                "metadata": getattr(r, "metadata", {}),
                "source": "pageindex",
            }
            for r in results
        ]
    except Exception as e:
        print(f"  PageIndex error: {e}")
        return []


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("Set PAGEINDEX_API_KEY trong .env")
        print("Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents ...")
        upload_documents()
        print("\nTest query:")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
