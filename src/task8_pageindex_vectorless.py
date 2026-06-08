"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tại: https://pageindex.ai/
SDK: https://github.com/VectifyAI/PageIndex

Khi PAGEINDEX_API_KEY chưa được set, hàm pageindex_search trả về [] thay vì
raise exception để pipeline (task9) không bị crash khi dùng fallback.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def upload_documents():
    """Upload markdown documents lên PageIndex."""
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("Hãy set PAGEINDEX_API_KEY trong file .env")

    try:
        from pageindex import PageIndex
    except ImportError:
        raise ImportError("pip install pageindex")

    pi = PageIndex(api_key=PAGEINDEX_API_KEY)
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        pi.upload(
            content=content,
            metadata={"filename": md_file.name, "type": md_file.parent.name},
        )
        print(f"  ✓ Uploaded: {md_file.name}")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Fallback khi hybrid search không trả về kết quả tốt.

    Trả về [] nếu PAGEINDEX_API_KEY chưa được cấu hình.
    """
    if not PAGEINDEX_API_KEY:
        return []

    try:
        from pageindex import PageIndex
    except ImportError:
        return []

    try:
        pi = PageIndex(api_key=PAGEINDEX_API_KEY)
        results = pi.query(query=query, top_k=top_k)
        return [
            {
                "content": r.text,
                "score": r.score,
                "metadata": r.metadata,
                "source": "pageindex",
            }
            for r in results
        ]
    except Exception:
        return []


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()
        print("\nTest query:")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
