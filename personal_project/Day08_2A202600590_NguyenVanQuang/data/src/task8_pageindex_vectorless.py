"""
Task 8 — PageIndex Vectorless RAG.

PageIndex là RAG không dùng vector — dùng structural understanding của document
(tree-based indexing) thay vì embedding. Phù hợp làm fallback khi hybrid search
không đủ confidence.

Flow:
  1. submit_document(pdf_path) → doc_id
  2. Lưu doc_ids vào cache JSON
  3. submit_query(doc_id, query) → retrieval_id
  4. Poll get_retrieval(retrieval_id) cho đến khi ready
  5. Trả về kết quả với source='pageindex'

Lưu ý: SDK chỉ nhận PDF. Các file .doc cần convert trước (xem convert_to_pdf()).
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
LANDING_LEGAL_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
DOC_IDS_CACHE = Path(__file__).parent.parent / "data" / "pageindex_docids.json"
PDF_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal_pdf"


def _get_client():
    from pageindex.client import PageIndexClient
    if not PAGEINDEX_API_KEY:
        raise EnvironmentError("PAGEINDEX_API_KEY chưa được set trong .env")
    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def convert_doc_to_pdf(doc_path: Path) -> Path | None:
    """Convert .doc/.docx → PDF bằng docx2pdf (yêu cầu Microsoft Word)."""
    pdf_path = PDF_DIR / f"{doc_path.stem}.pdf"
    if pdf_path.exists():
        return pdf_path
    try:
        from docx2pdf import convert
        PDF_DIR.mkdir(parents=True, exist_ok=True)
        convert(str(doc_path), str(pdf_path))
        return pdf_path
    except Exception:
        return None


def upload_documents() -> list[str]:
    """
    Upload PDF documents lên PageIndex.
    Trả về list doc_ids.
    Lưu cache vào data/pageindex_docids.json.
    """
    client = _get_client()
    doc_ids: list[str] = []

    # Tìm PDF files (ưu tiên file PDF trực tiếp, rồi convert .doc)
    pdf_files: list[Path] = list(LANDING_LEGAL_DIR.glob("*.pdf"))

    if not pdf_files:
        # Thử convert .doc → PDF
        for doc_file in LANDING_LEGAL_DIR.glob("*.doc*"):
            pdf = convert_doc_to_pdf(doc_file)
            if pdf:
                pdf_files.append(pdf)

    if not pdf_files:
        print("  ⚠ Không tìm thấy PDF files để upload")
        return []

    for pdf_path in pdf_files:
        print(f"  Uploading: {pdf_path.name}")
        try:
            resp = client.submit_document(str(pdf_path))
            doc_id = resp.get("doc_id") or resp.get("id", "")
            if doc_id:
                doc_ids.append(doc_id)
                print(f"    ✓ doc_id: {doc_id}")
        except Exception as e:
            print(f"    ✗ Lỗi upload {pdf_path.name}: {e}")

    # Lưu cache
    if doc_ids:
        DOC_IDS_CACHE.write_text(
            json.dumps({"doc_ids": doc_ids}, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  ✓ Saved {len(doc_ids)} doc_ids → {DOC_IDS_CACHE}")

    return doc_ids


def _load_doc_ids() -> list[str]:
    """Nạp doc_ids từ cache. Upload nếu cache chưa có."""
    if DOC_IDS_CACHE.exists():
        data = json.loads(DOC_IDS_CACHE.read_text(encoding="utf-8"))
        ids = data.get("doc_ids", [])
        if ids:
            return ids
    return upload_documents()


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval dùng PageIndex.
    Fallback khi hybrid search không đủ confidence.

    Args:
        query  : câu truy vấn
        top_k  : số kết quả tối đa

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict, 'source': 'pageindex'}
    """
    client = _get_client()
    doc_ids = _load_doc_ids()

    if not doc_ids:
        return []

    results: list[dict] = []

    for doc_id in doc_ids[:3]:  # Giới hạn 3 docs tránh quá nhiều API calls
        try:
            # Submit query
            q_resp = client.submit_query(doc_id=doc_id, query=query)
            retrieval_id = q_resp.get("retrieval_id", "")
            if not retrieval_id:
                continue

            # Poll cho đến khi ready (max 30s)
            for _ in range(15):
                r_resp = client.get_retrieval(retrieval_id)
                status = r_resp.get("status", "")
                if status in ("completed", "done", "success"):
                    break
                if status in ("failed", "error"):
                    break
                time.sleep(2)

            # Extract content
            chunks = (
                r_resp.get("results", [])
                or r_resp.get("chunks", [])
                or r_resp.get("data", [])
            )

            for i, chunk in enumerate(chunks[:top_k]):
                content = (
                    chunk.get("text")
                    or chunk.get("content")
                    or chunk.get("chunk_text", "")
                )
                score = float(chunk.get("score", 0.5 - i * 0.05))
                results.append({
                    "content": content,
                    "score": score,
                    "metadata": {
                        "doc_id": doc_id,
                        "source": chunk.get("source", doc_id),
                        "type": "legal",
                    },
                    "source": "pageindex",
                })

        except Exception as e:
            print(f"  ⚠ PageIndex query error (doc_id={doc_id}): {e}")
            continue

    # Sort and return top_k
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
