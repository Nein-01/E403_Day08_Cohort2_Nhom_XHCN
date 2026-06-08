"""
Task 10 — Generation Có Citation + Document Reordering.

Thiết kế chính xác cho văn bản pháp luật:
  - temperature=0.1: gần như deterministic, không "sáng tác" điều khoản
  - top_p=0.85: giữ focus, loại bỏ token có xác suất thấp
  - SYSTEM_PROMPT nghiêm ngặt: chỉ cite từ context, không được suy diễn
  - Nếu context không đủ → trả lời "Tôi không thể xác nhận..."

Document Reordering (tránh "lost in the middle"):
  Input (by relevance): [1, 2, 3, 4, 5]
  Output:               [1, 3, 5, 4, 2]
  - Chunk quan trọng nhất ở ĐẦU (LLM chú ý nhất)
  - Chunk quan trọng thứ 2 ở CUỐI (LLM nhớ cuối tốt)
  - Chunk kém quan trọng ở GIỮA

Token budget (gpt-4o-mini):
  - MAX_CONTEXT_CHUNKS = 4: đủ evidence, không thừa
  - MAX_CHUNK_CHARS    = 500: mỗi chunk max 500 chars
  - Tổng context      ≈ 2000 chars ≈ 500 tokens
  - Chi phí           ≈ $0.000075 per query (rất rẻ)
"""

import os

from dotenv import load_dotenv

from .task9_retrieval_pipeline import retrieve

load_dotenv()

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODEL = "gpt-4o-mini"
TEMPERATURE = 0.1   # Pháp luật cần chính xác → gần deterministic
TOP_P = 0.85        # Giữ focus, loại token xác suất thấp
MAX_CONTEXT_CHUNKS = 4  # Vừa đủ evidence, tiết kiệm token

# ─── SYSTEM PROMPT ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Bạn là trợ lý pháp lý chuyên về pháp luật ma tuý Việt Nam.
Trả lời bằng tiếng Việt. Chỉ sử dụng thông tin được cung cấp trong phần Context.

Quy tắc bắt buộc:
1. Mỗi thông tin pháp lý phải có citation ngay sau: [Tên nguồn, Điều/Khoản]
   Ví dụ: [Luật Phòng chống ma tuý 2021, Điều 3] hoặc [VnExpress, 2024]
2. KHÔNG được suy diễn, đoán mò, hay tự thêm thông tin ngoài Context.
3. Nếu Context không có thông tin cần thiết → trả lời chính xác:
   "Tôi không tìm thấy thông tin này trong tài liệu pháp luật được cung cấp."
4. Không nói "theo tôi", "có thể", "tôi nghĩ" — chỉ trích dẫn nguồn.
5. Cấu trúc rõ ràng: tóm tắt ngắn → chi tiết có citation → kết luận."""


# ─── DOCUMENT REORDERING ─────────────────────────────────────────────────────

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks tránh "lost in the middle" (Liu et al. 2023).

    Chunks quan trọng (score cao) → đặt ở đầu và cuối context.
    Chunks ít quan trọng hơn → đặt ở giữa.

    Input: [rank1, rank2, rank3, rank4, rank5]
    Output: [rank1, rank3, rank5, rank4, rank2]
    """
    if len(chunks) <= 2:
        return chunks

    # Tách odd-index (ưu tiên) và even-index
    front = [chunks[i] for i in range(0, len(chunks), 2)]   # [0,2,4,...]
    back = [chunks[i] for i in range(1, len(chunks), 2)]    # [1,3,5,...]

    # Front đặt trước, back đặt sau (reversed để rank cao hơn ở cuối)
    return front + back[::-1]


# ─── CONTEXT FORMATTING ──────────────────────────────────────────────────────

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string có label source để LLM cite.

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string dùng cho prompt.
    """
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("source") or meta.get("filename", f"Nguồn {i}")
        doc_type = meta.get("type", "unknown")
        score = chunk.get("score", 0.0)
        parts.append(
            f"[Tài liệu {i} | {source} | {doc_type} | score={score:.3f}]\n"
            f"{chunk['content']}"
        )
    return "\n\n---\n\n".join(parts)


# ─── GENERATION ──────────────────────────────────────────────────────────────

def generate_with_citation(query: str, top_k: int = MAX_CONTEXT_CHUNKS) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks (hybrid + RRF + MMR)
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Gọi gpt-4o-mini với SYSTEM_PROMPT nghiêm ngặt
        5. Trả về answer + sources

    Args:
        query : câu hỏi của user
        top_k : số chunks context đưa vào LLM

    Returns:
        {
            'answer'           : str   — câu trả lời có citation
            'sources'          : list  — chunks đã dùng
            'retrieval_source' : str   — 'hybrid' hoặc 'pageindex'
        }
    """
    from openai import OpenAI

    # Step 1: Retrieve
    chunks = retrieve(query, top_k=top_k)

    if not chunks:
        return {
            "answer": "Tôi không tìm thấy thông tin này trong tài liệu được cung cấp.",
            "sources": [],
            "retrieval_source": "none",
        }

    # Step 2: Reorder
    reordered = reorder_for_llm(chunks)

    # Step 3: Format context
    context = format_context(reordered)

    # Step 4: Build prompt
    user_message = (
        f"Context từ tài liệu pháp luật và tin tức:\n\n"
        f"{context}\n\n"
        f"---\n\n"
        f"Câu hỏi: {query}"
    )

    # Step 5: Call LLM
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=1024,
    )

    answer = response.choices[0].message.content or ""
    retrieval_source = chunks[0].get("source", "hybrid") if chunks else "none"

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": retrieval_source,
    }


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Luật phòng chống ma tuý 2021 quy định gì về cai nghiện bắt buộc?",
        "Những nghệ sĩ nào bị bắt vì liên quan tới ma tuý?",
    ]
    for q in test_queries:
        print(f"\n{'=' * 70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
