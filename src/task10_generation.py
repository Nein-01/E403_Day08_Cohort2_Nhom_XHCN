"""
Task 10 — Generation Có Citation.

- top_k=5: đủ evidence mà không quá dài gây lost in the middle
- top_p=0.9: nucleus sampling — đủ diverse, không quá random
- temperature=0.3: RAG cần factual output, ít sáng tạo

Document reordering (Liu et al. 2023 "Lost in the Middle"):
    Input order (by score):  [1, 2, 3, 4, 5]
    Output order:            [1, 3, 5, 4, 2]
    → Chunk quan trọng nhất ở đầu và cuối prompt, LLM chú ý hơn.
"""

import os

from dotenv import load_dotenv

load_dotenv()

from .task9_retrieval_pipeline import retrieve

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

SYSTEM_PROMPT = """Bạn là trợ lý pháp luật AI chuyên về lĩnh vực phòng, chống ma tuý tại Việt Nam.
Trả lời câu hỏi bằng tiếng Việt, dựa HOÀN TOÀN vào context được cung cấp.

Quy tắc bắt buộc:
1. Mỗi khẳng định sự thật PHẢI có trích dẫn ngay sau, ví dụ: [VnExpress, 2024] hoặc [Tuổi Trẻ, 2023]
2. Chỉ sử dụng thông tin trong context — KHÔNG suy đoán hay thêm thông tin ngoài
3. Nếu context không đủ để trả lời → nói rõ "Tôi không thể xác minh thông tin này từ nguồn hiện có"
4. Cấu trúc câu trả lời rõ ràng, có đoạn văn phân tách
5. Nếu câu hỏi liên quan đến nhiều nguồn, trích dẫn đầy đủ"""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle".

    Strategy: chunk điểm cao ở đầu và cuối, thấp ở giữa.
    [1,2,3,4,5] → [1,3,5,4,2]
    """
    if len(chunks) <= 2:
        return chunks

    # Tách chẵn/lẻ theo index
    evens = chunks[::2]     # index 0,2,4 — cao điểm, vào đầu
    odds = chunks[1::2]     # index 1,3,... — thấp hơn, vào cuối (đảo ngược)
    return evens + list(reversed(odds))


def format_context(chunks: list[dict]) -> str:
    """Format chunks thành context string có nhãn nguồn."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        title = meta.get("title", f"Tài liệu {i}")
        source_url = meta.get("source_url", "")
        crawled_at = (meta.get("crawled_at") or "")[:10]

        parts.append(
            f"[Tài liệu {i} | Nguồn: {title} | Ngày: {crawled_at}]\n"
            f"{chunk['content']}\n"
        )
    return "\n---\n".join(parts)


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.

    Returns:
        {
            'answer': str,
            'sources': list[dict],
            'retrieval_source': str
        }
    """
    # Step 1: Retrieve
    chunks = retrieve(query, top_k=top_k)

    if not chunks:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none",
        }

    # Step 2: Reorder (tránh lost in the middle)
    reordered = reorder_for_llm(chunks)

    # Step 3: Format context
    context = format_context(reordered)

    # Step 4: Build prompt
    user_message = f"Context:\n{context}\n\n---\n\nCâu hỏi: {query}"

    # Step 5: Call LLM
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none",
    }


if __name__ == "__main__":
    test_queries = [
        "Rapper Bình Gold bị bắt vì lý do gì?",
        "Ca sĩ Chu Bin bị bắt liên quan đến vụ việc nào?",
        "Những nghệ sĩ Việt nào bị khởi tố vì liên quan ma tuý?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
