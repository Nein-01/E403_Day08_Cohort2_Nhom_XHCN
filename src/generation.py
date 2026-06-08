"""
Generation + Citation Module (Group Project — Person 2).

Public API:
    from src.generation import generate_with_citation

Pipeline:
    Query
      └→ retrieve()              [src.retrieval — Person 1]
      └→ reorder_for_llm()       [lost-in-the-middle mitigation]
      └→ format_context()        [inject source labels]
      └→ OpenAI GPT-4o-mini      [generate answer with citation]
"""

import os
from dotenv import load_dotenv

load_dotenv()

try:
    from .retrieval import retrieve
except ImportError:
    # Fallback while Person 1's retrieval.py is being finalised
    from .task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION
# =============================================================================

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

SYSTEM_PROMPT = """Trả lời câu hỏi sau một cách toàn diện bằng tiếng Việt.
Với mọi khẳng định sự kiện hoặc thông tin, hãy chèn ngay trích dẫn trong ngoặc vuông
chỉ rõ nguồn cụ thể (ví dụ: [Luật Phòng chống ma tuý 2021, Điều 3]
hoặc [VnExpress, 2024]).

Nếu thông tin không được nêu rõ ràng trong context được cung cấp,
hãy trả lời 'Tôi không thể xác minh thông tin này từ nguồn hiện có'
thay vì đoán mò.

Quy tắc:
- Chỉ sử dụng thông tin từ context được cung cấp
- Mọi khẳng định sự kiện PHẢI có trích dẫn
- Nếu context không đủ, hãy nói rõ điều đó
- Cấu trúc câu trả lời thành các đoạn rõ ràng"""


# =============================================================================
# DOCUMENT REORDERING — tránh "lost in the middle"
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp lại chunks để LLM chú ý tốt hơn.

    LLM nhớ tốt thông tin ở đầu và cuối, bỏ sót thông tin ở giữa.
    Strategy: chunks quan trọng nhất (score cao) → đầu và cuối,
              chunks kém quan trọng → giữa.

    Ví dụ với 5 chunks (sorted by score desc):
      Input:  [0, 1, 2, 3, 4]
      Output: [0, 2, 4, 3, 1]  ← 0 đầu, 1 cuối, còn lại ở giữa
    """
    if len(chunks) <= 2:
        return chunks
    front = chunks[::2]          # vị trí 0, 2, 4, ... → đầu
    back = chunks[1::2][::-1]    # vị trí 1, 3, 5, ... → đảo → cuối
    return front + back


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """Format chunks thành context string với source label cho LLM."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("source", f"Source {i}")
        doc_type = meta.get("type", "unknown")
        parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type}]\n"
            f"{chunk['content']}\n"
        )
    return "\n---\n".join(parts)


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(
    query: str,
    top_k: int = TOP_K,
    use_reranking: bool = True,
) -> dict:
    """
    End-to-end RAG generation có citation.

    Args:
        query: Câu hỏi của user.
        top_k: Số chunks đưa vào context.
        use_reranking: Bật/tắt reranking (dùng cho A/B eval).

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng
            'retrieval_source': str  # 'hybrid' | 'pageindex' | 'none'
        }
    """
    chunks = retrieve(query, top_k=top_k, use_reranking=use_reranking)

    if not chunks:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none",
        }

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    user_message = f"Context:\n{context}\n\n---\n\nCâu hỏi: {query}"

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
    retrieval_source = chunks[0].get("source", "hybrid") if chunks else "none"

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": retrieval_source,
    }


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]

    for q in test_queries:
        print(f"\n{'=' * 70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
