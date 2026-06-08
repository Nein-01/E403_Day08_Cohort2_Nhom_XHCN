"""
Task 10 — Generation Có Citation.

top_k=5: đủ evidence mà không quá dài gây lost-in-the-middle
top_p=0.9: nucleus sampling — diverse nhưng không quá random
temperature=0.3: RAG cần factual, giảm hallucination
"""

import os

from dotenv import load_dotenv

load_dotenv()

from .task9_retrieval_pipeline import retrieve

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

SYSTEM_PROMPT = """Answer the following question comprehensively in Vietnamese.
For every statement of fact or claim, immediately insert a citation in brackets
linking to the specific source (e.g., [Luật Phòng chống ma tuý 2021, Điều 3]
or [VnExpress, 2024]).

If the information is not explicitly stated in the provided context or knowledge
base, state 'Tôi không thể xác minh thông tin này từ nguồn hiện có' rather than
guessing.

Rules:
- Only use information from the provided context
- Every factual claim MUST have a citation
- If context is insufficient, say so clearly
- Structure your answer with clear paragraphs"""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Tránh lost-in-the-middle: LLM nhớ tốt đầu và cuối, quên giữa.

    Sắp xếp lại: chunks quan trọng (rank cao) ở ĐẦU và CUỐI,
    chunks kém quan trọng hơn ở GIỮA.

    Input:  [rank1, rank2, rank3, rank4, rank5]
    Output: [rank1, rank3, rank5, rank4, rank2]
    """
    if len(chunks) <= 2:
        return chunks
    first_half = chunks[::2]    # index 0, 2, 4, ... → đặt ở đầu
    second_half = chunks[1::2]  # index 1, 3, 5, ... → đặt ở cuối (đảo ngược)
    return first_half + second_half[::-1]


def format_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("source", f"Source {i}")
        doc_type = meta.get("type", "unknown")
        parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type}]\n{chunk['content']}"
        )
    return "\n\n---\n\n".join(parts)


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.

    Returns:
        {
            'answer': str,           # Câu trả lời có citation [Nguồn, Năm]
            'sources': list[dict],   # Chunks đã dùng
            'retrieval_source': str  # 'hybrid' hoặc 'pageindex'
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

    # Step 2: Reorder để tránh lost-in-the-middle
    reordered = reorder_for_llm(chunks)

    # Step 3: Format context với source labels
    context = format_context(reordered)

    # Step 4: Build prompt
    user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"

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

    return {
        "answer": response.choices[0].message.content,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid"),
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
        print(f"\n[{len(result['sources'])} chunks | via {result['retrieval_source']}]")
