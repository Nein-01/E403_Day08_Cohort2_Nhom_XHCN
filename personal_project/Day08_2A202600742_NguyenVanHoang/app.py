"""
RAG Chatbot — Pháp luật Ma tuý Việt Nam
Group Project: Day 08 — RAG Pipeline v2

Chạy:
    streamlit run app.py

Features:
    - Hybrid search (Dense + BM25) + RRF + MMR
    - Citation trực tiếp từ nguồn pháp luật
    - Conversation memory (follow-up questions)
    - Hiển thị source documents + relevance score
    - Strict mode: chỉ trả lời từ tài liệu, không suy diễn
"""

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Thêm project root vào path
sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trợ lý Pháp luật Ma tuý",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.source-card {
    background-color: #f0f2f6;
    border-left: 3px solid #1f77b4;
    padding: 8px 12px;
    margin: 4px 0;
    border-radius: 4px;
    font-size: 0.85em;
}
.score-badge {
    background-color: #1f77b4;
    color: white;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 0.75em;
    float: right;
}
.legal-badge {
    background-color: #2ca02c;
    color: white;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 0.75em;
}
.news-badge {
    background-color: #ff7f0e;
    color: white;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 0.75em;
}
</style>
""", unsafe_allow_html=True)


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚖️ Cài đặt")

    st.markdown("### Retrieval Config")
    top_k = st.slider("Số chunks context", min_value=2, max_value=6, value=4)
    score_threshold = st.slider(
        "Fallback threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.05,
        help="RRF score thấp hơn ngưỡng này → dùng PageIndex fallback"
    )
    use_reranking = st.checkbox("Dùng reranking (RRF + MMR)", value=True)

    st.markdown("---")
    st.markdown("### Thông tin hệ thống")
    st.markdown("**Embedding:** BAAI/bge-m3")
    st.markdown("**Vector Store:** ChromaDB (local)")
    st.markdown("**Hybrid:** Dense + BM25 → RRF")
    st.markdown("**Dedup:** MMR (λ=0.7)")
    st.markdown("**LLM:** gpt-4o-mini (temp=0.1)")

    st.markdown("---")
    if st.button("🗑️ Xóa lịch sử chat"):
        st.session_state.messages = []
        st.session_state.retrieved_sources = []
        st.rerun()

    st.markdown("---")
    st.caption("Day 08 — RAG Pipeline v2 | Pháp luật Ma tuý VN")


# ─── SESSION STATE ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retrieved_sources" not in st.session_state:
    st.session_state.retrieved_sources = []


# ─── LOAD RAG PIPELINE ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Đang khởi động RAG pipeline...")
def load_pipeline():
    """Load pipeline 1 lần, cache lại."""
    from src.task4_chunking_indexing import get_collection, get_embedding_model
    # Trigger lazy loading
    get_embedding_model()
    get_collection()
    from src.task9_retrieval_pipeline import retrieve
    from src.task10_generation import generate_with_citation
    return retrieve, generate_with_citation


def is_pipeline_ready() -> bool:
    try:
        from src.task4_chunking_indexing import get_collection
        col = get_collection()
        return col.count() > 0
    except Exception:
        return False


# ─── MAIN UI ─────────────────────────────────────────────────────────────────
st.title("⚖️ Trợ lý Pháp luật Ma tuý Việt Nam")
st.caption("Hỏi về Luật Phòng chống ma tuý 2021, hình phạt, cai nghiện, và tin tức nghệ sĩ liên quan")

# Check OpenAI key
if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "sk-xxx":
    st.error("⚠️ Chưa có OPENAI_API_KEY. Điền vào file `.env` rồi restart app.")
    st.stop()

# Check pipeline
if not is_pipeline_ready():
    st.warning("⚠️ Vector store chưa có dữ liệu. Chạy `python -m src.task4_chunking_indexing` trước.")
    st.info("Sau khi index xong, refresh trang này.")
    st.stop()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📚 Sources ({len(msg['sources'])} chunks)", expanded=False):
                for i, src in enumerate(msg["sources"], 1):
                    meta = src.get("metadata", {})
                    source_name = meta.get("source", f"Source {i}")
                    doc_type = meta.get("type", "unknown")
                    score = src.get("score", 0.0)
                    badge_class = "legal-badge" if doc_type == "legal" else "news-badge"
                    st.markdown(
                        f'<div class="source-card">'
                        f'<span class="{badge_class}">{doc_type}</span> '
                        f'<strong>{source_name}</strong>'
                        f'<span class="score-badge">{score:.3f}</span><br/>'
                        f'<small>{src["content"][:200]}...</small>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

# Chat input
if question := st.chat_input("Đặt câu hỏi về pháp luật ma tuý..."):
    # Thêm conversation context vào query để xử lý follow-up
    context_query = question
    if len(st.session_state.messages) >= 2:
        last_q = st.session_state.messages[-2]["content"] if len(st.session_state.messages) >= 2 else ""
        # Simple follow-up detection
        followup_indicators = ["còn", "thêm", "tiếp", "nữa", "sao", "tại sao", "thế nào", "còn gì"]
        if any(kw in question.lower() for kw in followup_indicators):
            context_query = f"{last_q} | {question}"

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm và trả lời..."):
            try:
                retrieve, generate_with_citation = load_pipeline()
                result = generate_with_citation(context_query, top_k=top_k)
                answer = result["answer"]
                sources = result["sources"]
                retrieval_src = result["retrieval_source"]

                st.markdown(answer)

                if sources:
                    with st.expander(f"📚 Sources ({len(sources)} chunks) | via **{retrieval_src}**", expanded=False):
                        for i, src in enumerate(sources, 1):
                            meta = src.get("metadata", {})
                            source_name = meta.get("source", f"Source {i}")
                            doc_type = meta.get("type", "unknown")
                            score = src.get("score", 0.0)
                            badge_class = "legal-badge" if doc_type == "legal" else "news-badge"
                            st.markdown(
                                f'<div class="source-card">'
                                f'<span class="{badge_class}">{doc_type}</span> '
                                f'<strong>{source_name}</strong>'
                                f'<span class="score-badge">{score:.3f}</span><br/>'
                                f'<small>{src["content"][:200]}...</small>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except Exception as e:
                error_msg = f"❌ Lỗi: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg, "sources": []})
