"""
Task 4 — Chunking & Indexing

Chunking: RecursiveCharacterTextSplitter
  - CHUNK_SIZE = 600: đủ 1-2 điều khoản pháp luật + ngữ cảnh
  - CHUNK_OVERLAP = 80: 13% overlap giữ ngữ cảnh tại ranh giới chunk
  - Separators ưu tiên tách theo đoạn văn, rồi câu — tránh cắt giữa điều khoản

Embedding: BAAI/bge-m3
  - Multilingual, tốt nhất cho tiếng Việt
  - 1024 dimensions
  - Chạy local, không cần API key, không mất chi phí

Vector Store: ChromaDB (persistent local)
  - Không cần Docker hay cloud account
  - Cosine similarity built-in
  - JSON cache song song để BM25 (Task 6) dùng
"""

import json
from functools import lru_cache
from pathlib import Path

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
DB_DIR = Path(__file__).parent.parent / "data" / "chromadb"
BM25_CACHE = Path(__file__).parent.parent / "data" / "bm25_cache.json"

# Chunking config
CHUNK_SIZE = 600
CHUNK_OVERLAP = 80
CHUNKING_METHOD = "recursive"

# Embedding config
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

COLLECTION_NAME = "drug_law_docs"

# Fallback model nhỏ hơn nếu bge-m3 chưa download xong
_FALLBACK_MODEL = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """
    Lazy singleton embedding model.
    Ưu tiên BAAI/bge-m3 (tốt cho tiếng Việt, 1024 dim).
    Fallback: all-MiniLM-L6-v2 (22MB, đã cache sẵn).
    """
    try:
        print(f"  Loading embedding model: {EMBEDDING_MODEL}")
        model = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
        print(f"  ✓ Loaded {EMBEDDING_MODEL} (local cache)")
        return model
    except Exception:
        print(f"  ⚠ {EMBEDDING_MODEL} chưa download xong → dùng fallback: {_FALLBACK_MODEL}")
        return SentenceTransformer(_FALLBACK_MODEL)


def get_chroma_client() -> chromadb.PersistentClient:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(DB_DIR))


def get_collection() -> chromadb.Collection:
    """Lấy collection ChromaDB, tạo mới nếu chưa có."""
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str, ...}}
    """
    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if len(content) < 100:
            continue
        doc_type = "legal" if "legal" in str(md_file) else "news"
        documents.append({
            "content": content,
            "metadata": {
                "source": md_file.stem,
                "filename": md_file.name,
                "type": doc_type,
            },
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents bằng RecursiveCharacterTextSplitter.

    Returns:
        List of {'content': str, 'metadata': dict}
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Ưu tiên tách theo đoạn/câu để giữ nguyên vẹn điều khoản pháp luật
        separators=["\n\n## ", "\n\n### ", "\n\n", "\n", ". ", "，", " ", ""],
        length_function=len,
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, text in enumerate(splits):
            text = text.strip()
            if len(text) < 20:
                continue
            chunks.append({
                "content": text,
                "metadata": {**doc["metadata"], "chunk_index": i},
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Embed tất cả chunks bằng BAAI/bge-m3."""
    model = get_embedding_model()
    texts = [c["content"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """Index chunks vào ChromaDB và lưu JSON cache cho BM25."""
    client = get_chroma_client()

    # Xóa collection cũ để rebuild sạch
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Insert theo batch để tránh memory spike
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i: i + batch_size]
        collection.add(
            documents=[c["content"] for c in batch],
            embeddings=[c["embedding"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
            ids=[f"chunk_{i + j}" for j in range(len(batch))],
        )

    # Lưu BM25 cache (không cần embedding, chỉ cần text + metadata)
    BM25_CACHE.parent.mkdir(parents=True, exist_ok=True)
    bm25_data = [
        {"content": c["content"], "metadata": c["metadata"]} for c in chunks
    ]
    BM25_CACHE.write_text(
        json.dumps(bm25_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 60)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking : {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Store    : ChromaDB @ {DB_DIR}")
    print("=" * 60)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")
    if not docs:
        print("⚠  No documents. Run Task 3 first.")
        return

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print(f"✓ Indexed to ChromaDB")
    print(f"✓ BM25 cache: {BM25_CACHE}")


if __name__ == "__main__":
    run_pipeline()
