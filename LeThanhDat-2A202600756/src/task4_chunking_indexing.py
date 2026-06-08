"""
Task 4 — Chunking & Indexing vào Vector Store (ChromaDB).

Chunking: RecursiveCharacterTextSplitter
- chunk_size=500: đủ dài để giữ context, đủ ngắn cho embedding hiệu quả
- chunk_overlap=50: tránh mất thông tin tại ranh giới chunk

Embedding: BAAI/bge-m3
- Multilingual, hỗ trợ tiếng Việt tốt nhất trong các free model
- 1024 dimensions

Vector Store: ChromaDB
- Local persistent storage, không cần Docker hay cloud account
- Hỗ trợ cosine similarity search built-in
"""

from pathlib import Path
from typing import Optional

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
VECTORSTORE_DIR = Path(__file__).parent.parent / "data" / "vectorstore"

# chunk_size=500 đủ context, overlap=50 tránh mất thông tin tại ranh giới
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

# BAAI/bge-m3: multilingual, tốt cho tiếng Việt pháp lý, 1024 dim
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

COLLECTION_NAME = "drug_law_docs"

_chroma_client: Optional[chromadb.PersistentClient] = None
_embedding_model: Optional[SentenceTransformer] = None


def get_chroma_client() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
    return _chroma_client


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL} ...")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def load_documents() -> list[dict]:
    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in str(md_file) else "news"
        documents.append({
            "content": content,
            "metadata": {
                "source": md_file.name,
                "type": doc_type,
                "path": str(md_file),
            },
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": i},
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    model = get_embedding_model()
    texts = [c["content"] for c in chunks]
    print(f"Embedding {len(texts)} chunks ...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    client = get_chroma_client()

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        collection.add(
            ids=[f"chunk_{i + j}" for j in range(len(batch))],
            documents=[c["content"] for c in batch],
            embeddings=[c["embedding"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )

    print(f"Indexed {len(chunks)} chunks into '{COLLECTION_NAME}'")


def run_pipeline():
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: ChromaDB @ {VECTORSTORE_DIR}")
    print("=" * 50)

    docs = load_documents()
    if not docs:
        print(f"No documents in {STANDARDIZED_DIR}. Run task3 first.")
        return
    print(f"\nLoaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)

    index_to_vectorstore(chunks)
    print("Done!")


if __name__ == "__main__":
    run_pipeline()
