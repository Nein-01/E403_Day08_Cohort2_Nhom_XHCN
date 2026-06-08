"""
Task 4 — Chunking & Indexing vào Vector Store.

Lựa chọn:
- Chunking: RecursiveCharacterTextSplitter tự implement
  chunk_size=800 char, overlap=100 char
  Vì sao 800? Vừa đủ để capture 1 đoạn tin tức/pháp lý có nghĩa,
  không quá dài gây noise cho embedding.
  Overlap 100 để không mất ngữ cảnh ở ranh giới chunk.

- Embedding: OpenAI text-embedding-3-small (1536 dim)
  Vì sao? Multilingual tốt (hỗ trợ tiếng Việt), API key sẵn có,
  không cần cài model local.

- Vector Store: Numpy in-memory (lưu file .npy + .pkl)
  Vì sao? Corpus nhỏ (~100-200 chunks từ 7 bài báo), cosine similarity
  bằng numpy đủ nhanh, không cần Weaviate/FAISS.
"""

import json
import os
import pickle
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
INDEX_DIR = Path(__file__).parent.parent / "data" / "index"

# chunk_size=800: đủ để capture 1 đoạn văn có nghĩa trong tin tức tiếng Việt
CHUNK_SIZE = 600
# overlap=80: giữ ngữ cảnh ở ranh giới giữa các chunk
CHUNK_OVERLAP = 80
# Giới hạn tuyệt đối để tránh vượt quá 8192 token của OpenAI
MAX_CHARS_PER_CHUNK = 4000
CHUNKING_METHOD = "recursive"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
VECTOR_STORE = "numpy_file"


# =============================================================================
# TEXT SPLITTING (Recursive Character Text Splitter)
# =============================================================================

def _split_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Recursive character splitter — tách theo paragraph, rồi line, rồi câu."""
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    separators = ["\n\n", "\n", ". ", " ", ""]

    for sep in separators:
        if sep == "":
            # Fall back: split by character
            chunks = []
            for i in range(0, len(text), chunk_size - chunk_overlap):
                chunk = text[i : i + chunk_size].strip()
                if chunk:
                    chunks.append(chunk)
            return chunks

        if sep not in text:
            continue

        parts = text.split(sep)
        chunks: list[str] = []
        current = ""

        for part in parts:
            if not part.strip():
                continue
            candidate = (current + sep + part) if current else part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current.strip())
                    overlap_text = current[-chunk_overlap:] if len(current) > chunk_overlap else current
                    current = overlap_text + sep + part
                else:
                    # part itself > chunk_size → recurse with next separator
                    sub = _split_text(part, chunk_size, chunk_overlap)
                    chunks.extend(sub)
                    current = ""

        if current.strip():
            chunks.append(current.strip())

        return [c for c in chunks if len(c.strip()) >= 30]

    return [text.strip()] if text.strip() else []


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """Đọc toàn bộ JSON files từ data/standardized/."""
    documents = []
    for json_file in sorted(STANDARDIZED_DIR.glob("*.json")):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            markdown = data.get("markdown", "")
            title = data.get("title", json_file.stem)
            source_url = data.get("source_url", "")
            crawled_at = data.get("crawled_at", "")

            if markdown and data.get("success", False):
                documents.append({
                    "content": f"# {title}\n\n{markdown}",
                    "metadata": {
                        "source": json_file.name,
                        "title": title,
                        "source_url": source_url,
                        "crawled_at": crawled_at,
                        "type": "news",
                    },
                })
        except Exception as e:
            print(f"  [WARN] Loi doc {json_file.name}: {e}")

    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chunk documents theo RecursiveCharacterTextSplitter."""
    chunks = []
    for doc in documents:
        splits = _split_text(doc["content"], CHUNK_SIZE, CHUNK_OVERLAP)
        for i, chunk_text in enumerate(splits):
            # Safeguard: truncate nếu vượt quá giới hạn token (~4000 chars << 8192 tokens)
            if len(chunk_text) > MAX_CHARS_PER_CHUNK:
                chunk_text = chunk_text[:MAX_CHARS_PER_CHUNK]
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": i},
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Embed chunks bằng OpenAI text-embedding-3-small."""
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    texts = [c["content"] for c in chunks]
    batch_size = 100

    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        all_embeddings.extend([item.embedding for item in response.data])
        print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)} chunks...")

    for chunk, emb in zip(chunks, all_embeddings):
        chunk["embedding"] = emb

    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """Lưu chunks và embeddings (normalized) vào data/index/."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    embeddings = np.array([c.pop("embedding") for c in chunks], dtype=np.float32)
    # Normalize để cosine similarity = dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_norm = embeddings / (norms + 1e-9)

    np.save(INDEX_DIR / "embeddings.npy", embeddings_norm)
    with open(INDEX_DIR / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(f"  [OK] Saved {len(chunks)} chunks -> {INDEX_DIR}")


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n[OK] Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"[OK] Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"[OK] Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("[OK] Indexed to vector store")
    return len(chunks)


if __name__ == "__main__":
    run_pipeline()
