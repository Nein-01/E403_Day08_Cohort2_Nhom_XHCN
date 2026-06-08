# Person 1 — Data & Retrieval Pipeline

**MSSV:** 2A202600756  
**Họ tên:** Lê Thanh Đạt  
**File chính:** `src/retrieval.py`, `src/task4_chunking_indexing.py` → `src/task9_retrieval_pipeline.py`, `app.py`

---

## 1. Tổng Quan Kiến Trúc

```
data/standardized/*.json
    │
    ├─ load_documents()          Đọc JSON, lấy title + markdown
    ├─ chunk_documents()         Tách text (RecursiveCharacter, size=600, overlap=80)
    ├─ embed_chunks()            OpenAI text-embedding-3-small (1536 dim)
    └─ index_to_vectorstore()    Lưu data/index/embeddings.npy + chunks.pkl
                    │
                    ▼
    Query
    ├─ semantic_search()         Cosine similarity trên numpy array
    ├─ lexical_search()          BM25Okapi (rank-bm25)
    │
    └─ rerank_rrf()              Reciprocal Rank Fusion
                    │
                    ▼
    retrieve()                   Hybrid pipeline + PageIndex fallback
                    │
                    ▼
    generate_with_citation()     GPT-4o-mini + reorder + citation
                    │
                    ▼
    app.py                       FastAPI backend → index.html UI
```

---

## 2. Files Đã Thực Hiện

| File | Mô tả | Trạng thái |
|------|-------|------------|
| `src/task4_chunking_indexing.py` | Load data, chunking, embedding, index | ✅ Hoàn thành |
| `src/task5_semantic_search.py` | Semantic search (cosine similarity) | ✅ Hoàn thành |
| `src/task6_lexical_search.py` | BM25 lexical search | ✅ Hoàn thành |
| `src/task7_reranking.py` | RRF, MMR, Cross-encoder (stub) | ✅ Hoàn thành |
| `src/task8_pageindex_vectorless.py` | PageIndex fallback (graceful) | ✅ Hoàn thành |
| `src/task9_retrieval_pipeline.py` | Full hybrid pipeline + fallback | ✅ Hoàn thành |
| `src/task10_generation.py` | LLM generation với citation | ✅ Hoàn thành |
| `src/retrieval.py` | Public API wrapper | ✅ Hoàn thành |
| `app.py` | FastAPI backend + serve UI | ✅ Hoàn thành |
| `index.html` | Chat UI (cập nhật gọi real API) | ✅ Hoàn thành |
| `data/index/` | Vector index (tạo sau khi chạy pipeline) | ✅ Đã build |
| `group_project/evaluation/golden_dataset.json` | 18 cặp Q&A | ✅ Hoàn thành |
| `group_project/evaluation/eval_pipeline.py` | LLM-as-Judge evaluation + A/B | ✅ Hoàn thành |
| `group_project/evaluation/results.md` | Báo cáo kết quả eval | ✅ Hoàn thành |

---

## 3. Hướng Dẫn Cài Đặt & Chạy

### 3.1 Yêu cầu môi trường

```bash
# Python 3.11+
# Đã có trong .venv:
#   openai==2.41.0    (LLM + embeddings)
#   rank-bm25==0.2.2  (lexical search)
#   numpy==2.4.6      (vector operations)
#   fastapi==0.136.3  (web backend)
#   uvicorn==0.49.0   (ASGI server)

# Cài thêm:
.venv\Scripts\pip install fastapi
```

### 3.2 Cấu hình .env

```bash
# File .env phải có:
OPENAI_API_KEY=sk-proj-...       # Bắt buộc
PAGEINDEX_API_KEY=               # Tuỳ chọn (fallback retrieval)
```

### 3.3 Bước 1 — Build Index (chỉ cần chạy 1 lần)

```bash
# Tạo embeddings + BM25 index từ 7 bài báo trong data/standardized/
.venv\Scripts\python.exe -m src.task4_chunking_indexing

# Output:
# [OK] Loaded 7 documents
# [OK] Created 23 chunks
# Embedded 23/23 chunks...
# [OK] Saved 23 chunks -> data/index/
```

Index được lưu tại:
- `data/index/embeddings.npy` — numpy array (23, 1536) đã normalize
- `data/index/chunks.pkl` — list dicts chứa content + metadata

### 3.4 Bước 2 — Chạy App Chatbot

```bash
.venv\Scripts\python.exe app.py
# hoặc:
.venv\Scripts\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8000

# Mở trình duyệt: http://localhost:8000
```

### 3.5 Bước 3 — Test Pipeline

```bash
# Test toàn bộ pipeline (semantic → lexical → generation)
.venv\Scripts\python.exe test_pipeline.py

# Test API endpoint
.venv\Scripts\python.exe test_api.py
```

### 3.6 Bước 4 — Chạy Evaluation

```bash
# Chạy evaluation subset (6 câu, ~2 phút, tiết kiệm API)
.venv\Scripts\python.exe -m group_project.evaluation.eval_pipeline

# Chạy full evaluation (18 câu, ~8 phút) + export results.md
.venv\Scripts\python.exe run_eval_full.py
```

---

## 4. Chi Tiết Kỹ Thuật

### 4.1 Data Loading (`task4_chunking_indexing.py`)

**Nguồn dữ liệu:** 7 file JSON trong `data/standardized/`, mỗi file là bài báo được crawl bởi Crawl4AI với cấu trúc:
```json
{
  "source_url": "https://...",
  "crawled_at": "2025-...",
  "title": "Tiêu đề bài báo",
  "success": true,
  "markdown": "Nội dung bài báo dạng markdown (20-25K chars)",
  "metadata": {...}
}
```

**Preprocessing:** Chỉ lấy field `title` và `markdown`, bỏ qua HTML/navigation content.

### 4.2 Chunking Strategy

**Phương pháp:** `RecursiveCharacterTextSplitter` (tự implement không dùng langchain)  
**Tham số:**
- `chunk_size = 600` chars — đủ để capture 1 đoạn tin tức có nghĩa
- `chunk_overlap = 80` chars — giữ ngữ cảnh ở ranh giới
- `max_chars = 4000` chars — safeguard tránh vượt token limit OpenAI
- Separators: `["\n\n", "\n", ". ", " ", ""]`

**Kết quả:** 7 documents → 23 chunks (avg ~550 chars/chunk)

### 4.3 Embedding Model

**Model:** `text-embedding-3-small` (OpenAI)  
**Dimension:** 1536  
**Normalization:** L2-normalize để cosine similarity = dot product  
**Chi phí:** ~$0.001 cho 23 chunks (~11K tokens)

### 4.4 Vector Store

**Implementation:** Numpy in-memory  
**Tìm kiếm:** Dot product (= cosine similarity vì đã normalize)  
**Complexity:** O(n) — đủ nhanh với corpus nhỏ (23 chunks)

### 4.5 BM25 Lexical Search (`task6_lexical_search.py`)

**Library:** `rank-bm25==0.2.2`  
**Tokenization:** Simple whitespace split (word-level)  
**Params:** k1=1.5, b=0.75 (BM25Okapi defaults)  
**Use case:** Bắt exact keyword match (tên người, năm, điều luật)

### 4.6 Reranking — RRF (`task7_reranking.py`)

**Phương pháp:** Reciprocal Rank Fusion (Cormack et al. 2009)  
**Formula:** `RRF(d) = Σ 1/(k + rank_r(d))` với k=60  
**Ưu điểm:** Không cần model, kết hợp tốt dense + sparse results  
**Cross-encoder:** Stub sẵn sàng — cần `JINA_API_KEY` để dùng thực

### 4.7 Retrieval Pipeline (`task9_retrieval_pipeline.py`)

```
semantic_search(query, top_k=10)  →  dense_results
lexical_search(query, top_k=10)   →  sparse_results
        │
        ▼
rerank_rrf([dense, sparse], top_k=5)  →  merged
        │
        ▼  kiểm tra best_semantic_score < 0.25?
        │      → pageindex_search() fallback (nếu có API key)
        │      → dùng hybrid results nếu không có PageIndex
        ▼
top_k results  (mỗi item có: content, score, metadata, source)
```

### 4.8 Generation (`task10_generation.py`)

**Model:** `gpt-4o-mini`  
**Temperature:** 0.3 (factual output)  
**Top_p:** 0.9  
**Document Reordering** (Liu et al. 2023 "Lost in the Middle"):
- Input: [1, 2, 3, 4, 5] (sorted by score desc)
- Output: [1, 3, 5, 4, 2] — quan trọng nhất ở đầu và cuối

**Citation format:** `[Tên nguồn, Năm]`

### 4.9 Backend API (`app.py`)

**Framework:** FastAPI  
**Endpoints:**
| Method | Path | Mô tả |
|--------|------|-------|
| GET | `/` | Serve index.html |
| POST | `/api/query` | RAG query → {answer, sources} |
| GET | `/api/health` | Status check |
| POST | `/api/build-index` | Rebuild index |

**Request/Response:**
```json
// POST /api/query
{ "question": "Rapper Bình Gold bị bắt vì gì?" }

// Response
{
  "answer": "Rapper Bình Gold bị bắt tạm giam vì...[Bắt tạm giam rapper, 2025]",
  "sources": [
    {
      "id": 1, "title": "Bắt tạm giam rapper Bình Gold",
      "source_url": "https://...", "snippet": "...",
      "score": 97, "date": "2025-07-27", "type": "news"
    }
  ],
  "retrieval_source": "hybrid",
  "num_sources": 5
}
```

---

## 5. Evaluation Results

**Framework:** LLM-as-Judge (GPT-4o-mini)  
**Dataset:** 18 cặp Q&A về các nghệ sĩ liên quan ma túy  

| Metric | Config A (Dense-only) | Config B (Hybrid+RRF) |
|--------|----------------------|----------------------|
| Faithfulness | 0.583 | 0.311 |
| Answer Relevance | 0.889 | 0.889 |
| Context Recall | 0.639 | 0.556 |
| Context Precision | 0.683 | 0.656 |
| **Average** | **0.699** | **0.603** |

**Phân tích:**
- Answer Relevance cao (0.889) — pipeline trả lời đúng câu hỏi
- Dense-only có faithfulness cao hơn vì retrieves đúng bài báo hơn (corpus đơn giản)
- Hybrid+RRF bị ảnh hưởng bởi navigation/menu chunks trong BM25 index
- **Cải tiến cần làm:** Filter navigation content trước khi index

---

## 6. Những Vấn Đề Gặp Phải & Cách Giải Quyết

| Vấn đề | Nguyên nhân | Giải pháp |
|--------|-------------|-----------|
| `UnicodeEncodeError` khi print | Windows console CP1252 | Dùng `sys.stdout = io.TextIOWrapper(..., encoding='utf-8')` |
| Chunk vượt 8192 token limit | Markdown có table/URL dài không có separator | Thêm `MAX_CHARS_PER_CHUNK = 4000` |
| RRF score hiển thị 3% | RRF scores (~0.03) ≠ cosine scores (0-1) | Normalize scores tương đối vs max |
| Navigation menu trong chunks | Crawled HTML chứa menu items | TODO: filter với regex trước khi chunk |

---

## 7. Hướng Dẫn Thêm Dữ Liệu Mới

```python
# 1. Thêm JSON file vào data/standardized/
# Format phải có: success=True, title, markdown, source_url, crawled_at

# 2. Rebuild index
.venv\Scripts\python.exe -m src.task4_chunking_indexing

# 3. Reset cache (nếu server đang chạy)
import httpx
httpx.post("http://localhost:8000/api/build-index")
```

---

## 8. Public API cho Thành Viên Khác

```python
from src.retrieval import (
    build_index,           # Rebuild index từ đầu
    semantic_search,       # Dense retrieval only
    lexical_search,        # BM25 only
    retrieve,              # Full hybrid pipeline (dùng cái này)
    generate_with_citation,# End-to-end RAG (dùng cái này)
)

# Ví dụ sử dụng:
results = retrieve("Rapper Bình Gold bị bắt vì gì?", top_k=5)
# → [{'content': '...', 'score': 0.032, 'metadata': {...}, 'source': 'hybrid'}, ...]

result = generate_with_citation("Rapper Bình Gold bị bắt vì gì?")
# → {'answer': 'Rapper Bình Gold bị bắt...', 'sources': [...], 'retrieval_source': 'hybrid'}
```

---

## 9. Cấu Trúc File Đã Tạo

```
src/
├── retrieval.py              ← Main public API
├── task4_chunking_indexing.py ← Load, chunk, embed, index
├── task5_semantic_search.py  ← Dense retrieval
├── task6_lexical_search.py   ← BM25 lexical search
├── task7_reranking.py        ← RRF + MMR + Cross-encoder stub
├── task8_pageindex_vectorless.py ← PageIndex fallback
├── task9_retrieval_pipeline.py  ← Full hybrid pipeline
└── task10_generation.py      ← LLM generation + citation

data/index/                   ← Vector index (auto-created)
├── embeddings.npy            ← (23, 1536) float32
└── chunks.pkl                ← list[dict]

app.py                        ← FastAPI backend
index.html                    ← Chat UI (updated)
test_pipeline.py              ← Quick pipeline test
test_api.py                   ← API endpoint test
run_eval_full.py              ← Full evaluation runner

group_project/evaluation/
├── golden_dataset.json       ← 18 Q&A pairs
├── eval_pipeline.py          ← LLM-as-Judge + A/B comparison
└── results.md                ← Báo cáo kết quả
```

---

## 10. Cách Chạy Tests

### Test pipeline nhanh
```bash
.venv\Scripts\python.exe test_pipeline.py
# Kiểm tra: semantic search, lexical search, retrieval, generation
```

### Test API
```bash
# Terminal 1: start server
.venv\Scripts\python.exe app.py

# Terminal 2: test
.venv\Scripts\python.exe test_api.py
```

### Test từng module riêng lẻ
```bash
# Task 4: rebuild index
.venv\Scripts\python.exe -m src.task4_chunking_indexing

# Task 5: semantic search
.venv\Scripts\python.exe -m src.task5_semantic_search

# Task 6: lexical search
.venv\Scripts\python.exe -m src.task6_lexical_search

# Task 7: reranking (RRF demo)
.venv\Scripts\python.exe -m src.task7_reranking

# Task 9: full retrieval
.venv\Scripts\python.exe -m src.task9_retrieval_pipeline

# Task 10: generation
.venv\Scripts\python.exe -m src.task10_generation
```

### Chạy evaluation
```bash
# Subset 6 câu (nhanh, ~2 phút)
.venv\Scripts\python.exe -m group_project.evaluation.eval_pipeline

# Full 18 câu + export results.md (~8 phút)
.venv\Scripts\python.exe run_eval_full.py
```

---

## 11. Lưu Ý Quan Trọng

1. **Index phải được build trước** khi chạy search/generation. Nếu quên, app sẽ tự build khi start.
2. **OpenAI API key** là bắt buộc — cần cho cả embeddings và generation.
3. **Data index được cache** trong memory sau lần load đầu tiên (lazy loading). Nếu rebuild index, cần restart server hoặc gọi `/api/build-index`.
4. **Corpus hiện tại** chỉ có 7 bài báo tin tức — thiếu văn bản pháp luật. Cần thêm data để trả lời câu hỏi về điều luật.
5. **PageIndex fallback** sẽ không hoạt động nếu không có `PAGEINDEX_API_KEY` — pipeline vẫn hoạt động bình thường với hybrid results.
