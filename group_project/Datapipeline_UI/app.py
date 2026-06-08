"""
RAG Chatbot Backend — FastAPI + Starlette.

Endpoints:
    GET  /          → Serve index.html
    POST /api/query → RAG generation với citation
    GET  /api/health → Health check + index status
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.task10_generation import generate_with_citation
from src.task4_chunking_indexing import INDEX_DIR

app = FastAPI(title="Minh Luật AI — RAG Chatbot", version="1.0.0")


class QueryRequest(BaseModel):
    question: str


@app.get("/")
async def homepage():
    return FileResponse("index.html")


@app.post("/api/query")
async def query_endpoint(req: QueryRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Câu hỏi không được trống")

    try:
        result = generate_with_citation(question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Format sources cho frontend — normalize scores về 0-100%
    raw_chunks = result["sources"][:5]
    raw_scores = [c.get("score", 0) for c in raw_chunks]
    max_score = max(raw_scores) if raw_scores else 1.0

    sources = []
    for i, chunk in enumerate(raw_chunks, 1):
        meta = chunk.get("metadata", {})
        crawled_at = (meta.get("crawled_at") or "")[:10]
        # Normalize: top kết quả hiển thị cao nhất
        normalized = (chunk.get("score", 0) / max_score) if max_score > 0 else 0
        score_pct = max(30, round(normalized * 97))  # min 30%, max 97%
        sources.append(
            {
                "id": i,
                "title": meta.get("title", f"Tài liệu {i}"),
                "source_url": meta.get("source_url", ""),
                "snippet": chunk["content"][:220].replace("\n", " ") + "...",
                "score": score_pct,
                "date": crawled_at,
                "type": meta.get("type", "news"),
            }
        )

    return {
        "answer": result["answer"],
        "sources": sources,
        "retrieval_source": result["retrieval_source"],
        "num_sources": len(sources),
    }


@app.get("/api/health")
async def health():
    index_ready = (INDEX_DIR / "embeddings.npy").exists()
    return {"status": "ok", "index_ready": index_ready}


@app.post("/api/build-index")
async def build_index():
    """Rebuild index từ data/standardized/. Chỉ gọi khi cần rebuild."""
    from src.task4_chunking_indexing import run_pipeline
    from src.task5_semantic_search import reload_index as sem_reload
    from src.task6_lexical_search import reload_index as lex_reload

    n = run_pipeline()
    sem_reload()
    lex_reload()
    return {"status": "ok", "chunks_indexed": n}


if __name__ == "__main__":
    import uvicorn

    # Build index nếu chưa tồn tại
    if not (INDEX_DIR / "embeddings.npy").exists():
        print("⚠ Index chưa tồn tại. Đang build...")
        from src.task4_chunking_indexing import run_pipeline
        run_pipeline()

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
