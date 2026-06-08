"""
src/retrieval.py — Main Retrieval Module (Person 1 - 2A202600756_LeThanhDat)

Public API cho toàn bộ nhóm sử dụng:
    from src.retrieval import retrieve, generate_with_citation, build_index

Pipeline:
    data/standardized/*.json
        → load_documents()          [task4]
        → chunk_documents()         [task4]
        → embed_chunks()            [task4]
        → index_to_vectorstore()    [task4]
                ↓ data/index/
    Query
        ├→ semantic_search()        [task5]  (cosine similarity, OpenAI embeddings)
        ├→ lexical_search()         [task6]  (BM25Okapi, rank-bm25)
        │
        └→ rerank_rrf()             [task7]  (Reciprocal Rank Fusion)
                ↓
        retrieve()                  [task9]  (hybrid + fallback)
                ↓
        generate_with_citation()    [task10] (GPT-4o-mini + citation)
"""

# Re-export public interface
from .task4_chunking_indexing import run_pipeline as build_index
from .task5_semantic_search import semantic_search, reload_index
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank_rrf, rerank
from .task9_retrieval_pipeline import retrieve
from .task10_generation import generate_with_citation

__all__ = [
    "build_index",
    "semantic_search",
    "lexical_search",
    "rerank_rrf",
    "rerank",
    "retrieve",
    "generate_with_citation",
    "reload_index",
]
