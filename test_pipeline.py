"""Quick test for the retrieval pipeline."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=== Test Semantic Search ===")
from src.task5_semantic_search import semantic_search
results = semantic_search("Rapper Binh Gold bi bat", top_k=3)
for r in results:
    print(f"  [{r['score']:.3f}] {r['content'][:80]}")

print("\n=== Test Lexical Search ===")
from src.task6_lexical_search import lexical_search
results = lexical_search("ca si bi bat lien quan ma tuy", top_k=3)
for r in results:
    print(f"  [{r['score']:.3f}] {r['content'][:80]}")

print("\n=== Test Full Retrieval Pipeline ===")
from src.task9_retrieval_pipeline import retrieve
results = retrieve("Chu Bin bi bat vi ly do gi", top_k=3)
for r in results:
    print(f"  [{r['score']:.4f}] [{r['source']}] {r['content'][:80]}")

print("\n=== Test Generation ===")
from src.task10_generation import generate_with_citation
result = generate_with_citation("Nhung nghe si nao bi bat vi lien quan ma tuy?")
print(f"Answer (first 300 chars): {result['answer'][:300]}")
print(f"Sources: {len(result['sources'])} chunks")
print("[DONE] All tests passed!")
