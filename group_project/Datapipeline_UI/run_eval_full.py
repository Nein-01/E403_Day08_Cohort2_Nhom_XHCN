"""Run full evaluation and export results.md."""
import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, '.')

from pathlib import Path
from group_project.evaluation.eval_pipeline import (
    run_evaluation, compute_averages, export_results,
    rag_dense_only, rag_hybrid_rerank,
    GOLDEN_DATASET_PATH,
)

golden_dataset = json.loads(GOLDEN_DATASET_PATH.read_text(encoding='utf-8'))
print(f"Running full evaluation on {len(golden_dataset)} questions...")

print("\n--- Config A: Dense-only ---")
results_a = run_evaluation(golden_dataset, rag_dense_only, "Dense-only")

print("\n--- Config B: Hybrid + RRF ---")
results_b = run_evaluation(golden_dataset, rag_hybrid_rerank, "Hybrid+RRF")

avg_a = compute_averages(results_a)
avg_b = compute_averages(results_b)

print("\n=== FINAL RESULTS ===")
for m in ["faithfulness", "answer_relevance", "context_recall", "context_precision", "avg_score"]:
    print(f"  {m:<22} A={avg_a.get(m,0):.3f}  B={avg_b.get(m,0):.3f}")

export_results(results_a, results_b)
print("Done! results.md exported.")
