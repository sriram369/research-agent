from agents.pipeline import run_research

result = run_research("Apple")

print("\n" + "="*60)
print("FINAL MEMO (first 1500 chars):")
print("="*60)
print(result["final_memo"][:1500])
print("\nRevisions made:", result["revision_count"])
print("Sources:", result["sources_used"])
print("Confidence scores:", result["confidence_scores"])