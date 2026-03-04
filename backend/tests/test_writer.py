from agents.collector import collector_node
from agents.writer import writer_node

# First collect the data
print("Step 1: Collecting data...")
state = {
    "company": "Apple",
    "research_tasks": [],
    "sec_data": None,
    "news_data": None,
    "draft_memo": None,
    "critique": None,
    "critique_passed": False,
    "revision_count": 0,
    "final_memo": None,
    "confidence_scores": {},
    "sources_used": [],
}

collected = collector_node(state)
state.update(collected)

# Then write the memo
print("\nStep 2: Writing memo...")
result = writer_node(state)

print("\n" + "="*60)
print("DRAFT MEMO (first 1000 chars):")
print("="*60)
print(result["draft_memo"][:1000])
print("\nConfidence scores:", result["confidence_scores"])
