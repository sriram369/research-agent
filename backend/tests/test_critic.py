from agents.collector import collector_node
from agents.writer import writer_node
from agents.critic import critic_node

# Step 1: Collect
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
state.update(collector_node(state))

# Step 2: Write
print("\nStep 2: Writing memo...")
state.update(writer_node(state))

# Step 3: Critique
print("\nStep 3: Critiquing memo...")
result = critic_node(state)

print("\n" + "="*60)
print("CRITIC VERDICT:")
print("="*60)
print("Passed:", result["critique_passed"])
print("Critique:", result["critique"] or "None — memo approved!")
print("Revisions so far:", result["revision_count"])
