from agents.orchestrator import orchestrator_node

# Simulate the initial state — just the company name, everything else empty
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

result = orchestrator_node(state)
print("\nTasks generated:")
for task in result["research_tasks"]:
    print(f"  - {task}")