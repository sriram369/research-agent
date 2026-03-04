from agents.collector import collector_node

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

result = collector_node(state)
print("\nSources:", result["sources_used"])
print("\nSEC data preview:", result["sec_data"][:300])
print("\nNews preview:", result["news_data"][:300])