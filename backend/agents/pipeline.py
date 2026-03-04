from langgraph.graph import StateGraph, END
from agents.state import ResearchState
from agents.orchestrator import orchestrator_node
from agents.collector import collector_node
from agents.writer import writer_node
from agents.critic import critic_node, should_revise


def build_research_pipeline():
    """
    Wires all 4 agents into a LangGraph state machine.

    Flow:
    orchestrator → collector → writer → critic
                                          ↓
                               approved? → END
                               not approved? → writer (again, max 3x)
    """
    graph = StateGraph(ResearchState)

    # Register every agent as a node
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("collector", collector_node)
    graph.add_node("writer", writer_node)
    graph.add_node("critic", critic_node)

    # Fixed edges — these always happen in order
    graph.set_entry_point("orchestrator")
    graph.add_edge("orchestrator", "collector")
    graph.add_edge("collector", "writer")
    graph.add_edge("writer", "critic")

    # Conditional edge — after critic runs, check should_revise()
    # "revise" → go back to writer
    # "done"   → end the pipeline
    graph.add_conditional_edges(
        "critic",
        should_revise,
        {
            "revise": "writer",
            "done": END,
        }
    )

    return graph.compile()


def _initial_state(company: str) -> dict:
    return {
        "company": company,
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


def run_research(company: str) -> dict:
    """Runs the full pipeline and returns the final state."""
    pipeline = build_research_pipeline()
    final_state = pipeline.invoke(_initial_state(company))
    return final_state


def stream_research(company: str):
    """
    Streams pipeline progress as (node_name, state_snapshot) tuples.
    Use this to send live updates to the frontend via SSE.
    LangGraph's .stream() yields {node_name: state_updates} after each node runs.
    """
    pipeline = build_research_pipeline()
    for chunk in pipeline.stream(_initial_state(company)):
        # chunk is {node_name: {updated_fields}}
        node_name = list(chunk.keys())[0]
        state_update = chunk[node_name]
        yield node_name, state_update