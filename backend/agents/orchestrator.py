import os
import ast
from dotenv import load_dotenv
from anthropic import Anthropic
from agents.state import ResearchState

load_dotenv("../.env")
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def orchestrator_node(state: ResearchState) -> dict:
    """
    Stage 1 of the pipeline — the project manager.

    Reads: company name
    Writes: research_tasks (a list of 4 sub-tasks to research)

    Does NOT fetch any data itself. Just decides WHAT needs to be researched.
    The Collector will do the actual fetching.
    """
    company = state["company"]

    prompt = f"""You are a research project manager. A user wants a full investment memo on: {company}

Break this into exactly 4 research tasks. Return ONLY a Python list of strings, nothing else.

Example format:
["Find the latest 10-K annual filing for {company}", "Find recent news about {company} from the last 90 days", "Find {company}'s top 3 competitors and their market cap", "Find analyst consensus on {company}'s revenue growth and outlook"]

Now generate the 4 tasks for {company}:"""

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",  # cheap + fast, no need for Sonnet here
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )

    response_text = message.content[0].text.strip()

    # Parse the Python list from Claude's response
    try:
        tasks = ast.literal_eval(response_text)
    except Exception:
        # Fallback if Claude returns something unexpected
        tasks = [
            f"Find the latest 10-K annual filing for {company}",
            f"Find recent news about {company} from the last 90 days",
            f"Find {company}'s top 3 competitors",
            f"Find analyst consensus on {company}'s growth outlook",
        ]

    print(f"[Orchestrator] Generated {len(tasks)} tasks for {company}:")
    for i, task in enumerate(tasks, 1):
        print(f"  {i}. {task}")

    return {"research_tasks": tasks}