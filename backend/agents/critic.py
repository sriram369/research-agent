import os
from dotenv import load_dotenv
from anthropic import Anthropic
from agents.state import ResearchState

load_dotenv("../.env")
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

MAX_REVISIONS = 3


def critic_node(state: ResearchState) -> dict:
    """
    Stage 4 of the pipeline — the fact-checker.

    Reads: draft_memo, sec_data, news_data, revision_count
    Writes: critique, critique_passed, final_memo, revision_count

    If critique_passed = True  → pipeline ends, final_memo is set
    If critique_passed = False → pipeline sends back to Writer with critique
    Max 3 revision loops to avoid infinite loops.
    """
    draft_memo = state.get("draft_memo", "")
    sec_data = state.get("sec_data", "")
    news_data = state.get("news_data", "")
    revision_count = state.get("revision_count", 0)

    # Safety valve — don't loop forever
    if revision_count >= MAX_REVISIONS:
        print(f"[Critic] Max revisions ({MAX_REVISIONS}) reached. Approving as-is.")
        return {
            "critique": None,
            "critique_passed": True,
            "final_memo": draft_memo,
            "revision_count": revision_count,
        }

    prompt = f"""You are a strict fact-checker reviewing an investment memo.

SOURCE DATA (what the memo must be based on):
=== SEC 10-K FILING ===
{sec_data[:3000]}

=== RECENT NEWS ===
{news_data[:3000]}

=== DRAFT MEMO TO REVIEW ===
{draft_memo}

Check for these problems:
1. HALLUCINATED NUMBERS — any specific number not found in the source data above
2. UNSUPPORTED CLAIMS — statements presented as fact with no citation
3. CONTRADICTIONS — bull case and bear case saying opposite things without acknowledging it
4. MISSING CRITICAL DATA — financials section with no actual numbers for a public company

Respond in exactly one of these two formats:

If the memo is acceptable (no hallucinations, well-cited):
APPROVED: [one sentence explaining why it passes]

If the memo has real problems:
REVISION NEEDED:
- Issue 1: [specific problem and how to fix it]
- Issue 2: [specific problem and how to fix it]

Be specific. Only flag factual or logical problems — not writing style."""

    print(f"[Critic] Reviewing draft memo (revision {revision_count})...")

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",  # Haiku is fast and cheap for review tasks
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}]
    )

    critique_response = message.content[0].text.strip()
    print(f"[Critic] Verdict: {critique_response[:80]}...")

    if critique_response.startswith("APPROVED"):
        return {
            "critique": None,
            "critique_passed": True,
            "final_memo": draft_memo,
            "revision_count": revision_count,
        }
    else:
        return {
            "critique": critique_response,
            "critique_passed": False,
            "final_memo": None,
            "revision_count": revision_count + 1,
        }


def should_revise(state: ResearchState) -> str:
    """
    LangGraph routing function — called after the Critic runs.
    Returns "revise" to send back to Writer, or "done" to end the pipeline.
    """
    if state.get("critique_passed", False):
        return "done"
    return "revise"