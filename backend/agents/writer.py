import os
import json
from dotenv import load_dotenv
from anthropic import Anthropic
from agents.state import ResearchState

load_dotenv("../.env")
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def writer_node(state: ResearchState) -> dict:
    """
    Stage 3 of the pipeline — the analyst.

    Reads: company, sec_data, news_data, critique (if this is a revision)
    Writes: draft_memo, confidence_scores

    Calls Claude Sonnet to write the full investment memo.
    If critique exists, it addresses those issues in the new draft.
    """
    company = state["company"]
    sec_data = state.get("sec_data", "No SEC data available")
    news_data = state.get("news_data", "No news data available")
    critique = state.get("critique", "")
    revision_count = state.get("revision_count", 0)

    # If this is a revision, tell Claude what the critic found
    revision_instruction = ""
    if critique:
        revision_instruction = f"""
IMPORTANT: This is revision #{revision_count}. The critic found these issues with your previous draft:
{critique}

You MUST address ALL of these issues in this new draft.
"""

    prompt = f"""You are a senior financial analyst writing an investment memo.

COMPANY: {company}

SEC 10-K FILING DATA:
{sec_data}

RECENT NEWS:
{news_data}

{revision_instruction}

Write a complete investment memo with these sections:

# Investment Memo: {company}

## 1. Business Overview
(What the company does, key products, market position)

## 2. Financial Highlights
(Revenue, margins, growth — cite exact numbers with [SEC Filing] or [News: source])

## 3. Risk Factors
(Top 3-5 risks from the 10-K and recent news)

## 4. Competitive Landscape
(Key competitors and how this company stacks up)

## 5. Bull Case
(3 reasons to be optimistic)

## 6. Bear Case
(3 reasons to be cautious)

## 7. Analyst Consensus
(What the market currently thinks)

RULES:
- Cite every specific number with [SEC Filing] or [News: source name]
- If data is missing for a section, say so — do NOT invent numbers
- After the memo, add this JSON block with your confidence scores:

```json
{{
  "confidence_scores": {{
    "business_overview": "high",
    "financial_highlights": "medium",
    "risk_factors": "high",
    "competitive_landscape": "medium",
    "bull_case": "medium",
    "bear_case": "medium",
    "analyst_consensus": "low"
  }}
}}
```
Rate each section high / medium / low based on how much real source data you had.

Write the memo now:"""

    print(f"[Writer] Writing investment memo for {company} (revision {revision_count})...")

    message = client.messages.create(
        model="claude-sonnet-4-6",  # Sonnet for quality writing
        max_tokens=2500,
        messages=[{"role": "user", "content": prompt}]
    )

    draft = message.content[0].text

    # Extract confidence scores from the JSON block
    confidence_scores = {}
    try:
        if "```json" in draft:
            json_str = draft.split("```json")[1].split("```")[0].strip()
            parsed = json.loads(json_str)
            confidence_scores = parsed.get("confidence_scores", {})
    except Exception:
        pass

    print(f"[Writer] Draft complete ({len(draft)} chars)")
    return {
        "draft_memo": draft,
        "confidence_scores": confidence_scores,
        "revision_count": revision_count,
    }
