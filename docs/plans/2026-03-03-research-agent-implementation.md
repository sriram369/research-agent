# Research Agent Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an autonomous AI research agent that takes a company name and produces a full investment memo with self-critique and confidence scores.

**Architecture:** 4-stage LangGraph pipeline — Orchestrator breaks the query into tasks, Data Collector fetches from SEC EDGAR + Tavily, Writer produces the memo, Critic fact-checks and sends back for revision (up to 3 loops). FastAPI streams results to a Next.js frontend.

**Tech Stack:** Python 3.11, LangGraph, LangChain, FastAPI, Anthropic Claude API, Tavily API, SEC EDGAR (free), Next.js

---

## PHASE 1: Environment Setup

### Task 1: Install Python 3.11 via pyenv

**What this does:** pyenv lets you have multiple Python versions. We need 3.11 because LangGraph doesn't fully support 3.14 yet.

**Step 1: Install pyenv**

Open your terminal and run:
```bash
brew install pyenv
```

**Step 2: Add pyenv to your shell**

Run these 3 commands one by one:
```bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
```

**Step 3: Restart your terminal (close and reopen it)**

**Step 4: Install Python 3.11**
```bash
pyenv install 3.11.9
```
Expected: Downloads and installs Python 3.11.9 (takes 1-2 minutes)

**Step 5: Verify**
```bash
pyenv versions
```
Expected output includes: `3.11.9`

---

### Task 2: Set Python 3.11 for this project + create virtual environment

**What this does:** A virtual environment is an isolated box for your project's dependencies. It means packages you install here don't mess with other projects.

**Step 1: Go into your project folder**
```bash
cd /Users/sriram/Downloads/Final_Projects/research-agent
```

**Step 2: Tell pyenv to use 3.11 in this folder**
```bash
pyenv local 3.11.9
```
This creates a `.python-version` file. Every time you're in this folder, Python 3.11 is used automatically.

**Step 3: Verify the right Python is active**
```bash
python --version
```
Expected: `Python 3.11.9`

**Step 4: Create a virtual environment**
```bash
python -m venv venv
```
This creates a `venv/` folder. This is your isolated box.

**Step 5: Activate the virtual environment**
```bash
source venv/bin/activate
```
Expected: Your terminal prompt now starts with `(venv)` — this means you're inside the box.

> **Important:** Every time you open a new terminal to work on this project, run `source venv/bin/activate` first.

---

### Task 3: Create backend folder structure

**Step 1: Create all the folders**
```bash
mkdir -p backend/agents backend/tools backend/tests
```

**Step 2: Create empty placeholder files so Python knows these are modules**
```bash
touch backend/__init__.py
touch backend/agents/__init__.py
touch backend/tools/__init__.py
touch backend/tests/__init__.py
```

**Step 3: Verify structure**
```bash
ls backend/
```
Expected: `__init__.py  agents/  tools/  tests/`

---

### Task 4: Install backend dependencies

**Step 1: Create the requirements file**

Create `backend/requirements.txt` with this content:
```
anthropic==0.40.0
langgraph==0.2.50
langchain==0.3.0
langchain-anthropic==0.3.0
langchain-community==0.3.0
tavily-python==0.5.0
fastapi==0.115.0
uvicorn==0.32.0
httpx==0.27.0
python-dotenv==1.0.0
pydantic==2.9.0
requests==2.32.0
beautifulsoup4==4.12.0
```

**Step 2: Install them**
```bash
pip install -r backend/requirements.txt
```
Expected: Lots of output, ends with `Successfully installed ...` (takes 2-3 minutes)

**Step 3: Verify LangGraph installed**
```bash
python -c "import langgraph; print(langgraph.__version__)"
```
Expected: prints a version number like `0.2.50`

---

### Task 5: Set up API keys

**What this does:** Your `.env` file stores secret API keys. Never commit this file to git.

**Step 1: Create the .env file** in the root `research-agent/` folder:
```
ANTHROPIC_API_KEY=your_claude_key_here
TAVILY_API_KEY=your_tavily_key_here
```

Replace `your_claude_key_here` with your actual Claude API key.

**Step 2: Get a Tavily API key**
- Go to https://tavily.com
- Sign up (free)
- Copy your API key from the dashboard
- Paste it into `.env`

**Step 3: Create .gitignore** to make sure keys are never committed:

Create `research-agent/.gitignore`:
```
venv/
.env
__pycache__/
*.pyc
.python-version
.DS_Store
node_modules/
.next/
```

**Step 4: Verify .env is being read correctly**

Create a quick test file `backend/tests/test_env.py`:
```python
from dotenv import load_dotenv
import os

load_dotenv("../.env")

def test_env_keys_exist():
    assert os.getenv("ANTHROPIC_API_KEY") is not None, "ANTHROPIC_API_KEY missing"
    assert os.getenv("TAVILY_API_KEY") is not None, "TAVILY_API_KEY missing"
    print("Both API keys found!")
```

Run it:
```bash
cd backend
python -m pytest tests/test_env.py -v -s
```
Expected: `PASSED` with "Both API keys found!" printed

---

## PHASE 2: Data Collection Tools

### Task 6: Build the SEC EDGAR tool

**What this does:** SEC EDGAR is a free US government database of company filings. We'll pull 10-K (annual reports) which have financial data, risk factors, and business overviews.

**Step 1: Write a failing test first**

Create `backend/tests/test_sec_edgar.py`:
```python
from tools.sec_edgar import get_company_filings, get_filing_text

def test_get_company_cik():
    """SEC EDGAR identifies companies by CIK number. We need to look it up by name."""
    result = get_company_filings("Apple")
    assert result is not None
    assert "cik" in result
    assert "company_name" in result

def test_get_filing_sections():
    """We should be able to pull specific sections from a 10-K filing."""
    result = get_company_filings("Apple")
    text = get_filing_text(result["cik"])
    assert text is not None
    assert len(text) > 100  # should have actual content
```

**Step 2: Run test to verify it fails**
```bash
python -m pytest tests/test_sec_edgar.py -v
```
Expected: `ERROR` — `ModuleNotFoundError: No module named 'tools.sec_edgar'` — good, that's expected.

**Step 3: Create `backend/tools/sec_edgar.py`**:
```python
import requests
import re
from typing import Optional

EDGAR_BASE = "https://data.sec.gov"
HEADERS = {"User-Agent": "research-agent contact@example.com"}


def get_company_filings(company_name: str) -> Optional[dict]:
    """Search EDGAR for a company and return its CIK number and latest filings."""
    # Search for the company
    search_url = f"https://efts.sec.gov/LATEST/search-index?q=%22{company_name}%22&dateRange=custom&startdt=2023-01-01&forms=10-K"
    response = requests.get(
        f"https://efts.sec.gov/LATEST/search-index?q={company_name}&forms=10-K",
        headers=HEADERS
    )

    if response.status_code != 200:
        return None

    data = response.json()
    hits = data.get("hits", {}).get("hits", [])

    if not hits:
        return None

    # Get the first result
    first = hits[0]["_source"]
    return {
        "cik": first.get("entity_id", "").lstrip("0"),
        "company_name": first.get("display_names", [company_name])[0],
        "form_type": first.get("form_type"),
        "filed_at": first.get("period_of_report"),
    }


def get_filing_text(cik: str, max_chars: int = 8000) -> Optional[str]:
    """
    Pull the most recent 10-K filing for a company and return key sections.
    max_chars limits how much text we pass to the LLM (saves tokens).
    """
    # Get filing list for this company
    url = f"{EDGAR_BASE}/submissions/CIK{cik.zfill(10)}.json"
    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        return None

    filings = response.json()
    company_name = filings.get("name", "Unknown")

    # Find the most recent 10-K
    recent = filings.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accession_numbers = recent.get("accessionNumber", [])

    ten_k_index = None
    for i, form in enumerate(forms):
        if form == "10-K":
            ten_k_index = i
            break

    if ten_k_index is None:
        return f"No 10-K filing found for {company_name}"

    accession = accession_numbers[ten_k_index].replace("-", "")

    # Get the filing index to find the actual document
    index_url = f"{EDGAR_BASE}/Archives/edgar/data/{cik}/{accession}/{accession_numbers[ten_k_index]}-index.json"
    index_response = requests.get(index_url, headers=HEADERS)

    if index_response.status_code != 200:
        return f"Found 10-K for {company_name} but couldn't retrieve it"

    index_data = index_response.json()

    # Find the main document (usually the largest .htm file)
    documents = index_data.get("directory", {}).get("item", [])
    main_doc = None
    for doc in documents:
        if doc.get("type") == "10-K" and doc.get("name", "").endswith(".htm"):
            main_doc = doc.get("name")
            break

    if not main_doc:
        # Try to find any .htm file
        for doc in documents:
            if doc.get("name", "").endswith(".htm"):
                main_doc = doc.get("name")
                break

    if not main_doc:
        return f"Found 10-K for {company_name} but couldn't find document file"

    # Fetch the actual document
    doc_url = f"{EDGAR_BASE}/Archives/edgar/data/{cik}/{accession}/{main_doc}"
    doc_response = requests.get(doc_url, headers=HEADERS)

    if doc_response.status_code != 200:
        return f"Couldn't download 10-K for {company_name}"

    # Strip HTML tags to get plain text
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(doc_response.text, "html.parser")
    text = soup.get_text(separator=" ", strip=True)

    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)

    # Return trimmed version to save tokens
    return f"[10-K Filing for {company_name}]\n\n{text[:max_chars]}..."
```

**Step 4: Run the test**
```bash
python -m pytest tests/test_sec_edgar.py -v -s
```
Expected: `PASSED` — if it fails, check your internet connection and try again.

---

### Task 7: Build the Tavily web search tool

**Step 1: Write the failing test**

Create `backend/tests/test_web_search.py`:
```python
from tools.web_search import search_company_news

def test_search_returns_results():
    results = search_company_news("Apple Inc latest news")
    assert results is not None
    assert len(results) > 0
    assert "title" in results[0]
    assert "content" in results[0]

def test_search_returns_string_summary():
    summary = search_company_news("Tesla earnings 2024", as_string=True)
    assert isinstance(summary, str)
    assert len(summary) > 50
```

**Step 2: Run to verify it fails**
```bash
python -m pytest tests/test_web_search.py -v
```
Expected: `ERROR` — `ModuleNotFoundError`

**Step 3: Create `backend/tools/web_search.py`**:
```python
import os
from typing import Union
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv("../.env")


def search_company_news(query: str, max_results: int = 5, as_string: bool = False) -> Union[list, str]:
    """
    Search for recent news and information about a company using Tavily.

    Args:
        query: What to search for, e.g. "Apple Inc latest earnings news"
        max_results: How many results to return
        as_string: If True, returns a formatted string instead of a list

    Returns:
        List of {title, url, content} dicts, or a formatted string
    """
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    response = client.search(
        query=query,
        search_depth="advanced",
        max_results=max_results,
        include_raw_content=False,
    )

    results = []
    for r in response.get("results", []):
        results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
            "score": r.get("score", 0),
        })

    if as_string:
        # Format for passing to an LLM
        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(f"[Source {i}] {r['title']}\nURL: {r['url']}\n{r['content']}")
        return "\n\n".join(formatted)

    return results
```

**Step 4: Run the test**
```bash
python -m pytest tests/test_web_search.py -v -s
```
Expected: `PASSED`

---

## PHASE 3: LangGraph Agents

### Task 8: Understand LangGraph State (read this before coding)

LangGraph works like a flowchart where each node is a function. The "state" is a dictionary that gets passed from node to node — each node reads from it and adds to it.

Our state will look like this:
```python
{
    "company": "Rivian",
    "tasks": ["find 10-K", "find news", "find competitors"],
    "sec_data": "...text from SEC filing...",
    "news_data": "...text from Tavily search...",
    "draft_memo": "...first draft of report...",
    "critique": "...critic's feedback...",
    "final_memo": "...approved final report...",
    "revision_count": 0,
    "confidence_scores": {"overview": "high", "financials": "medium"},
}
```

---

### Task 9: Create the LangGraph state definition

**Step 1: Create `backend/agents/state.py`**:
```python
from typing import TypedDict, Optional, List


class ResearchState(TypedDict):
    """The state that flows through our LangGraph pipeline."""
    # Input
    company: str

    # Stage 1: Orchestrator output
    research_tasks: List[str]

    # Stage 2: Data collection output
    sec_data: Optional[str]
    news_data: Optional[str]

    # Stage 3: Writer output
    draft_memo: Optional[str]

    # Stage 4: Critic output
    critique: Optional[str]
    critique_passed: bool
    revision_count: int

    # Final output
    final_memo: Optional[str]
    confidence_scores: dict
    sources_used: List[str]
```

---

### Task 10: Build the Orchestrator agent

**What this does:** Takes a company name and decides what research tasks to run. It's the "project manager" of our pipeline.

**Step 1: Create `backend/agents/orchestrator.py`**:
```python
import os
from dotenv import load_dotenv
from anthropic import Anthropic
from agents.state import ResearchState

load_dotenv("../.env")
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def orchestrator_node(state: ResearchState) -> dict:
    """
    Takes a company name and produces a list of research sub-tasks.
    This is Stage 1 of our pipeline.
    """
    company = state["company"]

    prompt = f"""You are a research orchestrator. A user wants a full investment memo on: {company}

Break this into exactly 4 research tasks. Return ONLY a Python list of strings, nothing else.

Example format:
["Find the latest 10-K annual filing for {company}", "Find recent news about {company} from the last 90 days", "Find {company}'s top 3 competitors and their market cap", "Find analyst consensus on {company}'s revenue growth and outlook"]

Now generate the 4 tasks for {company}:"""

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )

    response_text = message.content[0].text.strip()

    # Parse the list from the response
    import ast
    try:
        tasks = ast.literal_eval(response_text)
    except Exception:
        # Fallback if parsing fails
        tasks = [
            f"Find the latest 10-K annual filing for {company}",
            f"Find recent news about {company} from the last 90 days",
            f"Find {company}'s top 3 competitors",
            f"Find analyst consensus on {company}'s growth outlook",
        ]

    print(f"[Orchestrator] Generated {len(tasks)} research tasks for {company}")
    return {"research_tasks": tasks}
```

**Step 2: Write a quick test**

Add to `backend/tests/test_agents.py`:
```python
import sys
sys.path.insert(0, "..")

from agents.orchestrator import orchestrator_node

def test_orchestrator_generates_tasks():
    state = {"company": "Apple", "research_tasks": [], "revision_count": 0}
    result = orchestrator_node(state)
    assert "research_tasks" in result
    assert len(result["research_tasks"]) > 0
    print(f"Tasks: {result['research_tasks']}")
```

Run:
```bash
python -m pytest tests/test_agents.py::test_orchestrator_generates_tasks -v -s
```
Expected: `PASSED` with tasks printed

---

### Task 11: Build the Data Collector node

**Step 1: Create `backend/agents/collector.py`**:
```python
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tools.sec_edgar import get_company_filings, get_filing_text
from tools.web_search import search_company_news
from agents.state import ResearchState


def collector_node(state: ResearchState) -> dict:
    """
    Fetches real data from SEC EDGAR and Tavily.
    This is Stage 2 of our pipeline.
    """
    company = state["company"]
    sources = []

    print(f"[Collector] Fetching SEC EDGAR data for {company}...")
    filing_info = get_company_filings(company)

    if filing_info:
        sec_text = get_filing_text(filing_info["cik"])
        sources.append(f"SEC EDGAR 10-K: {filing_info.get('company_name', company)}")
    else:
        sec_text = f"No SEC EDGAR filing found for {company}. Using general knowledge only."

    print(f"[Collector] Searching for recent news about {company}...")
    news_text = search_company_news(
        f"{company} stock earnings revenue news 2024 2025",
        as_string=True
    )
    sources.append(f"Tavily Web Search: recent {company} news")

    print(f"[Collector] Data collection complete.")
    return {
        "sec_data": sec_text,
        "news_data": news_text,
        "sources_used": sources,
    }
```

---

### Task 12: Build the Writer agent

**What this does:** Takes all the collected data and writes the investment memo. Cites every claim.

**Step 1: Create `backend/agents/writer.py`**:
```python
import os
import json
from dotenv import load_dotenv
from anthropic import Anthropic
from agents.state import ResearchState

load_dotenv("../.env")
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

MEMO_TEMPLATE = """
# Investment Memo: {company}

## 1. Business Overview
[2-3 paragraphs: what the company does, key products, market position]

## 2. Financial Highlights
[Revenue, margins, growth rates — cite exact numbers from SEC filing]

## 3. Risk Factors
[Top 3-5 risks from the 10-K + any recent developments]

## 4. Competitive Landscape
[Key competitors, their market cap, how this company stacks up]

## 5. Bull Case
[3 reasons a long-term investor would be optimistic]

## 6. Bear Case
[3 reasons to be cautious]

## 7. Analyst Consensus
[What the market currently thinks based on recent news]

## Confidence Scores
[Rate each section: HIGH / MEDIUM / LOW based on how much source data you had]
"""


def writer_node(state: ResearchState) -> dict:
    """
    Writes the investment memo using collected data.
    This is Stage 3 of our pipeline.
    """
    company = state["company"]
    sec_data = state.get("sec_data", "No SEC data available")
    news_data = state.get("news_data", "No news data available")
    critique = state.get("critique", "")
    revision_count = state.get("revision_count", 0)

    revision_instruction = ""
    if critique:
        revision_instruction = f"""
IMPORTANT: This is revision #{revision_count}. The critic found these issues with your previous draft:
{critique}

Address ALL of these issues in your new draft.
"""

    prompt = f"""You are a senior financial analyst writing an investment memo.

COMPANY TO ANALYZE: {company}

SEC FILING DATA (10-K):
{sec_data}

RECENT NEWS:
{news_data}

{revision_instruction}

Write a complete investment memo following this template:
{MEMO_TEMPLATE}

RULES:
1. Every factual number MUST be cited with [SEC Filing] or [News: source name]
2. If you don't have data for a section, say "Data not available" — do NOT invent numbers
3. Keep each section concise but substantive
4. After the memo, add a JSON block like this:
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

Write the memo now:"""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )

    draft = message.content[0].text

    # Try to extract confidence scores from the JSON block
    confidence_scores = {}
    try:
        if "```json" in draft:
            json_str = draft.split("```json")[1].split("```")[0].strip()
            parsed = json.loads(json_str)
            confidence_scores = parsed.get("confidence_scores", {})
    except Exception:
        pass

    print(f"[Writer] Draft memo written ({len(draft)} chars)")
    return {
        "draft_memo": draft,
        "confidence_scores": confidence_scores,
        "revision_count": revision_count,
    }
```

---

### Task 13: Build the Critic agent

**What this does:** This is the differentiator. Reads the memo, checks every claim, returns specific feedback or approves it.

**Step 1: Create `backend/agents/critic.py`**:
```python
import os
from dotenv import load_dotenv
from anthropic import Anthropic
from agents.state import ResearchState

load_dotenv("../.env")
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

MAX_REVISIONS = 3


def critic_node(state: ResearchState) -> dict:
    """
    Fact-checks the draft memo against source data.
    Returns critique or approves the memo.
    This is Stage 4 of our pipeline.
    """
    draft_memo = state.get("draft_memo", "")
    sec_data = state.get("sec_data", "")
    news_data = state.get("news_data", "")
    revision_count = state.get("revision_count", 0)

    # Don't loop forever
    if revision_count >= MAX_REVISIONS:
        print(f"[Critic] Max revisions ({MAX_REVISIONS}) reached. Approving as-is.")
        return {
            "critique": None,
            "critique_passed": True,
            "final_memo": draft_memo,
            "revision_count": revision_count,
        }

    prompt = f"""You are a strict fact-checker reviewing an investment memo.

ORIGINAL SOURCE DATA:
=== SEC FILING ===
{sec_data[:3000]}

=== RECENT NEWS ===
{news_data[:3000]}

=== DRAFT MEMO TO REVIEW ===
{draft_memo}

Your job: Find problems. Be harsh. Check for:
1. HALLUCINATED NUMBERS: Any specific number (revenue, %, dates) not found in the source data
2. UNSUPPORTED CLAIMS: Statements presented as fact without a citation
3. CONTRADICTIONS: Places where the bull case and bear case say opposite things without acknowledging it
4. MISSING DATA: Critical sections that say nothing substantive (especially financials for a public company)

Respond in one of two ways:

If the memo is ACCEPTABLE (minor issues only, no hallucinations):
APPROVED: [one sentence saying why it's good enough]

If the memo has REAL PROBLEMS:
REVISION NEEDED:
- Issue 1: [specific problem and how to fix it]
- Issue 2: [specific problem and how to fix it]
(etc.)

Be specific. Don't flag style issues. Only flag factual or logical problems."""

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}]
    )

    critique_response = message.content[0].text.strip()
    print(f"[Critic] Review complete: {critique_response[:100]}...")

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
    LangGraph routing function.
    Returns "revise" to send back to writer, or "done" to finish.
    """
    if state.get("critique_passed", False):
        return "done"
    return "revise"
```

---

### Task 14: Wire everything together with LangGraph

**Step 1: Create `backend/agents/pipeline.py`**:
```python
from langgraph.graph import StateGraph, END
from agents.state import ResearchState
from agents.orchestrator import orchestrator_node
from agents.collector import collector_node
from agents.writer import writer_node
from agents.critic import critic_node, should_revise


def build_research_pipeline():
    """
    Builds and compiles the LangGraph state machine.

    Flow:
    orchestrator → collector → writer → critic → (done OR back to writer)
    """
    graph = StateGraph(ResearchState)

    # Add all nodes
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("collector", collector_node)
    graph.add_node("writer", writer_node)
    graph.add_node("critic", critic_node)

    # Define the flow
    graph.set_entry_point("orchestrator")
    graph.add_edge("orchestrator", "collector")
    graph.add_edge("collector", "writer")
    graph.add_edge("writer", "critic")

    # Conditional edge: critic either approves (END) or sends back to writer
    graph.add_conditional_edges(
        "critic",
        should_revise,
        {
            "revise": "writer",
            "done": END,
        }
    )

    return graph.compile()


def run_research(company: str) -> dict:
    """
    Run the full research pipeline for a company.
    Returns the final state with the completed memo.
    """
    pipeline = build_research_pipeline()

    initial_state = {
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

    print(f"\n{'='*50}")
    print(f"Starting research pipeline for: {company}")
    print(f"{'='*50}\n")

    final_state = pipeline.invoke(initial_state)

    print(f"\n{'='*50}")
    print(f"Pipeline complete! Revisions: {final_state.get('revision_count', 0)}")
    print(f"{'='*50}\n")

    return final_state
```

**Step 2: Test the full pipeline end-to-end**

Create `backend/tests/test_pipeline.py`:
```python
from agents.pipeline import run_research

def test_full_pipeline_apple():
    """Run the full pipeline on Apple. This makes real API calls."""
    result = run_research("Apple")

    assert result.get("final_memo") is not None, "No final memo produced"
    assert len(result["final_memo"]) > 200, "Memo is too short"
    assert result.get("critique_passed") is True

    print("\n" + "="*60)
    print("FINAL MEMO:")
    print("="*60)
    print(result["final_memo"][:1000] + "...")
    print(f"\nRevisions made: {result.get('revision_count', 0)}")
    print(f"Confidence scores: {result.get('confidence_scores', {})}")
```

Run it (this makes real API calls, takes ~30-60 seconds):
```bash
python -m pytest tests/test_pipeline.py -v -s
```
Expected: `PASSED` with the memo printed to terminal

---

## PHASE 4: FastAPI Backend

### Task 15: Build the FastAPI server

**Step 1: Create `backend/main.py`**:
```python
import os
import json
import asyncio
from typing import AsyncIterator
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv("../.env")

# Add the backend directory to path so imports work
import sys
sys.path.insert(0, os.path.dirname(__file__))

from agents.pipeline import run_research

app = FastAPI(title="Research Agent API", version="1.0.0")

# Allow the Next.js frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://*.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResearchRequest(BaseModel):
    company: str


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Research Agent API is running"}


@app.post("/api/research")
async def research_company(request: ResearchRequest):
    """
    Run the full research pipeline for a company.
    Returns the completed investment memo.
    """
    if not request.company or len(request.company.strip()) < 2:
        raise HTTPException(status_code=400, detail="Company name too short")

    # Run the pipeline (this takes 30-60 seconds)
    # In a production app, you'd make this async and stream progress
    try:
        result = run_research(request.company.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")

    return {
        "company": request.company,
        "memo": result.get("final_memo"),
        "confidence_scores": result.get("confidence_scores", {}),
        "sources": result.get("sources_used", []),
        "revisions": result.get("revision_count", 0),
    }
```

**Step 2: Test the server manually**
```bash
cd backend
uvicorn main:app --reload --port 8000
```
Expected: `Uvicorn running on http://127.0.0.1:8000`

Open a new terminal and test:
```bash
curl -X POST http://localhost:8000/api/research \
  -H "Content-Type: application/json" \
  -d '{"company": "Apple"}'
```
Expected: JSON response with the memo (takes ~60 seconds)

---

## PHASE 5: Next.js Frontend

### Task 16: Set up Next.js

**Step 1: Go to the frontend folder**
```bash
cd /Users/sriram/Downloads/Final_Projects/research-agent
```

**Step 2: Create Next.js app**
```bash
npx create-next-app@latest frontend --typescript --tailwind --app --no-src-dir --import-alias "@/*"
```
When prompted, hit Enter for all defaults.

**Step 3: Go into the frontend folder**
```bash
cd frontend
```

**Step 4: Add an environment variable for the API URL**

Create `frontend/.env.local`:
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

### Task 17: Build the main UI page

**Step 1: Replace `frontend/app/page.tsx` with**:
```tsx
"use client";
import { useState } from "react";

interface ResearchResult {
  company: string;
  memo: string;
  confidence_scores: Record<string, string>;
  sources: string[];
  revisions: number;
}

export default function Home() {
  const [company, setCompany] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ResearchResult | null>(null);
  const [error, setError] = useState("");

  const handleResearch = async () => {
    if (!company.trim()) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/research`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ company: company.trim() }),
        }
      );

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || "Something went wrong. Is the backend running?");
    } finally {
      setLoading(false);
    }
  };

  const confidenceColor = (score: string) => {
    if (score === "high") return "bg-green-100 text-green-800";
    if (score === "medium") return "bg-yellow-100 text-yellow-800";
    return "bg-red-100 text-red-800";
  };

  return (
    <main className="min-h-screen bg-gray-50">
      <div className="max-w-4xl mx-auto px-4 py-12">
        {/* Header */}
        <div className="text-center mb-10">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Research Agent
          </h1>
          <p className="text-gray-500 text-lg">
            Autonomous investment memos powered by AI
          </p>
        </div>

        {/* Input */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 mb-8">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Company Name
          </label>
          <div className="flex gap-3">
            <input
              type="text"
              value={company}
              onChange={(e) => setCompany(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleResearch()}
              placeholder="e.g. Apple, Rivian, Nvidia..."
              className="flex-1 border border-gray-300 rounded-lg px-4 py-2.5 text-gray-900 focus:outline-none focus:ring-2 focus:ring-blue-500"
              disabled={loading}
            />
            <button
              onClick={handleResearch}
              disabled={loading || !company.trim()}
              className="px-6 py-2.5 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? "Researching..." : "Analyze"}
            </button>
          </div>
        </div>

        {/* Loading State */}
        {loading && (
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-8 text-center">
            <div className="animate-spin w-8 h-8 border-4 border-blue-600 border-t-transparent rounded-full mx-auto mb-4" />
            <p className="text-gray-600 font-medium">
              Running research pipeline...
            </p>
            <p className="text-gray-400 text-sm mt-1">
              Fetching SEC filings, searching news, writing memo, running
              critique
            </p>
            <p className="text-gray-400 text-sm">This takes about 60 seconds</p>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-xl p-4 text-red-700">
            {error}
          </div>
        )}

        {/* Result */}
        {result && (
          <div className="space-y-6">
            {/* Meta info */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-4 flex items-center justify-between">
              <div>
                <span className="text-sm text-gray-500">Research complete for </span>
                <span className="font-semibold text-gray-900">{result.company}</span>
              </div>
              <div className="flex items-center gap-4 text-sm text-gray-500">
                <span>{result.revisions} revision{result.revisions !== 1 ? "s" : ""} by critic</span>
              </div>
            </div>

            {/* Confidence Scores */}
            {Object.keys(result.confidence_scores).length > 0 && (
              <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
                <h2 className="font-semibold text-gray-900 mb-3">
                  Confidence Scores
                </h2>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(result.confidence_scores).map(
                    ([section, score]) => (
                      <span
                        key={section}
                        className={`px-3 py-1 rounded-full text-xs font-medium ${confidenceColor(score)}`}
                      >
                        {section.replace(/_/g, " ")}: {score}
                      </span>
                    )
                  )}
                </div>
              </div>
            )}

            {/* The Memo */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <h2 className="font-semibold text-gray-900 mb-4">
                Investment Memo
              </h2>
              <div className="prose prose-sm max-w-none text-gray-700 whitespace-pre-wrap font-mono text-sm leading-relaxed">
                {result.memo}
              </div>
            </div>

            {/* Sources */}
            {result.sources.length > 0 && (
              <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
                <h2 className="font-semibold text-gray-900 mb-3">
                  Sources Used
                </h2>
                <ul className="space-y-1">
                  {result.sources.map((source, i) => (
                    <li key={i} className="text-sm text-gray-600">
                      • {source}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </div>
    </main>
  );
}
```

**Step 2: Run the frontend**
```bash
npm run dev
```
Expected: `Ready on http://localhost:3000`

**Step 3: Open http://localhost:3000 in your browser**

Make sure the backend is running in another terminal first, then type a company name and click Analyze.

---

## PHASE 6: Running the Full App

### Task 18: Run frontend + backend together

**Terminal 1 (backend):**
```bash
cd /Users/sriram/Downloads/Final_Projects/research-agent/backend
source ../venv/bin/activate
uvicorn main:app --reload --port 8000
```

**Terminal 2 (frontend):**
```bash
cd /Users/sriram/Downloads/Final_Projects/research-agent/frontend
npm run dev
```

Open http://localhost:3000, type "Apple" or "Nvidia", click Analyze. Watch it work.

---

## Success Checklist
- [ ] pyenv installed, Python 3.11 active in project folder
- [ ] Virtual environment created and activated
- [ ] All dependencies installed without errors
- [ ] SEC EDGAR tool returns real filing data
- [ ] Tavily search returns real news articles
- [ ] Full pipeline test passes (test_pipeline.py)
- [ ] FastAPI server runs on port 8000
- [ ] Next.js UI runs on port 3000
- [ ] End-to-end: type company name, get investment memo in browser
