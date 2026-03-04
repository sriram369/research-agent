# Research Agent

Autonomous investment memo generator with a self-critique loop. Powered by real SEC EDGAR filings, live news via Tavily, and Claude AI.

## How it works

```
Orchestrator → Collector → Writer → Critic → (revise loop, max 3x) → Final Memo
```

- **Orchestrator** — Claude breaks the query into research tasks
- **Collector** — fetches real 10-K filings from SEC EDGAR + live news from Tavily
- **Writer** — Claude Sonnet writes a structured investment memo
- **Critic** — Claude fact-checks every claim, sends back for revision if hallucinations found

## Stack

| Layer | Tech |
|-------|------|
| Agent framework | LangGraph |
| LLM | Claude Haiku + Sonnet (Anthropic) |
| Web search | Tavily API |
| Financial data | SEC EDGAR (free) |
| Backend | FastAPI + SSE streaming |
| Frontend | Next.js + Tailwind |
| Backend deploy | Railway |
| Frontend deploy | Vercel |

## Local setup

**Backend:**
```bash
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp ../.env.example .env  # add your API keys
uvicorn main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

## Environment variables

```
ANTHROPIC_API_KEY=your_key
TAVILY_API_KEY=your_key
```
