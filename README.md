# Autonomous Research Agent with Self-Critique Loop

**One query → a fact-checked investment memo, streamed live.**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Visit-4A90D9?style=for-the-badge)](https://frontend-sriram369s-projects.vercel.app)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agent%20Framework-green?style=for-the-badge)](https://github.com/langchain-ai/langgraph)
[![Claude AI](https://img.shields.io/badge/Claude-Haiku%20+%20Sonnet-orange?style=for-the-badge)](https://anthropic.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge)](https://fastapi.tiangolo.com)

---

## What It Does

Type a company name. The agent autonomously:

1. Fetches the **real 10-K filing from SEC EDGAR**
2. Pulls **live news** via Tavily Search
3. Writes a structured **investment memo** with Claude Sonnet
4. Sends it to a **Critic agent** that fact-checks every claim
5. **Revises** (up to 3 rounds) until no hallucinations remain
6. **Streams the final memo** live to the browser — like Perplexity

The self-critique loop is the key differentiator: the Writer and Critic are separate Claude instances with different system prompts, preventing the model from agreeing with its own mistakes.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                    LangGraph Pipeline                │
│                                                     │
│  Orchestrator → Collector ──────────────────────┐   │
│                   │                             │   │
│             ┌─────┴──────┐                      │   │
│         SEC EDGAR      Tavily News               │   │
│             └─────┬──────┘                      │   │
│                   ▼                             │   │
│               Writer (Claude Sonnet)            │   │
│                   │                             │   │
│                   ▼                             │   │
│               Critic (Claude Sonnet) ───────────┘   │
│               [up to 3 revision rounds]             │
│                   │                                 │
│                   ▼                                 │
│             Final Memo (SSE stream)                 │
└─────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent Framework | LangGraph |
| LLM | Claude Haiku (orchestration) + Claude Sonnet (writing/critique) |
| Financial Data | SEC EDGAR API (free, real 10-K filings) |
| Live News | Tavily Search API |
| Streaming | FastAPI + Server-Sent Events (SSE) |
| Frontend | Next.js + Tailwind CSS |
| Backend Deploy | Railway |
| Frontend Deploy | Vercel |

---

## Key Technical Decisions

- **LangGraph state machine** — each agent node has defined inputs/outputs, making the revision loop deterministic and inspectable
- **Two separate Claude instances** — Writer and Critic have different system prompts so the Critic doesn't just rubber-stamp the Writer's output
- **SSE streaming** — the frontend receives tokens as they're generated, not after the full memo is done
- **SEC EDGAR directly** — no third-party financial data provider; real regulatory filings, always current

---

## Local Setup

**Backend**
```bash
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add your API keys
uvicorn main:app --reload --port 8000
```

**Frontend**
```bash
cd frontend
npm install
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
npm run dev
```

## Environment Variables

| Variable | Get It From |
|---|---|
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) |
| `TAVILY_API_KEY` | [app.tavily.com](https://app.tavily.com) |
