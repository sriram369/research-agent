# Research Agent — Design Document
**Date:** 2026-03-03
**Project:** Autonomous Research Agent with Self-Critique
**Stack:** Python 3.11, LangGraph, FastAPI, Next.js, Claude API, Tavily, SEC EDGAR

---

## Problem
Companies pay $5K–$15K/month for junior analysts to write research reports on companies. It takes 20–40 hours per report. We automate this with a multi-agent pipeline that produces a full investment memo in minutes.

---

## What We're Building
A web app where you type a company name and get a full structured investment memo — sourced from real SEC filings and live news — with confidence scores per section.

**Input:** "Analyze Rivian as an investment opportunity"
**Output:** 2,000-word memo with business overview, financials, risks, competitors, bull/bear case, confidence scores, and citations.

---

## Architecture

### 4-Stage Agent Pipeline

```
User Input
    ↓
[Stage 1: Orchestrator Agent]
  - Breaks query into sub-tasks
  - "Find 10-K", "Get news", "Find competitors"
    ↓
[Stage 2: Data Collection Layer]
  - SEC EDGAR API → financial filings
  - Tavily API → recent news articles
  - Competitor detection via industry codes
    ↓
[Stage 3: Writer Agent]
  - Produces structured memo from collected data
  - Cites every claim with source
  - Outputs JSON: {section, content, sources, confidence}
    ↓
[Stage 4: Critic Agent]  ← THE DIFFERENTIATOR
  - Fact-checks every number against source data
  - Flags unsupported claims
  - Checks logical consistency
  - Sends back to Writer if issues found (max 3 loops)
    ↓
Final Report with Confidence Scores
```

---

## Folder Structure

```
research-agent/
├── backend/
│   ├── agents/
│   │   ├── orchestrator.py
│   │   ├── writer.py
│   │   └── critic.py
│   ├── tools/
│   │   ├── sec_edgar.py
│   │   └── web_search.py
│   ├── main.py          ← FastAPI app
│   └── requirements.txt
├── frontend/
│   ├── app/
│   │   └── page.tsx     ← Main UI
│   └── package.json
├── docs/
│   └── plans/
│       └── 2026-03-03-research-agent-design.md
├── .env                 ← API keys (never commit)
├── .gitignore
└── README.md
```

---

## APIs & Keys Needed

| Service | Purpose | Cost |
|---------|---------|------|
| Anthropic Claude | All AI reasoning agents | ~$0.25/report (Haiku) |
| Tavily | Web search for news | Free tier: 1,000/month |
| SEC EDGAR | Financial filings | Free, no key needed |

---

## Data Flow Detail

1. User types company name in browser
2. Frontend POST to `/api/research` with `{company: "Rivian"}`
3. FastAPI kicks off LangGraph state machine
4. LangGraph streams progress events back via SSE
5. Frontend displays real-time progress + final memo
6. Memo stored in memory (no DB needed for MVP)

---

## Build Order (Implementation Phases)

1. **Phase 1:** Project setup — pyenv, virtualenv, install deps
2. **Phase 2:** Tools — SEC EDGAR fetcher + Tavily search
3. **Phase 3:** Agents — Orchestrator, Writer, Critic with LangGraph
4. **Phase 4:** FastAPI backend with streaming endpoint
5. **Phase 5:** Next.js frontend with streaming UI
6. **Phase 6:** Deploy — Railway (backend) + Vercel (frontend)

---

## Success Criteria
- [ ] Type a company name, get a full memo in under 2 minutes
- [ ] Every factual claim has a citation
- [ ] Critic catches at least 1 issue per report on average
- [ ] Live demo link works for anyone
- [ ] Cost per report under $0.50
