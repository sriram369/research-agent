import os
import sys
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))
load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

from agents.pipeline import stream_research, run_research

app = FastAPI(title="Research Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResearchRequest(BaseModel):
    company: str


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Research Agent API is running"}


@app.get("/api/research/stream")
def research_stream(company: str):
    """
    SSE endpoint — streams rich pipeline events to the frontend.
    Each event carries enough detail to show the user exactly what's happening,
    like Perplexity shows its search steps.
    """
    if not company or len(company.strip()) < 2:
        raise HTTPException(status_code=400, detail="Company name too short")

    def event_generator():
        try:
            merged_state: dict = {"company": company.strip()}

            for node_name, state_update in stream_research(company.strip()):
                merged_state.update(state_update)
                merged_state["company"] = company.strip()

                event: dict = {"type": "stage", "stage": node_name}

                if node_name == "orchestrator":
                    tasks = state_update.get("research_tasks", [])
                    event["title"] = "Orchestrator — Planning research"
                    event["detail"] = f"Broke down query into {len(tasks)} tasks"
                    event["tasks"] = tasks

                elif node_name == "collector":
                    sources = state_update.get("sources_used", [])
                    sec = state_update.get("sec_data", "")
                    news = state_update.get("news_data", "")
                    event["title"] = "Collector — Fetching data"
                    event["detail"] = f"Retrieved {len(sec or '')} chars from SEC EDGAR · {len(news or '')} chars of news"
                    event["sources"] = sources

                elif node_name == "writer":
                    rev = merged_state.get("revision_count", 0)
                    draft = state_update.get("draft_memo", "")
                    had_critique = bool(merged_state.get("critique"))
                    if had_critique:
                        event["title"] = f"Writer — Revision {rev}"
                        event["detail"] = f"Addressed critic feedback · rewrote memo ({len(draft or '')} chars)"
                    else:
                        event["title"] = "Writer — Drafting memo"
                        event["detail"] = f"Wrote first draft ({len(draft or '')} chars)"
                    event["revision"] = rev

                elif node_name == "critic":
                    passed = state_update.get("critique_passed", False)
                    critique = state_update.get("critique", "")
                    rev = state_update.get("revision_count", 0)
                    if passed:
                        event["title"] = "Critic — Approved ✓"
                        event["detail"] = "Memo passed fact-check. No hallucinations found."
                        event["passed"] = True
                    else:
                        # Parse issue count from critique text
                        issue_count = critique.count("- Issue") if critique else 0
                        event["title"] = f"Critic — {issue_count} issue{'s' if issue_count != 1 else ''} found"
                        event["detail"] = f"Sending back to writer for revision {rev}"
                        event["passed"] = False
                        event["critique"] = critique  # Full critique text for display

                yield f"data: {json.dumps(event)}\n\n"

            # Done
            memo = merged_state.get("final_memo") or merged_state.get("draft_memo") or ""
            yield f"data: {json.dumps({'type': 'done', 'company': company.strip(), 'memo': memo, 'confidence_scores': merged_state.get('confidence_scores', {}), 'sources': merged_state.get('sources_used', []), 'revisions': merged_state.get('revision_count', 0)})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/research")
def research_company(request: ResearchRequest):
    if not request.company or len(request.company.strip()) < 2:
        raise HTTPException(status_code=400, detail="Company name too short")
    try:
        result = run_research(request.company.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    memo = result.get("final_memo") or result.get("draft_memo") or ""
    return {"company": request.company, "memo": memo, "confidence_scores": result.get("confidence_scores", {}), "sources": result.get("sources_used", []), "revisions": result.get("revision_count", 0)}
