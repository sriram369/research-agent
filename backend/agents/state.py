from typing import TypedDict, Optional, List


class ResearchState(TypedDict):
    """
    The shared whiteboard that flows through every stage of the pipeline.
    Each agent reads from it and writes to it.
    """

    # str = one single value. Set by the user at the start.
    company: str

    # List[str] = multiple items. Orchestrator writes ["find 10-K", "find news", ...]
    research_tasks: List[str]

    # Optional[str] = text that might not exist yet (None until Collector runs)
    sec_data: Optional[str]

    # Optional[str] = None until Collector fetches news from Tavily
    news_data: Optional[str]

    # Optional[str] = None until Writer produces the first draft
    draft_memo: Optional[str]

    # Optional[str] = None if Critic approved it. Has feedback text if not.
    critique: Optional[str]

    # bool = yes/no. True means Critic approved. False means send back to Writer.
    critique_passed: bool

    # int = counts how many revision loops have run. Stops at 3 to avoid infinite loops.
    revision_count: int

    # Optional[str] = None until Critic approves. Then holds the final approved memo.
    final_memo: Optional[str]

    # dict = key-value pairs. e.g. {"financials": "high", "risks": "medium"}
    confidence_scores: dict

    # List[str] = multiple sources. e.g. ["SEC EDGAR - Apple Inc.", "CNBC article"]
    sources_used: List[str]
