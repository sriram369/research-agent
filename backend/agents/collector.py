import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tools.sec_edgar import get_company_filings, get_filing_text
from tools.web_search import search_company_news
from agents.state import ResearchState


def collector_node(state: ResearchState) -> dict:
    """
    Stage 2 of the pipeline — the data fetcher.

    Reads: company name
    Writes: sec_data, news_data, sources_used

    No Claude involved here — just calls the two tools we already built.
    """
    company = state["company"]
    sources = []

    # Fetch SEC filing
    print(f"[Collector] Fetching SEC EDGAR data for {company}...")
    filing_info = get_company_filings(company)

    if filing_info:
        sec_text = get_filing_text(filing_info["cik"])
        sources.append(f"SEC EDGAR 10-K: {filing_info['company_name']} ({filing_info['ticker']})")
        print(f"[Collector] Got 10-K for {filing_info['company_name']}")
    else:
        sec_text = f"No SEC EDGAR filing found for {company}."
        print(f"[Collector] No SEC filing found for {company}")

    # Fetch recent news
    print(f"[Collector] Searching for recent news about {company}...")
    news_text = search_company_news(
        f"{company} stock earnings revenue news 2025",
        as_string=True
    )
    sources.append(f"Tavily Web Search: recent {company} news")
    print(f"[Collector] Got news articles")

    return {
        "sec_data": sec_text,
        "news_data": news_text,
        "sources_used": sources,
    }