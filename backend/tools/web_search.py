import os
from typing import Union
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv("../.env")


def search_company_news(query: str, max_results: int = 5, as_string: bool = False) -> Union[list, str]:
    """
    Search for recent news about a company using Tavily.

    Args:
        query: What to search for, e.g. "Apple Inc latest earnings news"
        max_results: How many results to return
        as_string: If True, returns formatted text instead of a list
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
        })

    if as_string:
        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(f"[Source {i}] {r['title']}\nURL: {r['url']}\n{r['content']}")
        return "\n\n".join(formatted)

    return results
