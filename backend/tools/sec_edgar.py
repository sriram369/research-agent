import requests
import re
from typing import Optional
from bs4 import BeautifulSoup

EDGAR_BASE = "https://data.sec.gov"
HEADERS = {"User-Agent": "research-agent contact@example.com"}


def get_company_filings(company_name: str) -> Optional[dict]:
    """
    Look up a company using EDGAR's official tickers file.
    This is more reliable than the search endpoint.
    """

    # EDGAR publishes a complete list of all public companies with their CIK numbers
    tickers_url = "https://www.sec.gov/files/company_tickers.json"
    response = requests.get(tickers_url, headers=HEADERS)

    if response.status_code != 200:
        return None

    tickers_data = response.json()

    # Search for the company by name (case insensitive)
    search_term = company_name.lower()
    best_match = None

    for entry in tickers_data.values():
        title = entry.get("title", "").lower()
        ticker = entry.get("ticker", "").lower()

        # Exact ticker match (e.g. user types "AAPL")
        if ticker == search_term:
            best_match = entry
            break

        # Company name starts with search term (e.g. "apple" matches "Apple Inc.")
        if title.startswith(search_term):
            best_match = entry
            break

        # Company name contains search term
        if search_term in title and best_match is None:
            best_match = entry

    if not best_match:
        return None

    return {
        "cik": str(best_match["cik_str"]),
        "company_name": best_match["title"],
        "ticker": best_match["ticker"],
    }


def get_filing_text(cik: str, max_chars: int = 8000) -> Optional[str]:
    """Pull the most recent 10-K filing and return the key text."""

    url = f"{EDGAR_BASE}/submissions/CIK{cik.zfill(10)}.json"
    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        return None

    filings = response.json()
    company_name = filings.get("name", "Unknown")

    recent = filings.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accession_numbers = recent.get("accessionNumber", [])
    primary_documents = recent.get("primaryDocument", [])

    ten_k_index = None
    for i, form in enumerate(forms):
        if form == "10-K":
            ten_k_index = i
            break

    if ten_k_index is None:
        return f"No 10-K filing found for {company_name}"

    # Use primaryDocument directly from the submissions JSON — no index fetch needed
    accession = accession_numbers[ten_k_index].replace("-", "")
    main_doc = primary_documents[ten_k_index]

    doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{main_doc}"
    doc_response = requests.get(doc_url, headers=HEADERS)

    if doc_response.status_code != 200:
        return f"Couldn't download 10-K for {company_name} (status {doc_response.status_code})"

    soup = BeautifulSoup(doc_response.text, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r'\s+', ' ', text)

    # iXBRL filings have a huge metadata block at the top — skip past it
    # Find where the actual readable content starts
    markers = ["UNITED STATES", "FORM 10-K", "Annual Report", "fiscal year"]
    start = 0
    for marker in markers:
        idx = text.find(marker)
        if idx > 0:
            start = idx
            break

    readable_text = text[start:]
    return f"[10-K Filing for {company_name}]\n\n{readable_text[:max_chars]}..."
