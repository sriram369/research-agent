from tools.sec_edgar import get_company_filings, get_filing_text

info = get_company_filings("Apple")
print("Company info:", info)

text = get_filing_text(info["cik"])
print("\nFiling preview:")
print(text[:800])