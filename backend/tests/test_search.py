from tools.web_search import search_company_news

results = search_company_news("Apple Inc latest earnings news 2025")
for r in results:
    print(r["title"])
    print(r["url"])
    print()