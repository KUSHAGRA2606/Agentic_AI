"""
Two tools created:
1) search_arxiv(query, max_results)   --> query is the search query, max_results is max no of papers we want tool to retrieve
2) health_check()  --> to see if mcp server is running properly
"""

from typing import Any

import arxiv
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("arxiv-research-server")

@mcp.tool()
def search_arxiv(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """
    Search ArXiv for research papers matching a query.

    Args:
        query: Search query string.
        max_results: Maximum number of papers to return.

    Returns:
        A list of paper dictionaries.
    """
    if not query or not query.strip():
        raise ValueError("query cannot be empty")

    max_results = max(1, min(max_results, 10))

    client = arxiv.Client()

    search = arxiv.Search(
        query=query.strip(),
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers = []

    for result in client.results(search):
        paper = {
            "paper_id": result.entry_id.split("/")[-1],
            "title": result.title.strip(),
            "authors": [author.name for author in result.authors],
            "summary": result.summary.strip(),
            "published": result.published.strftime("%Y-%m-%d"),
            "url": result.entry_id,
            "pdf_url": result.pdf_url,
            "categories": result.categories,
        }

        papers.append(paper)

    return papers


@mcp.tool()
def health_check() -> dict[str, str]:
    """
    Simple test tool to verify that the MCP server is running.
    """
    return {
        "status": "ok",
        "server": "arxiv-research-server",
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")