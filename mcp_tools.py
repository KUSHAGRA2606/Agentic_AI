import os
import httpx
from typing import List, Dict, Any
from langchain_core.tools import tool
import feedparser
import time

from dotenv import load_dotenv
load_dotenv()


@tool
def search_arxiv_papers(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Searches ArXiv for academic papers related to the query."""
    print(f"   [Tool: ArXiv MCP] Fetching live data for: '{query}'")
    
    url = f"https://export.arxiv.org/api/query?search_query=all:{query}&max_results={max_results}&sortBy=relevance&sortOrder=descending"
    
    with httpx.Client() as client:
        response = client.get(url)
        if response.status_code != 200:
            return []

    feed = feedparser.parse(response.content)
    
    results = []
    for entry in feed.entries:
        results.append({
            "id": entry.id.split('/abs/')[-1], 
            "title": entry.title,
            "abstract": entry.summary,
            "year": entry.published[:4],         
            "source_type": "paper"
        })
        
    return results

@tool
def search_semantic_scholar(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Searches Semantic Scholar for academic papers and returns their metadata and citation counts."""
    print(f"   [Tool: Semantic Scholar MCP] Fetching live data for: '{query}'")
    
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key
    else:
        print("   [Tool: Semantic Scholar MCP] Warning: SEMANTIC_SCHOLAR_API_KEY not set. Running unauthenticated.")

    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": min(max_results, 10),
        "fields": "title,abstract,year,citationCount"
    }

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(url, headers=headers, params=params)
            if response.status_code != 200:
                print(f"   [Tool: Semantic Scholar MCP] Error {response.status_code}: {response.text}")
                return []
                
            data = response.json()
            papers = data.get("data", [])
            
            results = []
            for paper in papers:
                results.append({
                    "id": paper.get("paperId"),
                    "title": paper.get("title"),
                    "abstract": paper.get("abstract") or "No abstract available.",
                    "year": str(paper.get("year") or 0),
                    "citations": paper.get("citationCount", 0),
                    "source_type": "paper",
                    "platform": "semantic_scholar"
                })
            return results
    except Exception as e:
        print(f"   [Tool: Semantic Scholar MCP] Request dropped: {e}")
        return []

@tool
def search_github_repos(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Searches GitHub for repositories matching the tech stack or query.
    Extracts conceptual and metadata density to minimize noise and downstream tokens.
    """
    print(f"   [Tool: GitHub MCP] Initiating live repository search for: '{query}'")
    
    token = os.getenv("GITHUB_TOKEN")
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    else:
        print("   [Tool: GitHub MCP] Warning: GITHUB_TOKEN not set. Running unauthenticated (Strict Rate Limits Apply).")

    url = "https://api.github.com/search/repositories"
    params = {
        "q": query,
        "per_page": min(max_results, 100),
        "sort": "stars",
        "order": "desc"
    }

    for attempt in range(3):
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(url, headers=headers, params=params)
                
                if response.status_code in [403, 429] and "retry-after" in response.headers:
                    wait_time = int(response.headers["retry-after"])
                    print(f"   [Tool: GitHub MCP] Rate limited. Retrying after {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                elif response.status_code in [403, 429]:
                    print("   [Tool: GitHub MCP] Abrupt rate-limit fallback trigger.")
                    time.sleep(2 ** attempt)
                    continue
                
                if response.status_code != 200:
                    print(f"   [Tool: GitHub MCP] Error {response.status_code}: {response.text}")
                    return []
                
                data = response.json()
                items = data.get("items", [])
                
                results = []
                for item in items:
                    results.append({
                        "repo_url": item.get("html_url", ""),
                        "stars": item.get("stargazers_count", 0),
                        "tech_stack": [item.get("language")] if item.get("language") else [],
                        "description": item.get("description", "") or "No description provided.",
                        "source_type": "repo"
                    })
                return results

        except httpx.RequestError as e:
            print(f"   [Tool: GitHub MCP] Network connectivity failed: {e}")
            time.sleep(1)
            
    return []


@tool
def search_web_articles(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Uses Tavily API for high-quality, RAG-optimized web search results.
    Bypasses structural junk to provide dense context slices for downstream synthesis.
    """
    print(f"   [Tool: Web Search MCP] Initiating live web retrieval for: '{query}'")
    
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("   [Tool: Web Search MCP] Warning: TAVILY_API_KEY environment variable not set. Returning empty list.")
        return []

    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "advanced",
        "max_results": min(max_results, 20),
        "include_images": False,
        "include_answer": False
    }

    try:
        with httpx.Client(timeout=8.0) as client:
            response = client.post(url, json=payload)
            
            if response.status_code != 200:
                print(f"   [Tool: Web Search MCP] API Error {response.status_code}: {response.text}")
                return []
                
            data = response.json()
            results = data.get("results", [])
            
            formatted_results = []
            for r in results:
                formatted_results.append({
                    "url": r.get("url", ""),
                    "title": r.get("title", "Untitled Web Resource"),
                    "content": r.get("content", ""),
                    "source_type": "web" 
                })
            return formatted_results

    except httpx.RequestError as e:
        print(f"   [Tool: Web Search MCP] Request dropped due to connectivity limits: {e}")
        return []


from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

@tool
def search_hf_datasets(query: str) -> List[Dict[str, Any]]:
    """
    Searches Hugging Face Hub for free, available datasets matching the problem statement.
    Prioritizes highly-downloaded resources to ensure high data fidelity for validation.
    """
    print(f"   [Tool: HuggingFace MCP] Querying Hub datasets for: '{query}'")
    
    api = HfApi(token=os.getenv("HF_TOKEN")) 
    
    try:
        datasets = api.list_datasets(
            search=query,
            limit=5,
            sort="downloads"
        )
        
        results = []
        for d in datasets:
            downloads = getattr(d, "downloads", 0)
            author = getattr(d, "author", "unknown")
            
            tags = getattr(d, "tags", [])
            license_type = "unknown"
            for tag in tags:
                if tag.startswith("license:"):
                    license_type = tag.replace("license:", "")
                    break
            
            results.append({
                "name": d.id, 
                "author": author,
                "downloads_count": downloads,
                "license": license_type,
                "relevance_score": 1.0, 
                "source_type": "dataset"
            })
            
        return results

    except HfHubHTTPError as e: 
        print(f"   [Tool: HuggingFace MCP] API Error while fetching data: {e}")
        return []
    except Exception as e:
        print(f"   [Tool: HuggingFace MCP] Unexpected internal error: {e}")
        return []


@tool
def create_notion_prd_page(title: str, content: str) -> str:
    """
    Creates a dedicated PRD page in your Notion workspace using the official Markdown API.
    Natively accepts an enhanced Markdown string, avoiding raw Block JSON parsing limits.
    """
    print(f"   [Tool: Notion MCP] Initiating PRD export for: '{title}'")
    
    notion_token = os.getenv("NOTION_TOKEN")
    parent_page_id = os.getenv("NOTION_PAGE_ID")
    
    if not notion_token or not parent_page_id:
        print("   [Tool: Notion MCP] Configuration Error: NOTION_TOKEN or NOTION_PAGE_ID is missing.")
        return "ERROR: Missing API authentication credentials."

    url = "https://api.notion.com/v1/pages"
    headers = {
        "Authorization": f"Bearer {notion_token}",
        "Content-Type": "application/json",
        "Notion-Version": "2026-03-11"
    }

    payload = {
        "parent": {
            "type": "page_id",
            "page_id": parent_page_id
        },
        "properties": {
            "title": {
                "id": "title",
                "type": "title",
                "title": [
                    {
                        "type": "text",
                        "text": {
                            "content": title[:2000]
                        }
                    }
                ]
            }
        },
        "markdown": content
    }

    try:
        with httpx.Client(timeout=15.0) as client:
            response = client.post(url, headers=headers, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                page_url = data.get("url", "https://notion.so")
                print(f"   [Tool: Notion MCP] Success! PRD page securely synchronized at: {page_url}")
                return page_url
            else:
                print(f"   [Tool: Notion MCP] Validation/API Error {response.status_code}: {response.text}")
                return f"ERROR {response.status_code}: Unable to create document."
                
    except httpx.RequestError as e:
        print(f"   [Tool: Notion MCP] Failed to complete network task bounds: {e}")
        return "ERROR: Workspace synchronization timeout."


RESEARCH_TOOLS = [search_arxiv_papers, search_semantic_scholar] # 👈 Add it here
GITHUB_TOOLS = [search_github_repos]
WEB_TOOLS = [search_web_articles]
ORCHESTRATOR_TOOLS = [search_hf_datasets] 
OUTPUT_TOOLS = [create_notion_prd_page]