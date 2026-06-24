import os
import json
import operator
import time
from typing import Annotated, TypedDict, Optional, List, Dict, Any
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from langsmith import traceable

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from rag_core import rag_ingest, rag_query, COLLECTIONS
from mcp_tools import (
    search_arxiv_papers,
    search_github_repos,
    search_web_articles,
    search_hf_datasets,
    search_semantic_scholar
)


def merge_dicts(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
    """Safely merges dictionaries from parallel agent nodes."""
    if not a: a = {}
    if not b: b = {}
    return {**a, **b}


def get_pipeline_llm():
    return ChatNVIDIA(
        model="meta/llama-3.1-70b-instruct",
        temperature=0.2, 
        max_tokens=4096,
        base_url="https://integrate.api.nvidia.com/v1"
    )


class AgentState(TypedDict):
    ps: str                                 
    ps_parsed: Optional[Dict[str, Any]]     
    intent: Optional[str]                   
    user_docs_context: Optional[str]
    ps_summary: Optional[str]
    
    research_queries: List[str]
    github_search_terms: List[str]
    web_search_queries: List[str]
    
    research_report: Optional[Dict]
    github_report: Optional[Dict]
    web_report: Optional[Dict]
    hf_report: Optional[Dict]
    
    research_confidence: float
    github_confidence: float
    web_confidence: float
    
    conflicts: Annotated[List[str], operator.add]
    
    prd_version: int
    prd_sections: Dict[str, str]      
    notion_url: Optional[str]
    final_md: Optional[str]
    human_feedback: Optional[str]


@traceable(name="orchestrator", tags=["core", "nvidia"])
def orchestrator_node(state: AgentState) -> Dict[str, Any]:
    print("--> [Orchestrator] Parsing Problem Statement & Generating Queries...")
    llm = get_pipeline_llm()
    parser = JsonOutputParser()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the Lead Technical Architect. Analyze the Problem Statement and return a JSON object with:
        - "intent": Core goal interpretation.
        - "ps_summary": Dense search summary (under 150 words).
        - "ps_parsed": {{domain, core_challenge, constraints}}
        - "research_queries": [3-5 ArXiv search strings]
        - "github_search_terms": [3-5 GitHub search strings]
        - "web_search_queries": [3-5 general web search strings]"""),
        ("human", "Problem Statement: {ps}\n\nUser Context (if any): {context}")
    ])
    
    chain = prompt | llm | parser
    context = state.get("user_docs_context", "None provided.")
    
    try:
        parsed_data = chain.invoke({"ps": state["ps"], "context": context})
        
        return {
            "intent": parsed_data.get("intent", ""),
            "ps_summary": parsed_data.get("ps_summary", state["ps"][:300]),
            "ps_parsed": parsed_data.get("ps_parsed", {}),
            "research_queries": parsed_data.get("research_queries", []),
            "github_search_terms": parsed_data.get("github_search_terms", []),
            "web_search_queries": parsed_data.get("web_search_queries", [])
        }
    except Exception as e:
        print(f"--> [Orchestrator] Error parsing LLM output: {e}")
        return {
            "ps_summary": state["ps"][:300],
            "research_queries": [state["ps"][:200]],
            "github_search_terms": [state["ps"][:200]],
            "web_search_queries": [state["ps"][:200]]
        }


@traceable(name="research_agent", tags=["subagent", "nvidia"])
def research_agent_node(state: AgentState) -> Dict[str, Any]:
    print(f"--> [Research Agent] Executing {len(state.get('research_queries', []))} queries across ArXiv & Semantic Scholar...")
    all_papers = []
    
    for q in state.get("research_queries", []):
        arxiv_results = search_arxiv_papers.invoke({"query": q})
        s2_results = search_semantic_scholar.invoke({"query": q})
        all_papers.extend(arxiv_results)
        all_papers.extend(s2_results)
    
    seen_titles = set()
    unique_papers = []
    for p in all_papers:
        title_normalized = (p.get("title") or "").lower().strip()
        if title_normalized not in seen_titles and title_normalized != "":
            seen_titles.add(title_normalized)
            unique_papers.append(p)
    
    keywords = set(state.get("ps_summary", "").lower().split())
    scored_papers = []
    recent_count = 0
    current_year = datetime.now().year
    
    for paper in unique_papers:
        content = paper.get("abstract", "") or ""
        title = paper.get("title", "") or ""
        text_to_score = (title + " " + content).lower()
        
        kw_match = sum(1 for word in keywords if word in text_to_score)
        citation_bonus = min(paper.get("citations", 0) / 50.0, 3.0) 
        paper["relevance_score"] = kw_match + citation_bonus
        
        rag_ingest(
            content=content, 
            collection_name="papers", 
            metadata={
                "source_type": "paper", 
                "paper_id": paper.get("id", "unknown"), 
                "title": title
            }
        )
        
        if int(paper.get("year", 0)) >= current_year - 2:
            recent_count += 1
            
        scored_papers.append(paper)

    sorted_papers = sorted(scored_papers, key=lambda x: x.get("relevance_score", 0), reverse=True)
    N_p = len(sorted_papers)
    confidence = 0.0 if N_p == 0 else (0.4 * min(N_p / 10.0, 1.0) + 0.3 * (recent_count / N_p) + 0.3 * 0.8)

    return {
        "research_report": {
            "total_papers_found": N_p,
            "recent_papers": recent_count,
            "fetched_items": sorted_papers
        },
        "research_confidence": round(confidence, 2)
    }


@traceable(name="github_agent", tags=["subagent", "nvidia"])
def github_agent_node(state: AgentState) -> Dict[str, Any]:
    print(f"--> [GitHub Agent] Executing {len(state.get('github_search_terms', []))} queries...")
    all_repos = []
    
    for q in state.get("github_search_terms", []):
        results = search_github_repos.invoke({"query": q})
        all_repos.extend(results)
        
    unique_repos = {r["repo_url"]: r for r in all_repos if "repo_url" in r}.values()
    keywords = set(state.get("ps_summary", "").lower().split())
    scored_repos = []
    
    for repo in unique_repos:
        description = repo.get("description", "") or ""
        tech_stack = " ".join(repo.get("tech_stack", [])) or ""
        text_to_score = (description + " " + tech_stack).lower()
        
        kw_match = sum(1 for word in keywords if word in text_to_score)
        star_bonus = min(repo.get("stars", 0) / 1000.0, 5.0) 
        repo["relevance_score"] = kw_match + star_bonus
        
        rag_ingest(
            content=description, 
            collection_name="repos", 
            metadata={"source_type": "repo", "repo_url": repo["repo_url"]}
        )
        scored_repos.append(repo)

    sorted_repos = sorted(scored_repos, key=lambda x: x.get("relevance_score", 0), reverse=True)
    N_r = len(sorted_repos)
    confidence = 0.5 * min(N_r / 5.0, 1.0) + 0.5 if N_r > 0 else 0.5

    return {
        "github_report": {
            "total_repos_found": N_r,
            "fetched_items": sorted_repos
        },
        "github_confidence": round(confidence, 2)
    }


@traceable(name="web_agent", tags=["subagent", "nvidia"])
def web_agent_node(state: AgentState) -> Dict[str, Any]:
    print(f"--> [Web Agent] Executing {len(state.get('web_search_queries', []))} queries...")
    all_articles = []
    
    for q in state.get("web_search_queries", []):
        results = search_web_articles.invoke({"query": q})
        all_articles.extend(results)

    unique_articles = {a["url"]: a for a in all_articles if "url" in a}.values()
    keywords = set(state.get("ps_summary", "").lower().split())
    scored_articles = []
    
    for art in unique_articles:
        content = art.get("content", "") or ""
        title = art.get("title", "") or ""
        text_to_score = (title + " " + content).lower()
        
        art["relevance_score"] = sum(1 for word in keywords if word in text_to_score)
        
        rag_ingest(
            content=content, 
            collection_name="web", 
            metadata={"source_type": "web", "url": art.get("url", "")}
        )
        scored_articles.append(art)

    sorted_articles = sorted(scored_articles, key=lambda x: x.get("relevance_score", 0), reverse=True)
    N_a = len(sorted_articles)
    confidence = 0.8 * min(N_a / 8.0, 1.0) if N_a > 0 else 0.5

    return {
        "web_report": {
            "total_articles": N_a,
            "fetched_items": sorted_articles
        },
        "web_confidence": round(confidence, 2)
    }


@traceable(name="conflict_detect", tags=["core", "nvidia"])
def conflict_detect_node(state: AgentState) -> Dict[str, Any]:
    print("--> [Conflict Detection] Cross-referencing reports...")
    llm = get_pipeline_llm()
    parser = JsonOutputParser()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Analyze the summaries of Research, GitHub, and Web reports. 
        Identify any major contradictions.
        Return JSON with a single key "conflicts" containing a list of string descriptions. If none, return an empty list."""),
        ("human", "Research: {res}\nGitHub: {git}\nWeb: {web}")
    ])
    
    chain = prompt | llm | parser
    try:
        result = chain.invoke({
            "res": json.dumps(state.get("research_report", {})),
            "git": json.dumps(state.get("github_report", {})),
            "web": json.dumps(state.get("web_report", {}))
        })
        return {"conflicts": result.get("conflicts", [])}
    except Exception:
        return {"conflicts": []}


@traceable(name="feasibility_check", tags=["core"])
def feasibility_sanity_check_node(state: AgentState) -> Dict[str, Any]:
    print("--> [Feasibility Check] Verifying dataset availability...")
    clean_keywords = state.get("ps_parsed", {}).get("core_challenge", state["ps"][:50])
    
    try:
        datasets = search_hf_datasets.invoke({"query": clean_keywords})
    except Exception as e:
        print(f"--> [Feasibility Check] Tool invocation failed: {e}")
        datasets = []
        
    if datasets is None:
        datasets = []
        
    keywords = set(state.get("ps_summary", "").lower().split())
    scored_datasets = []
    
    for ds in datasets:
        name = ds.get("name", "").lower()
        kw_match = sum(2 for word in keywords if word in name)
        download_bonus = min(ds.get("downloads_count", 0) / 10000.0, 5.0)
        ds["relevance_score"] = kw_match + download_bonus
        scored_datasets.append(ds)
        
    sorted_datasets = sorted(scored_datasets, key=lambda x: x.get("relevance_score", 0), reverse=True)
    
    conflicts_update = []
    if not sorted_datasets:
        conflicts_update.append("WARNING: No relevant free datasets found on HuggingFace.")
        
    return {
        "conflicts": conflicts_update,
        "hf_report": {
            "fetched_items": sorted_datasets
        }
    }


@traceable(name="prd_agent", tags=["output", "nvidia"])
def prd_agent_node(state: AgentState) -> Dict[str, Any]:
    print("--> [PRD Agent] Generating Hardcoded Contextual PRD Sections...")
    llm = get_pipeline_llm()
    
    HARDCODED_SECTIONS = ["Executive Summary", "System Architecture", "ML Pipeline", "Dataset Strategy", "Tech Stack"]
    
    generated_sections = {}
    search_query_base = state.get("ps_summary", state["ps"][:300])
    
    for sec in HARDCODED_SECTIONS:
        print(f"    [PRD Agent] Drafting section: {sec}")
        time.sleep(1.0)  
        
        context = rag_query(
            query=f"{sec} engineering requirements technical specifications for {search_query_base}",
            target_collections=["papers", "repos", "web"],
            top_k=6
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a Principal Software Architect and Technical Product Manager.
            Write an implementation-ready, deeply technical section titled '{sec}' for the PRD.
            
            Determine the best technical subsections, structural design patterns, and operational boundaries yourself based on the requirements.
            
            CRITICAL INSTRUCTIONS FOR HIGH SPECIFICITY & COMPLETENESS:
            - Avoid high-level hand-waving or generic phrases like "industry-standard tools will be applied". 
            - Dive straight into concrete details: name explicit design patterns (e.g., MVC, DAO, Repository), engine selections, data flows, and error-handling conditions.
            - If writing 'Tech Stack', you MUST output an explicit markdown comparison table matching: Component Domain, Selected Framework/Tool, and Integration Interface Protocol.
            - If writing 'ML Pipeline' or 'System Architecture', define concrete operational modules, tensor/data shapes, and fallback routes.
            - Ensure your response is comprehensive and fully formulated. Do not leave trailing sentences, markdown blocks, or placeholders hanging. Base your technical claims strictly on verified facts from the context."""),
            ("human", "HACKATHON PROBLEM STATEMENT:\n{ps}\n\nRAG EXTRACTED CONTEXT BASE:\n{context}")
        ])
        
        chain = prompt | llm
        try:
            response = chain.invoke({"ps": state["ps"], "context": context})
            generated_sections[sec] = response.content
        except Exception as e:
            print(f"    [PRD Agent] Error generating section '{sec}': {e}")
            generated_sections[sec] = f"*(Technical specification generation for {sec} dropped due to connection timeout limits)*"
            
    return {
        "prd_sections": generated_sections,
        "prd_version": state.get("prd_version", 0) + 1
    }


from mcp_tools import create_notion_prd_page

@traceable(name="notion_exporter", tags=["output"])
def notion_exporter_node(state: AgentState) -> Dict[str, Any]:
    print("--> [Notion Exporter] Compiling and syncing PRD with Source Appendix to Notion...")
    
    prd_sections = state.get("prd_sections", {})
    if not prd_sections:
        print("--> [Notion Exporter] No PRD content found to export.")
        return {"notion_url": None}
    
    markdown_content = f"# Product Requirements Document: {state['ps'][:100]}...\n\n"
    markdown_content += f"*Generated dynamically by HiveMind AI on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
    
    for title, content in prd_sections.items():
        markdown_content += f"## {title}\n{content}\n\n---\n\n"
        
    markdown_content += "## Appendix: Verified Sources Cited\n\n"
    
    papers = state.get("research_report", {}).get("fetched_items", [])
    repos = state.get("github_report", {}).get("fetched_items", [])
    web_slices = state.get("web_report", {}).get("fetched_items", [])
    
    if papers:
        markdown_content += "### Academic Literature\n"
        for p in papers[:5]: 
            markdown_content += f"- **[{p.get('title')}](https://arxiv.org/abs/{p.get('id')})** ({p.get('year')}) | Relevance Score: `{p.get('relevance_score', 0):.2f}`\n"
            markdown_content += f"  *Abstract:* *{p.get('abstract')[:200]}...*\n"
        markdown_content += "\n"
        
    if repos:
        markdown_content += "### Open-Source Codebases Scoped\n"
        for r in repos[:5]:
            markdown_content += f"- **[{r.get('repo_url').split('/')[-1]}]({r.get('repo_url')})** | ⭐ Stars: {r.get('stars')} | Tech Stack: `{', '.join(r.get('tech_stack', []))}`\n"
            markdown_content += f"  *Description:* {r.get('description')}\n"
        markdown_content += "\n"

    if web_slices:
        markdown_content += "### Industry Slices & Technical Documentation\n"
        for w in web_slices[:5]:
            markdown_content += f"- **[{w.get('title')}]({w.get('url')})**\n"
            
    page_title = f"HiveMind PRD: {state.get('intent', 'Generated Solution')[:50]}"
    
    notion_link = create_notion_prd_page.invoke({
        "title": page_title,
        "content": markdown_content
    })
    
    return {
        "final_md": markdown_content,
        "notion_url": notion_link
    }



def should_re_run_agents(state: AgentState) -> str:
    """Conditional edge: Checks if any agent confidence drops below threshold (0.4)"""
    r_conf = state.get("research_confidence", 0.0)
    g_conf = state.get("github_confidence", 0.0)
    w_conf = state.get("web_confidence", 0.0)
    
    if r_conf < 0.4 or g_conf < 0.4 or w_conf < 0.4:
        print(f"--> [Router] Low confidence detected (R: {r_conf}, G: {g_conf}, W: {w_conf}). Looping...")
        return "orchestrator_node"
        
    return "conflict_detect_node"


workflow = StateGraph(AgentState)

workflow.add_node("orchestrator_node", orchestrator_node)
workflow.add_node("research_agent_node", research_agent_node)
workflow.add_node("github_agent_node", github_agent_node)
workflow.add_node("web_agent_node", web_agent_node)
workflow.add_node("conflict_detect_node", conflict_detect_node)
workflow.add_node("feasibility_sanity_check_node", feasibility_sanity_check_node)
workflow.add_node("prd_agent_node", prd_agent_node)
workflow.add_node("notion_exporter_node", notion_exporter_node)

workflow.add_edge(START, "orchestrator_node")
workflow.add_edge("orchestrator_node", "research_agent_node")
workflow.add_edge("orchestrator_node", "github_agent_node")
workflow.add_edge("orchestrator_node", "web_agent_node")

workflow.add_conditional_edges(
    "research_agent_node",
    should_re_run_agents,
    {"orchestrator_node": "orchestrator_node", "conflict_detect_node": "conflict_detect_node"}
)
workflow.add_conditional_edges(
    "github_agent_node",
    should_re_run_agents,
    {"orchestrator_node": "orchestrator_node", "conflict_detect_node": "conflict_detect_node"}
)
workflow.add_conditional_edges(
    "web_agent_node",
    should_re_run_agents,
    {"orchestrator_node": "orchestrator_node", "conflict_detect_node": "conflict_detect_node"}
)

workflow.add_edge("conflict_detect_node", "feasibility_sanity_check_node")
workflow.add_edge("feasibility_sanity_check_node", "prd_agent_node")
workflow.add_edge("prd_agent_node", "notion_exporter_node")
workflow.add_edge("notion_exporter_node", END)

app = workflow.compile()



def run_pipeline(problem_statement: str):
    initial_state = {
        "ps": problem_statement,
        "research_queries": [],
        "github_search_terms": [],
        "web_search_queries": [],
        "prd_version": 0,
        "conflicts": [],
        "user_docs_context": "",
        "prd_sections": {},            
        "research_confidence": 0.0,
        "github_confidence": 0.0,
        "web_confidence": 0.0
    }
    
    for event in app.stream(initial_state, {"recursion_limit": 10}):
        for key, value in event.items():
            print(f"✅ Finished step: {key}")
            
    return event