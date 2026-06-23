import os
import json
import operator
from typing import Annotated, TypedDict, Optional, List, Dict, Any
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from langsmith import traceable

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# Import internal modules
from rag_core import rag_ingest, rag_query, COLLECTIONS
from mcp_tools import (
    search_arxiv_papers,
    search_github_repos,
    search_web_articles,
    search_hf_datasets
)

# ==========================================
# 0. CUSTOM REDUCERS
# ==========================================

def merge_dicts(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
    """Safely merges dictionaries from parallel agent nodes."""
    if not a: a = {}
    if not b: b = {}
    return {**a, **b}

# ==========================================
# 1. LLM CONFIGURATION & STATE DEFINITION
# ==========================================

def get_pipeline_llm():
    return ChatNVIDIA(
        model="meta/llama-3.1-70b-instruct",
        temperature=0.2, 
        base_url="https://integrate.api.nvidia.com/v1"
    )

class AgentState(TypedDict):
    # Core (Standard types, NOT annotated. Nodes should NEVER return 'ps')
    ps: str                                 
    ps_parsed: Optional[Dict[str, Any]]     
    intent: Optional[str]                   
    user_docs_context: Optional[str]
    ps_summary: Optional[str]
    
    # Lists updated sequentially (Orchestrator overwrites these, so we don't use operator.add 
    # to avoid infinitely growing lists if the loop triggers)
    research_queries: List[str]
    github_search_terms: List[str]
    web_search_queries: List[str]
    
    # Reports written by specific unique agents
    research_report: Optional[Dict]
    github_report: Optional[Dict]
    web_report: Optional[Dict]
    hf_report: Optional[Dict]
    
    # Merged states: These MUST be annotated because parallel nodes write to them simultaneously
    confidence_scores: Annotated[Dict[str, float], merge_dicts]
    
    # Conflicts use operator.add so multiple nodes can append warnings
    conflicts: Annotated[List[str], operator.add]
    
    # Output variables
    prd_version: int
    prd_sections: Dict[str, str]
    notion_url: Optional[str]
    final_md: Optional[str]
    human_feedback: Optional[str]
    required_prd_sections: List[str]

# ==========================================
# 2. AGENT NODES (Delta Return Pattern)
# ==========================================

@traceable(name="orchestrator", tags=["core", "nvidia"])
def orchestrator_node(state: AgentState) -> Dict[str, Any]:
    print("--> [Orchestrator] Parsing Problem Statement & Classifying PRD Sections...")
    llm = get_pipeline_llm()
    parser = JsonOutputParser()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the Lead Technical Architect. Analyze the Problem Statement and return a JSON object with:
        - "intent": Core goal interpretation.
        - "ps_summary": Dense search summary (under 150 words).
        - "ps_parsed": {{domain, core_challenge, constraints}}
        - "research_queries": [3-5 ArXiv search strings]
        - "github_search_terms": [3-5 GitHub search strings]
        - "web_search_queries": [3-5 general web search strings]
        - "required_prd_sections": A list of relevant technical sections to generate. Always include 'Executive Summary', 'System Architecture', and 'Tech Stack'. 
                                  ONLY include 'ML Pipeline & Modeling Strategy' if the challenge explicitly involves training, fine-tuning, or inference pipelines. 
                                  ONLY include 'Data Engineering & Acquisition Strategy' if it requires heavy dataset ingestion, web-scraping, or knowledge graphs.
                                  Include sections like 'Security & Compliance' or 'Scale & Infrastructure' if the constraints demand it."""),
        ("human", "Problem Statement: {ps}\n\nUser Context (if any): {context}")
    ])
    
    chain = prompt | llm | parser
    context = state.get("user_docs_context", "None provided.")
    
    try:
        parsed_data = chain.invoke({"ps": state["ps"], "context": context})
        
        return {
            "intent": parsed_data.get("intent", ""),
            "ps_summary": parsed_data.get("ps_summary", state["ps"][:300]), # Fallback to short slice if missing
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
    print(f"--> [Research Agent] Executing {len(state.get('research_queries', []))} queries...")
    all_papers = []
    
    for q in state.get("research_queries", []):
        arxiv_results = search_arxiv_papers.invoke({"query": q})
        all_papers.extend(arxiv_results)
    
    unique_papers = {p["id"]: p for p in all_papers if "id" in p}.values()
    
    recent_count = 0
    current_year = datetime.now().year
    
    for paper in unique_papers:
        content = paper.get("abstract", "") or paper.get("content", "No abstract available.")
        rag_ingest(
            content=content, 
            collection_name="papers", 
            metadata={
                "source_type": "paper", 
                "paper_id": paper["id"], 
                "title": paper.get("title", "")
            }
        )
        if int(paper.get("year", 0)) >= current_year - 2:
            recent_count += 1

    N_p = len(unique_papers)
    if N_p == 0:
        confidence = 0.0  # Prevents instant failure loop if search yields nothing
    else:
        A_s = 0.8 # Kept static as requested
        confidence = 0.4 * min(N_p / 10.0, 1.0) + 0.3 * (recent_count / N_p) + 0.3 * A_s

    # RETURN ONLY WHAT THIS AGENT MODIFIES
    return {
        "research_report": {
            "total_papers_found": N_p,
            "recent_papers": recent_count,
            "key_findings": [f"Ingested {N_p} papers into RAG."],
            "fetched_items": unique_papers
        },
        "confidence_scores": {"research": round(confidence, 2)}
    }


@traceable(name="github_agent", tags=["subagent", "nvidia"])
def github_agent_node(state: AgentState) -> Dict[str, Any]:
    print(f"--> [GitHub Agent] Executing {len(state.get('github_search_terms', []))} queries...")
    all_repos = []
    
    for q in state.get("github_search_terms", []):
        results = search_github_repos.invoke({"query": q})
        all_repos.extend(results)
        
    unique_repos = {r["repo_url"]: r for r in all_repos if "repo_url" in r}.values()
    
    for repo in unique_repos:
        rag_ingest(
            content=repo.get("description", "No description."), 
            collection_name="repos", 
            metadata={"source_type": "repo", "repo_url": repo["repo_url"]}
        )

    N_r = len(unique_repos)
    confidence = 0.5 * min(N_r / 5.0, 1.0) + 0.5 if N_r > 0 else 0.5

    return {
        "github_report": {
            "total_repos_found": N_r,
            "fetched_items": list(unique_repos)
        },
        "confidence_scores": {"github": round(confidence, 2)}
    }


@traceable(name="web_agent", tags=["subagent", "nvidia"])
def web_agent_node(state: AgentState) -> Dict[str, Any]:
    print(f"--> [Web Agent] Executing {len(state.get('web_search_queries', []))} queries...")
    all_articles = []
    
    for q in state.get("web_search_queries", []):
        results = search_web_articles.invoke({"query": q})
        all_articles.extend(results)

    unique_articles = list({a["url"]: a for a in all_articles if "url" in a}.values())

    N_a = len(all_articles)
    for art in all_articles:
        rag_ingest(
            content=art.get("content", ""), 
            collection_name="web", 
            metadata={"source_type": "web", "url": art.get("url", "")}
        )

    confidence = 0.8 * min(N_a / 8.0, 1.0) if N_a > 0 else 0.5

    return {
        "web_report": {
            "total_articles": N_a,
            "fetched_items": unique_articles,
        },
        
        "confidence_scores": {"web": round(confidence, 2)}
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


# In workflow.py

@traceable(name="feasibility_check", tags=["core"])
def feasibility_sanity_check_node(state: AgentState) -> Dict[str, Any]:
    print("--> [Feasibility Check] Verifying dataset availability...")
    
    # Query HuggingFace using the clean architectural intent summary
    search_query = state.get("ps_summary", state["ps"])[:300]
    datasets = search_hf_datasets.invoke({"query": search_query})
    
    # If no datasets are recovered, register the conflict warning
    conflicts_update = []
    if not datasets:
        conflicts_update.append("WARNING: No relevant free datasets found on HuggingFace.")
        
    return {
        "conflicts": conflicts_update,
        # Save the dataset dictionary array into an easily queryable sub-report key
        "hf_report": {
            "fetched_items": datasets if datasets else []
        }
    }


@traceable(name="prd_agent", tags=["output", "nvidia"])
def prd_agent_node(state: AgentState) -> Dict[str, Any]:
    print("--> [PRD Agent] Generating Contextual PRD Sections...")
    llm = get_pipeline_llm()
    
    # Fallback to standard core sections if the key is missing or empty
    dynamic_sections = state.get("required_prd_sections")
    if not dynamic_sections:
        dynamic_sections = ["Executive Summary", "System Architecture", "Tech Stack"]
        
    generated_sections = {}
    search_query_base = state.get("ps_summary", state["ps"][:300])
    
    for sec in dynamic_sections:
        print(f"    [PRD Agent] Drafting: {sec}")
        context = rag_query(
            query=f"{sec} engineering requirements specifications for {search_query_base}",
            target_collections=["papers", "repos", "web"],
            top_k=5
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a Principal Software Architect and Technical Product Manager. 
            Write an implementation-ready, highly technical section titled '{sec}' for the PRD.
            
            CRITICAL INSTRUCTIONS FOR HIGH SPECIFICITY:
            - Provide concrete structural patterns, engineering choices, and precise frameworks. Avoid generic hand-waving.
            - Base technical claims strictly on verified facts from the RAG Context. 
            - If writing 'Tech Stack', output an explicit markdown table showing: Component, Chosen Framework/Tool, and Integration Protocol.
            - If the section mentions data schemas, write out clean, hypothetical JSON or database schema blocks."""),
            ("human", "HACKATHON PROBLEM STATEMENT:\n{ps}\n\nRAG EXTRACTED CONTEXT:\n{context}")
        ])
        
        chain = prompt | llm
        response = chain.invoke({"ps": state["ps"], "context": context})
        generated_sections[sec] = response.content
        
    return {
        "prd_sections": generated_sections,
        "prd_version": state.get("prd_version", 0) + 1
    }

from mcp_tools import create_notion_prd_page  # Ensure this is imported at the top

@traceable(name="notion_exporter", tags=["output"])
def notion_exporter_node(state: AgentState) -> Dict[str, Any]:
    print("--> [Notion Exporter] Compiling and syncing PRD to Notion workspace...")
    
    prd_sections = state.get("prd_sections", {})
    if not prd_sections:
        print("--> [Notion Exporter] No PRD content found to export.")
        return {"notion_url": None}
    
    # Compile sections dictionary into a single unified Markdown string
    markdown_content = f"# Product Requirements Document: {state['ps'][:100]}...\n\n"
    markdown_content += f"*Generated dynamically by HiveMind AI on {datetime.now().strftime('%Y-%m-%d')}*\n\n"
    
    for title, content in prd_sections.items():
        markdown_content += f"## {title}\n{content}\n\n---\n\n"
    
    # Format a clean title for the Notion Page
    page_title = f"HiveMind PRD: {state.get('intent', 'Generated Solution')[:50]}"
    
    # Invoke your existing MCP Tool
    notion_link = create_notion_prd_page.invoke({
        "title": page_title,
        "content": markdown_content
    })
    
    return {
        "final_md": markdown_content,
        "notion_url": notion_link
    }

# ==========================================
# 3. ROUTING LOGIC & GRAPH COMPILATION
# ==========================================

def should_re_run_agents(state: AgentState) -> str:
    """Conditional edge: Checks if any agent confidence is below threshold (0.4)"""
    scores = state.get("confidence_scores", {})
    if not scores:
        return "conflict_detect_node"
        
    if any(score < 0.4 for score in scores.values()):
        print(f"--> [Router] Low confidence detected: {scores}. Looping back to Orchestrator...")
        return "orchestrator_node"
        
    return "conflict_detect_node"

# Initialize Graph
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("orchestrator_node", orchestrator_node)
workflow.add_node("research_agent_node", research_agent_node)
workflow.add_node("github_agent_node", github_agent_node)
workflow.add_node("web_agent_node", web_agent_node)
workflow.add_node("conflict_detect_node", conflict_detect_node)
workflow.add_node("feasibility_sanity_check_node", feasibility_sanity_check_node)
workflow.add_node("prd_agent_node", prd_agent_node)
workflow.add_node("notion_exporter_node", notion_exporter_node)

# Define Edges
workflow.add_edge(START, "orchestrator_node")

# Orchestrator fans out to parallel subagents
workflow.add_edge("orchestrator_node", "research_agent_node")
workflow.add_edge("orchestrator_node", "github_agent_node")
workflow.add_edge("orchestrator_node", "web_agent_node")

# Parallel agents fan in to the conditional router
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

# Proceed through validation and output
workflow.add_edge("conflict_detect_node", "feasibility_sanity_check_node")
workflow.add_edge("feasibility_sanity_check_node", "prd_agent_node")

workflow.add_edge("prd_agent_node", "notion_exporter_node")
workflow.add_edge("notion_exporter_node", END)

# Compile Graph
app = workflow.compile()

# ==========================================
# 4. EXECUTION HELPERS
# ==========================================

def run_pipeline(problem_statement: str):
    for collection in COLLECTIONS.values():
        collection.delete(where={})

    initial_state = {
        "ps": problem_statement,
        "research_queries": [],
        "github_search_terms": [],
        "web_search_queries": [],
        "prd_version": 0,
        "confidence_scores": {},
        "conflicts": [],
        "user_docs_context": ""
    }
    
    for event in app.stream(initial_state, {"recursion_limit": 10}):
        for key, value in event.items():
            print(f"✅ Finished step: {key}")
            
    return event