import os
import json
import arxiv
import chromadb
import operator
from typing import Annotated, List, Dict, Any, TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langgraph.graph import StateGraph, END
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from github import Github, Auth
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone

def merge_dicts(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
    return {**a, **b}

load_dotenv()
auth = Auth.Token(os.getenv("GITHUB_TOKEN"))
g = Github(auth=auth)
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# RAG Setup
chroma_client = chromadb.Client()
papers_collection = chroma_client.get_or_create_collection(name="papers")
repos_collection = chroma_client.get_or_create_collection(name="repos")
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

# State Definition
class AgentState(TypedDict):
    ps: str
    research_queries: List[str]
    github_queries: List[str]
    research_report: Dict[str, Any]
    github_report: Dict[str, Any]
    conflicts: List[str]
    confidence_scores: Annotated[Dict[str, float], merge_dicts]
    iterations: int

# Calculate Agreement Score
def calculate_mathematical_agreement(papers: List[Any], embedder: Any) -> float:
    """Calculates Agreement Score using Cosine Similarity."""
    if not papers:
        return 0.0
    embeddings_list = np.array([embedder.embed_query(p.summary) for p in papers])
    sim_matrix = cosine_similarity(embeddings_list)
    threshold = 0.85
    consensus_counts = [np.sum(sim_matrix[i] > threshold) for i in range(len(papers))]
    return max(consensus_counts) / len(papers) if papers else 0.0

# MCP Tools
def arxiv_mcp_tool(queries: List[str], ps: str) -> Dict[str, Any]:
    """Fetches papers and calculates consensus."""
    client = arxiv.Client()
    papers = []
    for q in queries:
        try:
            papers.extend(list(client.results(arxiv.Search(query=q, max_results=5))))
        except Exception as e:
            print(f"ArXiv search failed for '{q}': {e}")
            
    as_score = calculate_mathematical_agreement(papers, embeddings) 
    np_count = len(papers)
    n_recent = len([p for p in papers if p.published.year >= 2024])
    
    confidence = (0.4 * min(np_count/10, 1)) + (0.3 * (n_recent/np_count if np_count > 0 else 0)) + (0.3 * as_score)

    papers_list = []
    for p in papers:
        papers_collection.upsert(documents=[p.summary], ids=[p.entry_id], 
                                 metadatas={"title": p.title, "year": p.published.year})
        papers_list.append({"title": p.title, "url": p.entry_id, "year": p.published.year})
    
    return {
        "confidence": min(confidence, 1.0), 
        "papers_found": np_count,
        "fetched_items": papers_list
    }

def github_mcp_tool(queries: List[str]) -> Dict[str, Any]:
    all_repos = []
    for query in queries:
        print(f"DEBUG: GitHub Querying: {query}")
        try:
            repos = g.search_repositories(query=query, sort='stars', order='desc')
            if repos.totalCount > 0:
                for repo in repos[:5]:
                    all_repos.append(repo)
        except Exception as e:
            print(f"GitHub search failed for '{query}': {e}")
            continue
    
    unique_repos = {r.full_name: r for r in all_repos}.values()
    
    repos_list = []
    for repo in unique_repos:
        repos_collection.upsert(
            documents=[repo.description or ""],
            metadatas={
                "name": repo.name,
                "stars": repo.stargazers_count,
                "url": repo.html_url,
                "source_type": "repo"
            },
            ids=[str(repo.id)]
        )
        repos_list.append({"name": repo.full_name, "url": repo.html_url, "stars": repo.stargazers_count})
    
    nr = len(unique_repos)
    if nr == 0:
        return {"status": "No repos found", "confidence": 0.0, "fetched_items": []}

    # 1 year ago calculation
    one_year_ago = datetime.now(timezone.utc) - timedelta(days=365)
    n_active = len([r for r in unique_repos if r.pushed_at and r.pushed_at > one_year_ago])

    avg_stars = sum(r['stars'] for r in repos_list) / nr if nr > 0 else 0
    star_score = min(avg_stars / 1000, 1)  
    confidence = (0.5 * min(nr / 3, 1)) + (0.3 * (n_active / nr if nr > 0 else 0)) + (0.2 * star_score)
    
    return {"confidence": max(confidence, 0.2), "fetched_items": repos_list}

# Node definitions
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.2)

def orchestrator_node(state: AgentState) -> Dict[str, Any]:
    print("[Orchestrator]: Planning research strategy...")
    prompt = ChatPromptTemplate.from_template(
        "Analyze this hackathon problem statement: '{ps}'\n"
        "Generate a JSON object with 2 targeted academic 'research_queries' and 2 'github_queries'.\n"
        "FOR GITHUB: Generate simple queries using language and topic qualifiers only.\n"
        "Example: 'language:python trading' or 'language:cpp high-frequency-trading'\n"
        "Format: {{\"research_queries\": [\"...\"], \"github_queries\": [\"...\"]}}"
    )
    chain = prompt | llm | JsonOutputParser()
    res = chain.invoke({"ps": state["ps"]})
    
    return {
        "research_queries": res.get("research_queries", []),
        "github_queries": res.get("github_queries", []),
        "iterations": state.get("iterations", 0) + 1
    }

def research_agent_node(state: AgentState) -> Dict[str, Any]:
    print("[Research Agent]: Querying ArXiv...")
    mcp_result = arxiv_mcp_tool(state["research_queries"], state["ps"])
    return {
        "research_report": {"status": "Complete", "papers": mcp_result["fetched_items"]},
        "confidence_scores": {"research": mcp_result["confidence"]}
    }

def github_agent_node(state: AgentState) -> Dict[str, Any]:
    print("[GitHub Agent]: Scanning repositories and updating RAG...")
    mcp_result = github_mcp_tool(state["github_queries"])
    return {
        "github_report": {"status": "Complete", "repos": mcp_result["fetched_items"]},
        "confidence_scores": {"github": mcp_result["confidence"]}
    }

def critic_node(state: AgentState) -> Dict[str, Any]:
    print("[Critique Agent]: Evaluating sub-agent findings for contradictions...")
    res_conf = state["confidence_scores"].get("research", 0)
    git_conf = state["confidence_scores"].get("github", 0)
    
    conflicts = []
    if res_conf < 0.4:
        conflicts.append("Research confidence too low. Need better academic grounding.")
    if git_conf < 0.5:
        conflicts.append("GitHub repository evidence is lacking or outdated.")
        
    print(f"  Research Confidence: {res_conf:.2f} | GitHub Confidence: {git_conf:.2f}")
    return {"conflicts": conflicts}

def critic_router(state: AgentState) -> str:
    if state["conflicts"] and state["iterations"] < 3:
        print("[System]: Conflicts detected. Routing back to Orchestrator to refine queries.")
        return "retry"
    print("[System]: Feasibility checks passed. Ready for PRD Generation.")
    return "end"

# Graph Configuration
workflow = StateGraph(AgentState)
workflow.add_node("orchestrator", orchestrator_node)
workflow.add_node("research_agent", research_agent_node)
workflow.add_node("github_agent", github_agent_node)
workflow.add_node("critic", critic_node)

workflow.set_entry_point("orchestrator")
workflow.add_edge("orchestrator", "research_agent")
workflow.add_edge("orchestrator", "github_agent")
workflow.add_edge("research_agent", "critic")
workflow.add_edge("github_agent", "critic")
workflow.add_conditional_edges("critic", critic_router, {"retry": "orchestrator", "end": END})

app = workflow.compile()

if __name__ == "__main__":
    ps = "Project Overview: Real-Time AI Trading SimulatorThe objective is to build a " \
    "deep learning-powered system that simulates real-time trading using live market data " \
    "(e.g., Dogecoin). The project focuses on asset classes that move independently of broad " \
    "market conditions to purely test model performance.  Core TasksLearn Concepts: Study " \
    "Neural Networks, RNNs, Attention mechanisms, AutoEncoders, and basic trading " \
    "terminologies/strategies (e.g., Kelly criterion).  Research & Develop: Review financial " \
    "ML papers and develop at least 2 Deep Learning models in PyTorch to predict either " \
    "log-returns or volatility.  Strategy (Stretch Goal): Build a trading strategy using your " \
    "models' predictions to manage a fixed amount of initial capital.  Target PipelineData " \
    "Ingestion: Real-time prices are pulled and routed through a high-speed C++ queue.  " \
    "Forecasting & Execution: Multiple DL models calculate metrics (log-returns/volatility), " \
    "and the trading strategy uses these to size positions.  Logging: Trades are stored in a " \
    "database to monitor metrics like Profit and Loss (PnL).  Technical DeliverablesTraining " \
    "Notebook (.ipynb): A Jupyter notebook (preferably PyTorch) used to train the models and " \
    "save static weights.  Execution Script (.py): A python file containing an execute function.  " \
    "Inputs: Only the current asset price and your remaining capital. (You must internally track " \
    "derived metrics like RSI or rolling averages) .  Output: A dictionary \
    format: {'buy': X, 'sell': Y}, where X and Y are whole numbers."
    initial_state = {
        "ps": ps, "research_queries": [], "github_queries": [],
        "research_report": {}, "github_report": {}, "conflicts": [],
        "confidence_scores": {}, "iterations": 0
    }
    
    final_state = app.invoke(initial_state)

    
    print("\nGATHERED RESEARCH PAPERS:")
    for paper in final_state["research_report"].get("papers", []):
        print(f" - [{paper['year']}] {paper['title']} -> {paper['url']}")
        
    print("\nGATHERED REPOSITORIES:")
    for repo in final_state["github_report"].get("repos", []):
        print(f" - {repo['name']} (Stars: {repo['stars']}) -> {repo['url']}")