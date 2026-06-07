import os
from typing import List, Dict, Any, TypedDict
import arxiv 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.2)

class AgentHiveState(TypedDict):
    problem_statement: str
    research_queries: List[str]
    fetched_papers: List[Dict[str, Any]]
    evaluation_report: Dict[str, Any]
    iterations: int

# Analyses the PS and generates phrases to be queried to find research papers
def orchestrator_node(state: AgentHiveState) -> Dict[str, Any]:
    """Deconstructs the complex PS into targeted research sub-problems/queries."""
    print("Orchestrator running")
    
    prompt = ChatPromptTemplate.from_template(
        "You are the Principal Investigator. Analyze this competitive problem statement:\n"
        "'{ps}'\n\n"
        "Generate a JSON object containing a list of exactly 2 or 3 highly targeted, single-topic "
        "academic search phrases (e.g., 'Graph Retrieval-Augmented Generation' or 'PDF formula extraction'). "
        "Keep them focused on solving the core technical bottleneck.\n"
        "Format: {{\"queries\": [\"query 1\", \"query 2\"]}}"
    )
    
    chain = prompt | llm | JsonOutputParser()
    response = chain.invoke({"ps": state["problem_statement"]})
    
    return {
        "research_queries": response.get("queries", []),
        "iterations": state.get("iterations", 0) + 1
    }

# Uses the generated phrases to query arxiv library for research papers
def librarian_node(state: AgentHiveState) -> Dict[str, Any]:
    """Queries the live arXiv API directly using the generated research phrases."""
    queries = state["research_queries"]
    print(f"Librarian fetching papers")
    
    retrieved_papers = []
    seen_titles = set()
    
    client = arxiv.Client()
    
    for query in queries:
        try:
            search = arxiv.Search(
                query=query,
                max_results=3,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            results = client.results(search)
            
            for result in results:
                if result.title not in seen_titles:
                    seen_titles.add(result.title)
                    retrieved_papers.append({
                        "title": result.title,
                        "abstract": result.summary.replace("\n", " "),
                        "url": result.entry_id
                    })
        except Exception as e:
            print(f"Error querying arXiv for '{query}': {e}")
            
    print(f"Successfully gathered {len(retrieved_papers)} real-world papers from arXiv.")
    return {"fetched_papers": retrieved_papers}

# Evaluates the papers and generates a relevance score. If the score is too low, the agent is 
# routed back to the orchestrator to generate new keywords, and hence get different and 
# potentially better papers
# One imporvement : we give the orchestrator the phrases it generated in the previous 
# iteration, so it does not repeat them
def critic_node(state: AgentHiveState) -> Dict[str, Any]:
    """Evaluates paper quality and strictly calculates a relevance score relative to the PS."""
    print("Critic running")
    
    if not state["fetched_papers"]:
        return {
            "evaluation_report": {
                "relevance_score": 0,
                "feedback": "No papers were retrieved by the Librarian."
            }
        }
        
    papers_summary = ""
    for idx, paper in enumerate(state["fetched_papers"][:6]): 
        papers_summary += f"\n[{idx}] Title: {paper['title']}\nAbstract: {paper['abstract']}\nURL: {paper['url']}\n"
        
    prompt = ChatPromptTemplate.from_template(
        "You are the Peer-Review Critic. Evaluate these actual research papers against our target problem statement.\n"
        "Remember that we value algorithmic/methodological relevance to our problem statement above publication date or venue.\n\n"
        "Target Problem Statement: {ps}\n\n"
        "Papers to Evaluate:\n{papers}\n\n"
        "Provide your evaluation strictly as a JSON object with two fields:\n"
        "1. 'relevance_score': an integer from 1 to 100 assessing if these papers provide a concrete algorithmic path for our specific PS.\n"
        "2. 'feedback': A brief sentence on what is missing or what terms to target next if the score is under 75.\n"
        "Format: {{\"relevance_score\": 85, \"feedback\": \"...\"}}"
    )
    
    chain = prompt | llm | JsonOutputParser()
    evaluation = chain.invoke({"ps": state["problem_statement"], "papers": papers_summary})
    
    print(f"Score Awarded: {evaluation.get('relevance_score')}/100")
    return {"evaluation_report": evaluation}

# Decides whether to end or to reiterate
def should_continue(state: AgentHiveState) -> str:
    """Conditional router determining if we need to loop back or terminate."""
    report = state.get("evaluation_report", {})
    score = report.get("relevance_score", 0)
    iterations = state.get("iterations", 0)
    
    if score >= 75 or iterations >= 3:
        if iterations >= 3 and score < 75:
            print("Too many iterations without clearing threshold")
        print("Done")
        return "end"
    else:
        print(f"[System]: Relevance score ({score}) too low. Feedback: {report.get('feedback')}")
        print("Routing back to orchestrator")
        return "retry"


workflow = StateGraph(AgentHiveState)

workflow.add_node("orchestrator", orchestrator_node)
workflow.add_node("librarian", librarian_node)
workflow.add_node("critic", critic_node)

workflow.set_entry_point("orchestrator")

workflow.add_edge("orchestrator", "librarian")
workflow.add_edge("librarian", "critic")

workflow.add_conditional_edges(
    "critic",
    should_continue,
    {
        "retry": "orchestrator",
        "end": END
    }
)

app = workflow.compile()

# Using the time series ps from iitg.ai projects
if __name__ == "__main__":
    sample_ps = "Project Overview: Real-Time AI Trading SimulatorThe objective is to build a " \
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
    
    print("Starting agentic pipeline")
    initial_state = {
        "problem_statement": sample_ps,
        "research_queries": [],
        "fetched_papers": [],
        "evaluation_report": {},
        "iterations": 0
    }
    
    final_output = app.invoke(initial_state)
    
    print("\nFinal Output")
    print(f"Total Logic Loop Iterations: {final_output['iterations']}")
    print(f"Final Critic Score: {final_output['evaluation_report']['relevance_score']}/100")
    print("\nLive Papers Gathered from arXiv:")
    for paper in final_output['fetched_papers']:
        print(f"\nTitle: {paper['title']}")
        print(f"Link: {paper['url']}")