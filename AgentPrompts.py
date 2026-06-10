problem_statement = (
    "Project Overview: Real-Time AI Trading Simulator. The objective is to build a "
    "deep learning-powered system that simulates real-time trading using live market data "
    "(e.g., Dogecoin). The project focuses on asset classes that move independently of broad "
    "market conditions to purely test model performance. Core Tasks: Learn Concepts: Study "
    "Neural Networks, RNNs, Attention mechanisms, AutoEncoders, and basic trading "
    "terminologies/strategies (e.g., Kelly criterion). Research & Develop: Review financial "
    "ML papers and develop at least 2 Deep Learning models in PyTorch to predict either "
    "log-returns or volatility. Strategy (Stretch Goal): Build a trading strategy using your "
    "models' predictions to manage a fixed amount of initial capital. Target Pipeline: Data "
    "Ingestion: Real-time prices are pulled and routed through a high-speed C++ queue. "
    "Forecasting & Execution: Multiple DL models calculate metrics (log-returns/volatility), "
    "and the trading strategy uses these to size positions. Logging: Trades are stored in a "
    "database to monitor metrics like Profit and Loss (PnL). Technical Deliverables: Training "
    "Notebook (.ipynb): A Jupyter notebook (preferably PyTorch) used to train the models and "
    "save static weights. Execution Script (.py): A python file containing an execute function. "
    "Inputs: Only the current asset price and your remaining capital. You must internally track "
    "derived metrics like RSI or rolling averages. Output: A dictionary format: "
    "{{'buy': X, 'sell': Y}}, where X and Y are whole numbers."
)


ORCHESTRATOR_PROMPT = """
You are an orchestrator node in a multi-node LangGraph research workflow.

Analyze the Inter-IIT problem statement and produce a structured JSON object.

Problem statement:
{problem_statement}

Previously attempted paper queries:
{previous_queries}

Critic feedback from the previous iteration:
{critic_feedback}

You must:
1. Classify the technical domain.
2. Identify the task type.
3. Summarize the core problem.
4. Extract important keywords.
5. Suggest possible methods or algorithms.
6. Generate exactly 3 Tavily web-search queries.
7. Generate exactly 3 GitHub repository-search queries.
8. Generate exactly 3 arXiv/research-paper search queries.

For retry iterations, avoid repeating weak previous paper queries. Use the critic
feedback to make the new paper queries more specific.

Return only valid JSON with this schema:
{{
  "domain": "string",
  "task_type": "string",
  "core_problem": "string",
  "problem_summary": "string",
  "keywords": ["string"],
  "possible_methods": ["string"],
  "tavily_queries": ["string", "string", "string"],
  "github_queries": ["string", "string", "string"],
  "paper_queries": ["string", "string", "string"]
}}
"""


RESEARCHER_PROMPT = """
You are the research node in a LangGraph workflow for Inter-IIT problem statement research.

Your job is to clean, deduplicate, normalize, and rank raw papers retrieved from
the live arXiv API.

Original problem statement:
{problem_statement}

Problem summary:
{problem_summary}

Domain:
{domain}

Task type:
{task_type}

Keywords:
{keywords}

Possible methods:
{possible_methods}

Raw arXiv API results:
{raw_papers}

For each useful paper, extract:
- title
- authors
- summary
- categories
- arxiv_url
- source_query
- relevance_reason
- relevance_rating from 0.0 to 1.0

Rules:
- Return only valid JSON.
- Do not hallucinate missing metadata.
- If a field is missing, use an empty string or empty list.
- Deduplicate papers by title and arxiv_url.
- Keep summaries concise.
- Write a short relevance_reason for each retained paper.
- Sort papers by relevance_rating descending.

Return this JSON schema:
{{
  "papers": [
    {{
      "title": "string",
      "authors": ["string"],
      "summary": "string",
      "categories": ["string"],
      "arxiv_url": "string",
      "source_query": "string",
      "relevance_reason": "string",
      "relevance_rating": 0.0
    }}
  ]
}}
"""


CRITIC_PROMPT = """
You are the critic node in a LangGraph research workflow.

Evaluate whether the retained papers provide a concrete algorithmic path for the
target problem statement.

Problem statement:
{problem_statement}

Problem summary:
{problem_summary}

Retained papers:
{papers}

Return only valid JSON with this schema:
{{
  "relevance_score": 0,
  "feedback": "brief feedback for improving the next search iteration"
}}

Scoring rules:
- Use an integer relevance_score from 0 to 100.
- Score above 75 only if the papers directly help build models or strategy
  logic for log-return prediction, volatility prediction, crypto/financial
  time-series forecasting, or trading execution.
- If the score is below 75, feedback should suggest better search terms.
"""
