**Unified Shared Memory**: The entire system runs on a central dictionary (`AgentState`) that acts as a shared scratchpad, allowing parallel agents to read data and write verification reports without losing track of the problem statement.

**Intelligent Query Generation**: The Orchestrator acts as the system's brain, stripping unnecessary natural language words from the problem statement to create simple, target-specific keyword searches for external databases.

**Real-Time Academic Search**: The Research Agent calls the live ArXiv API automatically to fetch the latest relevant papers based on the Orchestrator's keywords.

**Live Ecosystem Matching**: The GitHub Agent queries the authentic GitHub API using the token to find actual codebases, tracking stars and true development activity.

**Automated RAG Core**: Every retrieved paper summary and repository description is immediately converted into numerical embeddings and indexed into multi-collection vector databases (`papers` and `repos` in ChromaDB).

**Mathematical Consensus Score**: Instead of relying on subjective LLM opinions, the system computes a strict mathematical agreement score ($A_s$) by using pairwise Cosine Similarity vectors to find matching technical clusters across paper abstracts.

**Parallel State Merger**: A custom dictionary reducer (`merge_dicts`) prevents the state graph from crashing when the parallel Research and GitHub agents return their data to the central scratchpad at the exact same time.

**Strict Gatekeeping**: The Critique Agent reads the unified confidence metrics from the scratchpad. If the research grounding or repository activity scores fall below the target limits, it blocks the flow and forces a query retry.