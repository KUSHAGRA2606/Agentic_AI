# Agentic Framework for Automated Research Synthesis
**IITG.AI Project - Phase 2 Prototype**

## Overview
This repository contains the Week 2 prototype for the Agentic Framework project. Building upon the RAG system developed in Week 1, this phase implements the core Agentic Logic Loop using `LangGraph`.

## Architecture: The "Hive-Mind"
The system is modeled as a state machine with three specialized agents:
1. **The Orchestrator:** Deconstructs the primary Problem Statement (PS) into specific, actionable research hypotheses.
2. **The Librarian:** Designed to interact with academic APIs (Semantic Scholar, ArXiv) to retrieve relevant literature based on the Orchestrator's sub-queries.
3. **The Critic:** Evaluates the retrieved papers based on mathematical and conceptual relevance to the core PS.

## Current Status
* **Implemented:** The `StateGraph` routing logic between the Orchestrator, Librarian, and Critic is fully functional. The data schema (`AgentState`) successfully passes the context down the pipeline.
* **Testing:** The current `prototype.ipynb` uses mock data to validate the routing loops and state management before consuming API credits.
* **Next Steps:** Integrate the live academic scraping pipeline (Phase 1) and implement the Cross-Encoder Re-Ranking methodology.
