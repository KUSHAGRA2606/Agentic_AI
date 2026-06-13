# Agentic RAG — Week 2

## What I Built

An agentic RAG pipeline over research papers, starting from a naive retrieve-and-generate notebook and upgrading every component with measured results.

```
PDF → marker-pdf → manifest (SHA-256 dedup)
    → structure-aware chunks (440 tok, section-tagged)
    → BGE embeddings → ChromaDB + BM25 → RRF fusion → rerank
    → grader (sufficiency check before generation)
```

## Tasks Completed

**1. MCP** — Built an arXiv MCP server with a `search_papers` tool, connected via stdio, and wired a LangGraph ReAct agent (Groq 70B) that chooses the tool at runtime.

**2. Critique Agent** — Prototyped a set-level grader that judges whether retrieved chunks actually answer the question. Tested 8B vs 70B: 8B scores 4/6 (fails unanswerable questions — mistakes topic for answer), 70B scores 6/6. Finding: grading needs 70B.

**3. RAG Implementation** — Structure-aware chunking, BGE embeddings with query prefix, hybrid dense+BM25 retrieval with RRF, cross-encoder reranking. All measured with a labeled eval set (9 questions, ground-truth chunk IDs).

**4. Retrieval Ablation**

| Retriever | hit@5 | MRR |
|---|---|---|
| dense (BGE) | 0.889 | 0.587 |
| hybrid (fixed tokenizer) | 0.778 | **0.722** |
| hybrid + reranker | **0.889** | 0.491 |

Rerankers improve recall into top-5 but demote answer chunks below abstracts on this corpus size. Decision: rerank for set coverage, hybrid for ranking.

## Files

| File | Description |
|---|---|
| `agentic.ipynb` | Main notebook — pipeline + eval + MCP demo + grader |
| `arxiv_server.py` | MCP server for arXiv search |

## Stack

marker-pdf · BGE-base-en-v1.5 · ChromaDB · rank_bm25 · bge-reranker-base · Groq (Llama 8B/70B) · LangGraph · MCP
