# Agentic RAG — Automated Research Synthesis Pipeline

A complete agentic framework that takes a competition problem statement as input and produces a grounded, cited literature review as output — by autonomously discovering, filtering, indexing, and synthesizing research papers.

## End-to-End Pipeline

```
INPUT: Competition Problem Statement (text)
                    │
    ┌───────────────▼───────────────┐
    │     ORCHESTRATOR (70B LLM)    │  Decomposes PS into 3-5 research
    │     Cold-start, no corpus     │  hypotheses with search queries
    └───────────────┬───────────────┘
                    │
    ┌───────────────▼───────────────┐
    │     LIBRARIAN                 │  Phase 1: arXiv + Semantic Scholar
    │     arXiv API + S2 API        │  Phase 2: Critic filters (62% kill rate)
    │     + citation chase          │  Phase 3: Citation chase on survivors
    │     + disk caching            │  Phase 4: Critic filters chase results
    └───────────────┬───────────────┘
                    │ ~22 papers survive from ~60 candidates
    ┌───────────────▼───────────────┐
    │     INGEST                    │  marker-pdf → manifest (SHA-256 dedup)
    │     marker → chunks → embed   │  → structure-aware chunks (440 tok)
    │     → ChromaDB + BM25         │  → BGE embeddings → ChromaDB + BM25
    └───────────────┬───────────────┘
                    │ corpus: ~500+ chunks from ~22 papers
    ┌───────────────▼───────────────┐
    │     AGENTIC LOOP (per hyp.)   │  route → plan → retrieve (hybrid+rerank)
    │     LangGraph StateGraph      │  → grade (70B) → reformulate (×3 max)
    │     with bounded retry        │  → generate with citations
    └───────────────┬───────────────┘
                    │
    ┌───────────────▼───────────────┐
    │     VERIFICATION              │  Checks every cited claim against
    │     Entailment check (70B)    │  its source chunk
    └───────────────┬───────────────┘
                    │
    ┌───────────────▼───────────────┐
    │     SYNTHESIS                 │  Map: one section per hypothesis
    │     Map-reduce assembly       │  Reduce: assemble + bibliography
    │     + honest gaps appendix    │  + gaps appendix for uncovered topics
    └───────────────┬───────────────┘
                    │
OUTPUT: Cited literature review with per-section sources
```

## Components Built

### Orchestrator
- Decomposes a competition PS into 3-5 research hypotheses with targeted search queries
- Domain-anchored prompting with good/bad query examples to avoid generic searches
- Cold-start LLM call (70B) — RAG can't help here because the corpus doesn't exist yet

### Librarian (Paper Acquisition)
- **Dual-source search**: arXiv API + Semantic Scholar Graph API per hypothesis
- **Recursive citation chase**: follows references (backward → foundations) and citations (forward → SOTA) of surviving papers, pruned by the Critic
- **Rate limiting**: token bucket + exponential backoff for both APIs
- **Disk caching**: every API response cached to `store/cache/s2/` with negative caching for failures — crashed sessions never re-spend API budget
- **Title-level dedup** at acquisition, **content-hash dedup** at ingestion

### Critic (Paper-Level Relevance Filter)
- **Two-signal scoring**: fast cross-encoder pass + LLM verdict for borderline cases
- Cross-encoder alone handles clear accepts/rejects; LLM only called in the gray zone (-2 to +2 score range)
- Strict relevance prompt: "papers about unrelated domains that happen to share terminology are NOT relevant"
- Runs twice: after keyword search (before citation chase) and after citation chase
- Typical kill rate: 60-65% of candidates filtered

### Ingestion & Indexing
- **marker-pdf** for structure-preserving PDF → markdown (headers, tables, equations)
- **Content-hash manifest** (SHA-256) — idempotent ingestion, parse-failure quarantine, embedder-version guard
- **Structure-aware chunker** — split on markdown headers, pack to 440-token budget with 60-token overlap, contextual header prepending, image/span stripping, sentence-level hard-split fallback for oversized table blobs
- **Markdown caching** — parsed output cached so re-chunking cycles take seconds
- **BGE-base-en-v1.5** embeddings with asymmetric query prefix
- **ChromaDB** (persistent) + **BM25 sidecar** with RRF fusion (K=60)

### Agentic Loop (LangGraph StateGraph)
- **Router**: classifies input as `rag` (needs corpus) or `direct` (greeting/chat)
- **Planner**: decomposes question into 1-3 standalone search queries
- **Retriever**: hybrid dense+BM25 → RRF fusion → cross-encoder rerank, per sub-query, merged and deduplicated
- **Grader (70B)**: judges the retrieved chunk SET — `sufficient` / `insufficient` / `irrelevant` with a specific `missing` diagnosis
- **Reformulator**: rewrites query using different vocabulary, conditioned on the grader's diagnosis and rewrite history, with mechanical widening (k increases per retry)
- **Hard retry cap**: 3 retries maximum, then calibrated refusal
- **Generator**: citation-enforced prompt (`[1]`, `[2]` markers), 70B
- **Calibrated refusal**: on exhaustion, explicitly states what the corpus lacks instead of hallucinating

### Verification Pass
- Checks every cited claim in the generated answer against its source chunk
- Entailment check via 70B: "supported" only if the passage actually states it
- Returns `all_supported: true/false` with list of unsupported claims

### Synthesis Mode
- **Map**: runs the full agentic loop independently per hypothesis (with verification per section)
- **Reduce**: assembles sections into a report with deduplicated bibliography
- **Honest gaps appendix**: sub-problems that exhausted retries listed as "not covered by discovered literature"

### MCP Integration
- Custom **arXiv MCP server** (`arxiv_server.py`) with `search_papers` tool via FastMCP + stdio transport
- **LangGraph ReAct agent** (70B) discovers and calls the tool at runtime
- Demonstrated working end-to-end; positioned at the tools node (exhausted-retries fallback edge) in the architecture
- Tools node code ready; not wired into sync graph due to async constraint (documented)

### Evaluation Framework
- **Labeled eval set**: 9 questions (6 answerable with ground-truth chunk IDs, 3 unanswerable)
- **Metrics**: hit@5 and MRR with per-question miss diagnostics
- **Ablation methodology**: one change per row, baseline measured before any upgrade

## Retrieval Ablation Results

132 chunks, 9 labeled questions, 2 papers (CRAG + Self-RAG).

| Retriever | hit@5 | MRR |
|---|---|---|
| dense (BGE + struct chunks) | 0.889 | 0.587 |
| hybrid, broken tokenizer | 0.667 | 0.593 |
| **hybrid, fixed tokenizer** | **0.778** | **0.722** |
| hybrid + reranker-base | **0.889** | 0.491 |
| dense + reranker-base | **0.889** | 0.593 |
| hybrid + reranker-v2-m3 | 0.778 | 0.417 |

**Decision**: rerank for set coverage (hit@5 0.889 → grader input), fixed hybrid for rank-sensitive use. Re-evaluate rerankers when corpus grows.

## Grader (Critique Agent) Results

| Case | Expected | 8B | 70B |
|---|---|---|---|
| Answerable (×4) | sufficient/insufficient | 4/4 ✓ | — |
| RAPTOR (unanswerable) | irrelevant | sufficient ✗ | irrelevant ✓ |
| GraphRAG (unanswerable) | irrelevant | insufficient ✗ | irrelevant ✓ |

**Finding**: 8B mistakes topic similarity for answer presence. Grading moved to 70B tier.

## Pitfalls Discovered (First-Hand)

1. **8B hallucinated `brave_search`** — called a nonexistent tool from training priors instead of the available `search_papers`
2. **70B fumbled tool-call syntax** — malformed XML; fixed with temperature=0 + system prompt + retry
3. **marker embeds `<span>` anchors in headers** → duplicate chunk IDs; fixed with regex stripping
4. **BM25 tokenizer must strip punctuation** — silent 0.22 hit@5 loss otherwise
5. **Cross-encoder rerankers favor abstracts over answer chunks** at small corpus scale — both base and v2-m3 demoted method chunks below intros
6. **`embed_text` must fit 512 tokens** — chunk (440) + contextual header (~25) combined
7. **8B grader mistakes topic for answer** — grades RAPTOR question as "sufficient" from RAG-adjacent chunks
8. **Orchestrator query quality determines everything downstream** — generic queries ("NLP techniques") → 69% irrelevant papers; specific queries ("retrieval augmented generation survey") → relevant corpus
9. **S2 API returns `null` not `[]` for empty results** — `.get("data", [])` doesn't catch it; need `or []`
10. **Citation chase amplifies bad seeds** — chasing irrelevant Phase 1 papers produces more irrelevant papers; Critic must run BETWEEN phases

## Files

| File | Description |
|---|---|
| `agentic.ipynb` | Main notebook — full pipeline with all outputs |
| `arxiv_server.py` | MCP server for arXiv search |
| `agentic_rag_methods.tex` | 28-page technical methods document |
| `agentic_rag_methods.pdf` | Compiled PDF |

## Stack

| Component | Choice |
|---|---|
| Parsing | marker-pdf |
| Embeddings | BAAI/bge-base-en-v1.5 |
| Vector DB | ChromaDB (persistent) + BM25 (rank_bm25) |
| Reranker | BAAI/bge-reranker-base |
| LLM (control) | Groq — Llama-3.1-8B-instant |
| LLM (grading/generation) | Groq — Llama-3.3-70B-versatile |
| Orchestration | LangGraph StateGraph |
| Paper APIs | arXiv + Semantic Scholar Graph API |
| MCP | Python SDK, FastMCP, stdio transport |
| Eval | Custom hit@k/MRR harness |

## Architecture Decisions

| Decision | Rationale |
|---|---|
| 70B for grading, 8B for routing | 8B fails unanswerable detection (4/6 vs 6/6) |
| Hybrid for ranking, rerank for pool | Reranker has abstract-bias at small corpus scale |
| Direct API for Librarian, MCP for tools node | Librarian needs rate-limit control; tools node needs runtime tool choice |
| Critic between phases, not just at the end | Prevents citation chase from amplifying irrelevant seeds |
| NetworkX not Neo4j for future graph | Same capability at this scale, zero infrastructure overhead |
| Calibrated refusal over hallucination | 3-retry cap + explicit "corpus lacks X" message |
