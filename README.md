# Research Paper RAG Pipeline

**Made by Atrijo Pal**

Local, fully-offline RAG pipeline for research paper analysis, built for an agentic AI project.

## Stack

| Layer | Choice | Notes |
|---|---|---|
| Embeddings | `BAAI/bge-large-en-v1.5` via **FastEmbed** | 1024-dim, ONNX-optimised, ~600 MB, free |
| Vector DB | **Qdrant** (local persistent) | Data survives restarts, stored in `./qdrant_db/` |
| LLM | **Ollama** (local) | Default: `llama3.2` — swap freely |
| PDF parsing | **PyMuPDF** (`fitz`) | Fast, preserves layout |

## Chunking Strategy

Research papers have predictable structure. The pipeline exploits it in two stages:

1. **Section detection** — regex matches numbered headings (`2.1 Methods`), ALL-CAPS headers (`ABSTRACT`), and 20+ known section keywords (Introduction, Methodology, Results, etc.) across ACM / IEEE / arXiv / NeurIPS paper styles.

2. **Sliding-window token chunking within each section** — target 400 tokens, 80-token overlap. Each chunk carries metadata: `paper_title`, `paper_id`, `section`, `pages`, `chunk_index`.

This means you can filter retrieval by section (e.g. only search `methodology` chunks) and the LLM always gets context that is semantically coherent.

## Agentic Features

- **Multi-query decomposition** — complex questions are split into 2-3 focused sub-queries by the LLM, retrieved independently, deduplicated, re-ranked by cosine score, then synthesised into one answer.
- **`ResearchRAG` class** — callable tool wrapper that drops into any agent framework (LangGraph, custom loops, etc.).

## Setup

```bash
# 1. Install Python deps
pip install -r requirements.txt

# 2. Pull an Ollama model (once)
ollama pull llama3.2        # or: mistral, phi4, deepseek-r1, gemma3 …

# 3. Start Ollama (keep running in a separate terminal)
ollama serve

# 4. Open the notebook
jupyter notebook rag_pipeline.ipynb
```

## Quick Start

```python
from rag_pipeline import ResearchRAG   # or run cells top-to-bottom

rag = ResearchRAG()

# Index papers
rag.add_paper("papers/attention_is_all_you_need.pdf")
rag.add_folder("papers/")              # index a whole folder

# Query (multi-query agentic mode by default)
answer = rag("What attention mechanism is proposed and what are its advantages?")
print(answer)

# Single-query, section-filtered
answer = rag(
    "Describe the training procedure.",
    multi_query=False,
    section="methodology",
)

# Raw retrieval (for debugging / custom re-ranking)
chunks = rag.search("scaled dot-product attention")
```

## Changing the LLM

Edit `OLLAMA_MODEL` in the config cell:

```python
OLLAMA_MODEL = "mistral"      # fast, 7B
OLLAMA_MODEL = "phi4"         # Microsoft, strong reasoning
OLLAMA_MODEL = "deepseek-r1"  # strong on technical/research content
OLLAMA_MODEL = "llama3.3"     # 70B if you have the VRAM
```

## Project Structure

```
.
├── rag_pipeline.ipynb   # main notebook
├── requirements.txt
├── README.md
├── papers/              # drop your PDFs here
└── qdrant_db/           # auto-created on first run (local vector store)
```
