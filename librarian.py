"""
Agent 2 — The Librarian (retrieval)
===================================

Baseline implementation of every Librarian tool, in the LangChain environment,
using the SAME open-source HuggingFace chat model and embeddings as Agent 1.

Connection to Agent 1
---------------------
This module imports `llm`, `embedder`, and `SubHypothesis` directly from
`orchestrator.py`, so both agents share one model instance and one schema. The
Librarian exposes ONE entry point the Orchestrator needs:

    librarian.retrieve(hypothesis) -> list[Document]     # ~50 candidate chunks

and a helper `build_evidence_provider(librarian)` that returns a callable with
exactly the signature Agent 1's loop expects:

    evidence_provider(SubHypothesis) -> (best_score, answer, citations)

(The scorer inside it is a TEMPORARY stand-in for Agent 3 — the Critic — so the
1+2 pipeline runs end-to-end today. Swap it for the real Critic later.)

Baseline vs production
----------------------
* Vector DB  : in-memory Qdrant   -> swap for a Qdrant server in prod.
* Knowledge graph : networkx       -> swap for Neo4j in prod.
* PDF -> Markdown : pdfplumber fallback -> swap for Nougat/Marker (LaTeX) in prod.

Requirements:
    pip install langchain langchain-community langchain-experimental \
                langchain-qdrant qdrant-client langchain-huggingface \
                networkx requests pdfplumber pydantic
    export HUGGINGFACEHUB_API_TOKEN=...
"""

from __future__ import annotations

import time
import hashlib
import tempfile
from pathlib import Path
from typing import Literal, Optional

import requests
import networkx as nx
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.tools import StructuredTool
from langchain_community.document_loaders import ArxivLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# --- shared with Agent 1 : SAME model, SAME embeddings, SAME schema ---------
from orchestrator import llm, embedder, SubHypothesis

from dotenv import load_dotenv
load_dotenv()
# ----------------------------------------------------------------------------- #
# Configuration
# ----------------------------------------------------------------------------- #

S2_BASE = "https://api.semanticscholar.org/graph/v1"
MAX_PER_SOURCE = 5        # papers pulled from each search source
MAX_NEIGHBOURS = 6        # references/citations kept per seed paper
N_CANDIDATES = 50         # chunks handed to the Critic
MAX_EDGE_CLASSIFY = 12    # cap LLM edge-classification calls per retrieve()
REQUEST_DELAY = 1.0       # seconds between rate-limited API calls
CACHE_DIR = Path(tempfile.gettempdir()) / "librarian_pdf_cache"
CACHE_DIR.mkdir(exist_ok=True)


# ----------------------------------------------------------------------------- #
# Rate-limited HTTP helper (handles the 429s the PS warns about)
# ----------------------------------------------------------------------------- #

def _get(url: str, params: dict | None = None, max_retries: int = 4) -> Optional[dict]:
    """GET with exponential backoff on HTTP 429 / transient errors."""
    delay = REQUEST_DELAY
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                time.sleep(delay)
                delay *= 2
                continue
            resp.raise_for_status()
            time.sleep(REQUEST_DELAY)      # be polite between successful calls
            return resp.json()
        except requests.RequestException:
            time.sleep(delay)
            delay *= 2
    return None


# ----------------------------------------------------------------------------- #
# Edge classifier schema (improves upon / contradicts / uses)
# ----------------------------------------------------------------------------- #

class EdgeLabel(BaseModel):
    relation: Literal["improves_upon", "contradicts", "uses", "cites"] = Field(
        description="relationship the citing paper has toward the cited paper")


_EDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Classify the relationship the citing paper expresses toward the cited "
     "work, based on the citation context. Answer with one relation only.\n\n"
     "{format_instructions}"),
    ("human", "Citation context:\n{context}"),
])


# ----------------------------------------------------------------------------- #
# The Librarian
# ----------------------------------------------------------------------------- #

class Librarian:
    """Agent 2. Finds, ingests, indexes, and retrieves papers for a hypothesis."""

    def __init__(self, llm=llm, embedder=embedder, collection: str = "papers") -> None:
        self.llm = llm
        self.embedder = embedder
        self.collection = collection

        # semantic chunker uses the SAME embeddings as everything else
        self.splitter = SemanticChunker(embedder)

        # in-memory vector store (swap location for a Qdrant URL in production)
        self._dim = len(embedder.embed_query("dimension probe"))
        self._client = QdrantClient(":memory:")
        self._client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=self._dim, distance=Distance.COSINE),
        )
        self.vstore = QdrantVectorStore(
            client=self._client, collection_name=collection, embedding=embedder)

        # in-memory knowledge graph (swap for Neo4j in production)
        self.graph = nx.DiGraph()

        # per-paper chunk index, used for graph-expansion retrieval
        self.chunks_by_paper: dict[str, list[Document]] = {}
        self.ingested: set[str] = set()

        # edge classifier chain
        parser = PydanticOutputParser(pydantic_object=EdgeLabel)
        self._edge_chain = (
            _EDGE_PROMPT.partial(format_instructions=parser.get_format_instructions())
            | self.llm | parser
        )
        self._edge_budget = 0

        self.tools = self._register_tools()

    # ----- Tool 1+2 : search -------------------------------------------------
    def search_arxiv(self, query: str, k: int = MAX_PER_SOURCE) -> list[dict]:
        """Keyword search of ArXiv -> lightweight paper records."""
        records = []
        try:
            docs = ArxivLoader(query=query, load_max_docs=k).load()
        except Exception:
            docs = []
        for d in docs:
            meta = d.metadata
            entry = meta.get("Entry ID", "") or meta.get("entry_id", "")
            pid = entry.rsplit("/", 1)[-1] if entry else meta.get("Title", "")[:40]
            records.append({
                "paper_id": f"arxiv:{pid}",
                "title": meta.get("Title", ""),
                "abstract": meta.get("Summary", "") or d.page_content[:1500],
                "pdf_url": entry.replace("abs", "pdf") + ".pdf" if entry else None,
            })
        return records

    def search_semantic_scholar(self, query: str, k: int = MAX_PER_SOURCE) -> list[dict]:
        """Keyword search of Semantic Scholar -> records (incl. paperId for crawl)."""
        data = _get(f"{S2_BASE}/paper/search", {
            "query": query, "limit": k,
            "fields": "title,abstract,externalIds,openAccessPdf",
        })
        records = []
        for p in (data or {}).get("data", []) or []:
            pdf = (p.get("openAccessPdf") or {}).get("url")
            records.append({
                "paper_id": f"s2:{p['paperId']}",
                "s2_id": p["paperId"],
                "title": p.get("title", ""),
                "abstract": p.get("abstract") or "",
                "pdf_url": pdf,
            })
        return records

    # ----- Tool 3+4 : recursive crawl ---------------------------------------
    def _crawl(self, s2_id: str, edge: str) -> list[dict]:
        """Shared backward/forward crawl. edge in {'references','citations'}."""
        data = _get(f"{S2_BASE}/paper/{s2_id}/{edge}", {
            "limit": MAX_NEIGHBOURS,
            "fields": "title,abstract,externalIds,openAccessPdf,contexts",
        })
        out = []
        for item in (data or {}).get("data", []) or []:
            p = item.get("citedPaper") or item.get("citingPaper") or {}
            if not p.get("paperId"):
                continue
            pdf = (p.get("openAccessPdf") or {}).get("url")
            out.append({
                "paper_id": f"s2:{p['paperId']}",
                "s2_id": p["paperId"],
                "title": p.get("title", ""),
                "abstract": p.get("abstract") or "",
                "pdf_url": pdf,
                "context": " ".join(item.get("contexts", []) or [])[:600],
            })
        return out

    def get_references(self, s2_id: str) -> list[dict]:
        """Backward crawl: works this paper cites."""
        return self._crawl(s2_id, "references")

    def get_citations(self, s2_id: str) -> list[dict]:
        """Forward crawl: newer papers that cite this one."""
        return self._crawl(s2_id, "citations")

    # ----- Tool 5 : fetch_pdf (rate-limited + cached) ------------------------
    def fetch_pdf(self, pdf_url: str) -> Optional[str]:
        """Download a PDF with caching. Returns local path or None."""
        if not pdf_url:
            return None
        key = hashlib.sha1(pdf_url.encode()).hexdigest()
        path = CACHE_DIR / f"{key}.pdf"
        if path.exists():
            return str(path)
        delay = REQUEST_DELAY
        for _ in range(3):
            try:
                r = requests.get(pdf_url, timeout=60)
                if r.status_code == 429:
                    time.sleep(delay); delay *= 2; continue
                r.raise_for_status()
                path.write_bytes(r.content)
                return str(path)
            except requests.RequestException:
                time.sleep(delay); delay *= 2
        return None

    # ----- Tool 6 : parse_to_markdown ---------------------------------------
    def parse_to_markdown(self, pdf_path: str) -> str:
        """PDF -> text. Baseline uses pdfplumber; prod should use Nougat/Marker
        to preserve LaTeX. Falls back gracefully if parsing fails."""
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                return "\n\n".join((pg.extract_text() or "") for pg in pdf.pages)
        except Exception:
            return ""

    # ----- Tool 7 : chunk_document ------------------------------------------
    def chunk_document(self, text: str, paper_id: str, title: str) -> list[Document]:
        """Semantic-split text into chunks carrying {paper_id, title} metadata."""
        if not text.strip():
            return []
        try:
            chunks = self.splitter.create_documents([text])
        except Exception:
            chunks = [Document(page_content=text[:2000])]
        for c in chunks:
            c.metadata.update({"paper_id": paper_id, "title": title})
        return chunks

    # ----- Tool 8 : classify_edge -------------------------------------------
    def classify_edge(self, context: str) -> str:
        """Label a citation edge. Budgeted; defaults to 'cites' when exhausted."""
        if not context or self._edge_budget >= MAX_EDGE_CLASSIFY:
            return "cites"
        self._edge_budget += 1
        try:
            return self._edge_chain.invoke({"context": context}).relation
        except Exception:
            return "cites"

    # ----- Tool 9 : upsert_graph --------------------------------------------
    def upsert_graph(self, src: str, dst: str, relation: str) -> None:
        self.graph.add_node(src)
        self.graph.add_node(dst)
        self.graph.add_edge(src, dst, relation=relation)

    # ----- ingestion (chains tools 5-9 for one paper) ------------------------
    def _ingest_record(self, rec: dict, full: bool) -> None:
        """Add one paper to the index. full=True downloads + parses the PDF;
        otherwise the abstract is indexed (cheap)."""
        pid = rec["paper_id"]
        if pid in self.ingested:
            return
        self.ingested.add(pid)

        chunks: list[Document] = []
        if full and rec.get("pdf_url"):
            local = self.fetch_pdf(rec["pdf_url"])
            if local:
                md = self.parse_to_markdown(local)
                chunks = self.chunk_document(md, pid, rec.get("title", ""))
        if not chunks and rec.get("abstract"):     # fallback: index the abstract
            chunks = [Document(page_content=rec["abstract"],
                               metadata={"paper_id": pid, "title": rec.get("title", "")})]

        if chunks:
            self.chunks_by_paper[pid] = chunks
            self.vstore.add_documents(chunks)       # tool 10: upsert_vector_db
        self.graph.add_node(pid)

    # ----- Tool 11 : hybrid_retrieve ----------------------------------------
    def hybrid_retrieve(self, hypothesis_text: str) -> list[Document]:
        """Vector search UNION graph expansion -> ~N_CANDIDATES chunks."""
        # (a) dense vector search
        vec_hits = self.vstore.similarity_search(hypothesis_text, k=N_CANDIDATES)

        seen = {(d.metadata.get("paper_id"), d.page_content[:80]) for d in vec_hits}
        results = list(vec_hits)

        # (b) graph expansion: neighbours of the papers we just hit
        hit_papers = {d.metadata.get("paper_id") for d in vec_hits}
        for pid in list(hit_papers):
            if pid not in self.graph:
                continue
            neighbours = set(self.graph.successors(pid)) | set(self.graph.predecessors(pid))
            for npid in neighbours:
                for c in self.chunks_by_paper.get(npid, []):
                    key = (npid, c.page_content[:80])
                    if key not in seen:
                        seen.add(key)
                        results.append(c)
                    if len(results) >= N_CANDIDATES:
                        return results[:N_CANDIDATES]
        return results[:N_CANDIDATES]

    # ----- ENTRY POINT used by Agent 1 --------------------------------------
    def retrieve(self, hypothesis: SubHypothesis) -> list[Document]:
        """Full Librarian pass for one hypothesis: search -> crawl -> ingest ->
        hybrid retrieve. Returns ~50 candidate chunks for the Critic."""
        self._edge_budget = 0
        query = " ; ".join(hypothesis.query_terms) or hypothesis.hypothesis

        # 1) discovery
        seeds = self.search_arxiv(query) + self.search_semantic_scholar(query)

        # 2) ingest seeds (full text) + crawl their neighbours (abstracts)
        for s in seeds:
            self._ingest_record(s, full=True)
            if not s.get("s2_id"):
                continue
            for rel_rec in self.get_references(s["s2_id"]) + self.get_citations(s["s2_id"]):
                self._ingest_record(rel_rec, full=False)
                relation = self.classify_edge(rel_rec.get("context", ""))
                self.upsert_graph(s["paper_id"], rel_rec["paper_id"], relation)

        # 3) retrieve candidates for THIS hypothesis
        return self.hybrid_retrieve(hypothesis.hypothesis)

    # ----- expose every tool as a LangChain StructuredTool -------------------
    def _register_tools(self) -> list[StructuredTool]:
        spec = [
            ("search_arxiv", self.search_arxiv, "Keyword search ArXiv."),
            ("search_semantic_scholar", self.search_semantic_scholar, "Keyword search Semantic Scholar."),
            ("get_references", self.get_references, "Backward citation crawl."),
            ("get_citations", self.get_citations, "Forward citation crawl."),
            ("fetch_pdf", self.fetch_pdf, "Download a PDF (rate-limited, cached)."),
            ("parse_to_markdown", self.parse_to_markdown, "Convert a PDF to text/Markdown."),
            ("classify_edge", self.classify_edge, "Label a citation edge relation."),
            ("hybrid_retrieve", self.hybrid_retrieve, "Vector + graph retrieval."),
        ]
        return [StructuredTool.from_function(func=fn, name=name, description=desc)
                for name, fn, desc in spec]


# ----------------------------------------------------------------------------- #
# Connecting Agent 2 to Agent 1
# ----------------------------------------------------------------------------- #

def build_evidence_provider(librarian: Librarian):
    """Return an evidence_provider(SubHypothesis) -> (score, answer, citations)
    matching Agent 1's loop.

    NOTE: the scoring + answer drafting here is a TEMPORARY stand-in for Agent 3
    (the Critic). It does a quick cosine relevance over retrieved chunks so the
    1+2 pipeline runs now. Replace the body with the real Critic when ready.
    """
    import numpy as np

    def provider(hypothesis: SubHypothesis) -> tuple[float, str, list[str]]:
        chunks = librarian.retrieve(hypothesis)
        if not chunks:
            return 0.0, "", []

        q = np.asarray(embedder.embed_query(hypothesis.hypothesis), dtype=float)
        cvs = np.asarray(embedder.embed_documents([c.page_content for c in chunks]), dtype=float)
        q /= np.linalg.norm(q) + 1e-12
        cvs /= (np.linalg.norm(cvs, axis=1, keepdims=True) + 1e-12)
        sims = cvs @ q

        order = sims.argsort()[::-1][:3]                  # TEMP: top-3 by cosine
        best = float(sims[order[0]])
        answer = " ".join(chunks[i].page_content[:400] for i in order)
        citations = list({chunks[i].metadata.get("paper_id", "?") for i in order})
        return best, answer, citations

    return provider


# ----------------------------------------------------------------------------- #
# Demo : the connected Agent 1 + Agent 2 pipeline
# ----------------------------------------------------------------------------- #

if __name__ == "__main__":
    import os
    from orchestrator import Orchestrator

    if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        raise SystemExit("Set HUGGINGFACEHUB_API_TOKEN to run the demo.")

    librarian = Librarian()
    evidence_provider = build_evidence_provider(librarian)

    orchestrator = Orchestrator(evidence_provider=evidence_provider)
    report = orchestrator.run(
        "Build a low-latency system to detect fraudulent transactions in real time."
    )
    print("\n===== FINAL REPORT =====\n")
    print(report)