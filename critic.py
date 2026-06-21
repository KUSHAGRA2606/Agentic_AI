"""
Agent 3 — The Critic (relevance judge)  +  the finished pipeline
================================================================

Baseline implementation of every Critic tool, in the LangChain environment,
sharing the SAME open-source HuggingFace chat model and embeddings as Agents 1
and 2. The Critic receives the Librarian's ~50 candidate chunks and returns the
single tuple Agent 1's loop consumes:

    critic.evaluate(hypothesis, chunks) -> (best_score, answer, citations)

That is exactly the `evidence_provider` contract, so the Critic is the real
replacement for the temporary scorer used in `librarian.build_evidence_provider`.

The relevance metric (the part the PS asks us to "mathematically figure out"):

        Rel(d, q) = alpha * cos(q, d)        # bi-encoder cosine   (recall)
                  + beta  * sigmoid(CE(q, d)) # cross-encoder       (precision)
                  + gamma * centrality(d)     # PageRank in the crawl graph

    with alpha + beta + gamma = 1. Each term lands in [0, 1], so `Rel` is an
    absolute score comparable across hypotheses and directly against TAU. When
    no citation graph is available the gamma term is dropped and alpha/beta are
    renormalised, so the score stays well-defined.

Baseline vs production
----------------------
* cross-encoder : HuggingFaceCrossEncoder (BAAI/bge-reranker-base) — production-ready.
* centrality    : networkx PageRank over the Librarian's graph -> Neo4j GDS in prod.

Requirements (additional to Agent 2's):
    pip install langchain sentence-transformers
    export HUGGINGFACEHUB_API_TOKEN=...
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import networkx as nx

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# --- shared with Agents 1 & 2 : SAME model, embeddings, schema, threshold ----
from orchestrator import llm, embedder, SubHypothesis, TAU


# ----------------------------------------------------------------------------- #
# Configuration
# ----------------------------------------------------------------------------- #

RERANK_MODEL = "BAAI/bge-reranker-base"   # open-source cross-encoder
TOP_N = 10                                # chunks kept after re-ranking (from ~50)
ANSWER_CHUNKS = 5                         # chunks fed to the answer drafter
WEIGHTS = (0.25, 0.60, 0.15)              # (alpha, beta, gamma)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


@dataclass
class ScoredChunk:
    """A candidate chunk with its three signals and the combined relevance."""
    chunk: Document
    paper_id: str
    cos: float
    cross: float
    centrality: float
    relevance: float


# ----------------------------------------------------------------------------- #
# Answer drafting prompt (grounded strictly in the supplied chunks)
# ----------------------------------------------------------------------------- #

_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are answering ONE research sub-hypothesis using ONLY the evidence "
     "passages provided. Be concise and factual, do not invent claims, and cite "
     "the supporting paper ids inline like [arxiv:1234.5678]. If the evidence is "
     "thin, say so explicitly."),
    ("human", "Sub-hypothesis:\n{hypothesis}\n\nEvidence passages:\n{evidence}"),
])


# ----------------------------------------------------------------------------- #
# The Critic
# ----------------------------------------------------------------------------- #

class Critic:
    """Agent 3. Re-ranks, scores relevance, and drafts grounded answers."""

    def __init__(
        self,
        llm=llm,
        embedder=embedder,
        graph: nx.DiGraph | None = None,
        weights: tuple[float, float, float] = WEIGHTS,
        tau: float = TAU,
    ) -> None:
        self.llm = llm
        self.embedder = embedder
        self.graph = graph                      # live reference to Librarian's graph
        self.alpha, self.beta, self.gamma = weights
        self.tau = tau

        self.cross_encoder = HuggingFaceCrossEncoder(model_name=RERANK_MODEL)
        self._answer_chain = _ANSWER_PROMPT | self.llm

        self.tools = self._register_tools()

    # ----- internal : raw cross-encoder scores ------------------------------
    def _cross_scores(self, query: str, passages: list[str]) -> np.ndarray:
        if not passages:
            return np.array([])
        return np.asarray(
            self.cross_encoder.score([(query, p) for p in passages]), dtype=float)

    # ----- Tool 1 : cross_encoder_rerank ------------------------------------
    def cross_encoder_rerank(
        self, query: str, passages: list[str], top_n: int = TOP_N,
    ) -> list[int]:
        """Re-rank passages against the query; return indices of the top_n.

        Solves 'lost in the middle': a cross-encoder reads (query, passage)
        jointly, so it is far more precise than the bi-encoder cosine — but too
        slow for the whole corpus, which is why it runs only on the ~50 here.
        """
        scores = self._cross_scores(query, passages)
        if scores.size == 0:
            return []
        return np.argsort(scores)[::-1][:top_n].tolist()

    # ----- Tool 2 : graph_centrality ----------------------------------------
    def graph_centrality(self, paper_ids: list[str]) -> list[float]:
        """Normalised PageRank of each paper within the crawled citation graph.

        This is structural importance *inside the set our crawl built* — not
        venue or date — so it still measures relevance to this problem.
        """
        if self.graph is None or self.graph.number_of_nodes() == 0:
            return [0.0] * len(paper_ids)
        try:
            pr = nx.pagerank(self.graph)
        except Exception:
            return [0.0] * len(paper_ids)
        top = max(pr.values()) if pr else 1.0
        top = top or 1.0
        return [pr.get(pid, 0.0) / top for pid in paper_ids]

    # ----- Tool 3 : compute_relevance ---------------------------------------
    def compute_relevance(
        self, query: str, passages: list[str], paper_ids: list[str],
    ) -> list[float]:
        """Combine cosine + cross-encoder + centrality into Rel in [0, 1]."""
        if not passages:
            return []

        # cosine (bi-encoder) — reuse the shared embedding space
        q = np.asarray(self.embedder.embed_query(query), dtype=float)
        q /= np.linalg.norm(q) + 1e-12
        dv = np.asarray(self.embedder.embed_documents(passages), dtype=float)
        dv /= np.linalg.norm(dv, axis=1, keepdims=True) + 1e-12
        cos = np.clip(dv @ q, 0.0, 1.0)

        # cross-encoder (precision), squashed to [0, 1]
        cross = _sigmoid(self._cross_scores(query, passages))

        # centrality (structural)
        cen = np.asarray(self.graph_centrality(paper_ids), dtype=float)

        # drop gamma and renormalise if no centrality signal is available
        a, b, g = self.alpha, self.beta, self.gamma
        if not np.any(cen):
            a, b, g = a / (a + b), b / (a + b), 0.0

        rel = a * cos + b * cross + g * cen
        return rel.tolist()

    # ----- Tool 4 : draft_answer --------------------------------------------
    def draft_answer(
        self, hypothesis: str, passages: list[str], paper_ids: list[str],
    ) -> str:
        """Write a grounded answer to the hypothesis from the supplied passages."""
        evidence = "\n\n".join(
            f"[{pid}] {text[:600]}" for pid, text in zip(paper_ids, passages))
        return self._answer_chain.invoke(
            {"hypothesis": hypothesis, "evidence": evidence}).content

    # ----- ENTRY POINT used by the pipeline (the evidence_provider body) -----
    def evaluate(
        self, hypothesis: SubHypothesis, chunks: list[Document],
    ) -> tuple[float, str, list[str]]:
        """Score the Librarian's candidates and, if good enough, draft the answer.

        Returns (best_score, answer, citations) — Agent 1's loop contract.
        """
        if not chunks:
            return 0.0, "", []

        passages = [c.page_content for c in chunks]
        pids = [c.metadata.get("paper_id", "?") for c in chunks]

        # 1) rerank 50 -> top 10
        keep = self.cross_encoder_rerank(hypothesis.hypothesis, passages, TOP_N)
        k_pass = [passages[i] for i in keep]
        k_pids = [pids[i] for i in keep]
        k_chunks = [chunks[i] for i in keep]

        # 2) relevance over the top 10
        rels = self.compute_relevance(hypothesis.hypothesis, k_pass, k_pids)
        scored = [
            ScoredChunk(chunk=c, paper_id=p, cos=0.0, cross=0.0, centrality=0.0,
                        relevance=r)
            for c, p, r in zip(k_chunks, k_pids, rels)
        ]
        scored.sort(key=lambda s: s.relevance, reverse=True)
        best = scored[0].relevance if scored else 0.0

        # 3) draft an answer only when the hypothesis clears the bar
        if best >= self.tau:
            top = scored[:ANSWER_CHUNKS]
            answer = self.draft_answer(
                hypothesis.hypothesis,
                [s.chunk.page_content for s in top],
                [s.paper_id for s in top],
            )
            citations = list(dict.fromkeys(s.paper_id for s in top))  # unique, ordered
        else:
            answer, citations = "", []

        return best, answer, citations

    # ----- expose every tool as a LangChain StructuredTool -------------------
    def _register_tools(self) -> list[StructuredTool]:
        spec = [
            ("cross_encoder_rerank", self.cross_encoder_rerank,
             "Re-rank passages against a query; return top-n indices."),
            ("graph_centrality", self.graph_centrality,
             "Normalised PageRank of papers in the citation graph."),
            ("compute_relevance", self.compute_relevance,
             "Combine cosine + cross-encoder + centrality into a relevance score."),
            ("draft_answer", self.draft_answer,
             "Draft a grounded answer to a hypothesis from evidence passages."),
        ]
        return [StructuredTool.from_function(func=fn, name=name, description=desc)
                for name, fn, desc in spec]


# ============================================================================= #
# The finished system : Agent 1 (Orchestrator) + 2 (Librarian) + 3 (Critic)
# ============================================================================= #

class ResearchPipeline:
    """Wires all three agents into one runnable research-synthesis pipeline."""

    def __init__(self) -> None:
        # imported here so this module also stands alone for unit-testing the Critic
        from orchestrator import Orchestrator
        from librarian import Librarian

        self.librarian = Librarian()
        # the Critic reads the Librarian's LIVE graph, so centrality reflects
        # whatever the crawl has built so far
        self.critic = Critic(graph=self.librarian.graph)
        self.orchestrator = Orchestrator(evidence_provider=self._evidence_provider)

    def _evidence_provider(
        self, hypothesis: SubHypothesis,
    ) -> tuple[float, str, list[str]]:
        """Librarian retrieves -> Critic judges. This single function is the
        seam that joins Agents 2 and 3 onto Agent 1's loop."""
        chunks = self.librarian.retrieve(hypothesis)
        return self.critic.evaluate(hypothesis, chunks)

    def run(self, problem_statement: str) -> str:
        return self.orchestrator.run(problem_statement)


# ----------------------------------------------------------------------------- #
# Demo : run the complete pipeline end-to-end
# ----------------------------------------------------------------------------- #

if __name__ == "__main__":
    import os

    if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        raise SystemExit("Set HUGGINGFACEHUB_API_TOKEN to run the demo.")

    pipeline = ResearchPipeline()
    report = pipeline.run(
        "Build a low-latency system to detect fraudulent transactions in real time."
    )
    print("\n===== FINAL LITERATURE REVIEW =====\n")
    print(report)