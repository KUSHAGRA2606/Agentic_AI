"""
Agent 1 — The Orchestrator (Principal Investigator)
===================================================

Built with LangChain. The Orchestrator owns FOUR tools and connects them into a
pipeline that drives the whole research-synthesis loop:

    1. decompose_ps   -> splits the problem statement into sub-hypotheses
    2. state_tool     -> reads / writes the shared run state
    3. check_coverage -> decides which hypotheses are answered vs still open
    4. compose_report -> stitches per-hypothesis answers into a final review

The Librarian (retrieval) and Critic (relevance scoring) live in their own
modules. Here they are abstracted behind a single `evidence_provider` callable
so this file runs end-to-end on its own (a mock provider is supplied in
__main__). Swap that callable for the real Librarian+Critic later.

-------------------------------------------------------------------------------
Requirements:
    pip install langchain langchain-openai pydantic
    export OPENAI_API_KEY=...
-------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import json
from typing import Callable, Optional
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env

import numpy as np
from pydantic import BaseModel, Field

from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
    HuggingFaceEmbeddings,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool
from langchain_core.output_parsers import PydanticOutputParser


# ----------------------------------------------------------------------------- #
# Configuration
# ----------------------------------------------------------------------------- #

TAU = 0.62             # relevance threshold a hypothesis must clear to be "answered"
MAX_RETRIES = 3        # how many times the Librarian/Critic loop may retry one hypothesis
SIM_THRESHOLD = 0.85   # cosine sim above which two hypotheses count as duplicates

# Open-source chat model, hosted on the HuggingFace Inference API.
# (For a fully local run, swap HuggingFaceEndpoint for HuggingFacePipeline.)
_endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",   # any open instruct model with decent JSON output
    task="text-generation",
    temperature=0.01,                      # ~0; exact 0 is rejected by some endpoints
    max_new_tokens=1024,                   # needs HUGGINGFACEHUB_API_TOKEN in the environment
)
llm = ChatHuggingFace(llm=_endpoint)

# Open-source sentence embeddings, used for the semantic dedup pass (step B).
embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ----------------------------------------------------------------------------- #
# Data models (the shared state schema)
# ----------------------------------------------------------------------------- #

class SubHypothesis(BaseModel):
    """One narrow, independently-researchable question carved from the PS."""
    id: str = Field(description="short stable id, e.g. 'H1'")
    hypothesis: str = Field(description="the claim/question to research")
    query_terms: list[str] = Field(description="search terms for the Librarian")
    success_criterion: str = Field(description="what an adequate answer must contain")

    # filled in as the loop runs
    status: str = "open"                 # open | answered | gap
    best_score: float = 0.0              # best relevance score seen (from Critic)
    answer: Optional[str] = None         # drafted answer once accepted
    citations: list[str] = Field(default_factory=list)
    retries: int = 0


class DecompositionResult(BaseModel):
    """Structured output of the decompose_ps tool."""
    hypotheses: list[SubHypothesis]


class CoverageReport(BaseModel):
    open_ids: list[str]
    answered_ids: list[str]
    gap_ids: list[str]
    complete: bool


class OrchestratorState(BaseModel):
    """The single object every tool reads from and writes to."""
    problem_statement: str = ""
    hypotheses: list[SubHypothesis] = Field(default_factory=list)
    final_report: Optional[str] = None


# ----------------------------------------------------------------------------- #
# Tool 2: state read/write  (defined first because other tools use it)
# ----------------------------------------------------------------------------- #

class StateStore:
    """In-memory shared state. In production back this with Redis / a DB row."""
    def __init__(self) -> None:
        self._state = OrchestratorState()

    def read(self) -> OrchestratorState:
        return self._state

    def write(self, **updates) -> OrchestratorState:
        self._state = self._state.model_copy(update=updates)
        return self._state


def _make_state_tool(store: StateStore) -> StructuredTool:
    def state_tool(action: str, payload_json: str = "{}") -> str:
        """Read or write the shared run state.

        action: 'read' returns the current state as JSON.
                'write' merges payload_json into the state and returns the result.
        """
        if action == "read":
            return store.read().model_dump_json(indent=2)
        if action == "write":
            updates = json.loads(payload_json)
            return store.write(**updates).model_dump_json(indent=2)
        raise ValueError("action must be 'read' or 'write'")

    return StructuredTool.from_function(
        func=state_tool,
        name="state_tool",
        description="Read or write the shared Orchestrator state.",
    )


# ----------------------------------------------------------------------------- #
# Tool 1: decompose the problem statement into sub-hypotheses
# ----------------------------------------------------------------------------- #

_DECOMPOSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are the Principal Investigator of a research team. "
     "Strip logistics (deadlines, prizes, submission rules) from the problem "
     "statement and decompose ONLY its technical core into 5-8 candidate "
     "sub-hypotheses (over-generate; near-duplicates will be filtered later). "
     "Each must be self-contained and phrased so it works as a standalone "
     "search-engine query, in the style 'State-of-the-art for Y' or "
     "'Feasibility of method W'. Expand acronyms in query_terms so academic "
     "search does not miss them.\n\n{format_instructions}"),
    ("human", "Problem statement:\n{ps}"),
])


def _semantic_dedup(
    hyps: list[SubHypothesis],
    threshold: float = SIM_THRESHOLD,
) -> list[SubHypothesis]:
    """Step B: drop hypotheses that are semantically near-duplicates.

    Embeds each hypothesis, then greedily keeps one and discards any later
    hypothesis whose cosine similarity to an already-kept one exceeds the
    threshold. Surviving hypotheses are guaranteed to be distinct, so each
    is worth a separate search. IDs are reassigned afterwards.
    """
    if len(hyps) <= 1:
        return hyps

    # embed "hypothesis + its query terms" so phrasing AND search intent count
    texts = [f"{h.hypothesis} {' '.join(h.query_terms)}" for h in hyps]
    vecs = np.asarray(embedder.embed_documents(texts), dtype=float)

    # L2-normalise so a dot product == cosine similarity
    vecs /= np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12, None)

    kept: list[SubHypothesis] = []
    kept_vecs: list[np.ndarray] = []
    for h, v in zip(hyps, vecs):
        if kept_vecs and float(np.max(np.stack(kept_vecs) @ v)) > threshold:
            continue  # too similar to something already kept -> redundant search
        kept.append(h)
        kept_vecs.append(v)

    for i, h in enumerate(kept, start=1):   # stable ids after filtering
        h.id = f"H{i}"
    return kept


def _make_decompose_tool() -> StructuredTool:
    # Parse JSON from the model's text output instead of relying on native
    # function calling (which open HuggingFace models often don't support).
    parser = PydanticOutputParser(pydantic_object=DecompositionResult)
    prompt = _DECOMPOSE_PROMPT.partial(
        format_instructions=parser.get_format_instructions())
    chain = prompt | llm | parser

    def decompose_ps(problem_statement: str) -> str:
        """Decompose a PS into distinct, independently-searchable hypotheses."""
        result: DecompositionResult = chain.invoke({"ps": problem_statement})
        result.hypotheses = _semantic_dedup(result.hypotheses)  # step B filter
        return result.model_dump_json(indent=2)

    return StructuredTool.from_function(
        func=decompose_ps,
        name="decompose_ps",
        description="Split a problem statement into distinct researchable sub-hypotheses.",
    )


# ----------------------------------------------------------------------------- #
# Tool 3: check coverage
# ----------------------------------------------------------------------------- #

def _make_coverage_tool(store: StateStore) -> StructuredTool:
    def check_coverage() -> str:
        """Decide which hypotheses are answered, still open, or dead-ended."""
        state = store.read()
        open_ids, answered_ids, gap_ids = [], [], []
        for h in state.hypotheses:
            if h["best_score"] >= TAU:
                answered_ids.append(h["id"])
            elif h["retries"] >= MAX_RETRIES:
                gap_ids.append(h["id"])
            else:
                open_ids.append(h["id"])
        report = CoverageReport(
            open_ids=open_ids,
            answered_ids=answered_ids,
            gap_ids=gap_ids,
            complete=(len(open_ids) == 0),
        )
        return report.model_dump_json(indent=2)

    return StructuredTool.from_function(
        func=check_coverage,
        name="check_coverage",
        description="Report which sub-hypotheses are answered, open, or gaps.",
    )


# ----------------------------------------------------------------------------- #
# Tool 4: compose the final report
# ----------------------------------------------------------------------------- #

_REPORT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are writing a concise literature-review synthesis. Use ONLY the "
     "per-hypothesis answers provided. Organise by hypothesis, cite the paper "
     "ids inline, and end with a short 'Open gaps' section for any hypothesis "
     "marked as a gap."),
    ("human", "Problem statement:\n{ps}\n\nFindings:\n{findings}"),
])


def _make_report_tool(store: StateStore) -> StructuredTool:
    chain = _REPORT_PROMPT | llm

    def compose_report() -> str:
        """Stitch all per-hypothesis answers into the final review."""
        state = store.read()
        findings = "\n\n".join(
            f"[{h['id']}] {h['hypothesis']}\nstatus={h['status']} score={h['best_score']:.2f}\n"
            f"answer: {h['answer'] or '(no adequate evidence found)'}\n"
            f"citations: {', '.join(h['citations']) or 'none'}"
            for h in state.hypotheses
        )
        report = chain.invoke({"ps": state.problem_statement, "findings": findings})
        store.write(final_report=report.content)
        return report.content

    return StructuredTool.from_function(
        func=compose_report,
        name="compose_report",
        description="Compose the final literature-review report from findings.",
    )


# ----------------------------------------------------------------------------- #
# The Orchestrator pipeline — connects the 4 tools
# ----------------------------------------------------------------------------- #

# The Librarian + Critic, abstracted. Given one hypothesis it must return:
#   (best_score, drafted_answer, citations)
EvidenceProvider = Callable[[SubHypothesis], tuple[float, str, list[str]]]


class Orchestrator:
    """Agent 1. Owns the 4 tools and runs them as a pipeline."""

    def __init__(self, evidence_provider: EvidenceProvider) -> None:
        self.store = StateStore()
        self.evidence_provider = evidence_provider

        # instantiate the four tools, all sharing one state store
        self.t_decompose = _make_decompose_tool()
        self.t_state = _make_state_tool(self.store)
        self.t_coverage = _make_coverage_tool(self.store)
        self.t_report = _make_report_tool(self.store)

    # --- helper: refine a stuck hypothesis before retrying -------------------
    def _broaden(self, h: SubHypothesis) -> None:
        msg = llm.invoke(
            f"The search for '{h.hypothesis}' returned weak results with terms "
            f"{h.query_terms}. Give 4 broader/alternative search terms as a JSON "
            f"list, nothing else."
        )
        try:
            h.query_terms = json.loads(msg.content)
        except json.JSONDecodeError:
            pass
        h.retries += 1

    def _save_hypotheses(self, hyps: list[SubHypothesis]) -> None:
        self.store.write(hypotheses=[h.model_dump() for h in hyps])

    # --- the pipeline --------------------------------------------------------
    def run(self, problem_statement: str) -> str:
        # 1) seed state
        self.store.write(problem_statement=problem_statement)

        # 2) decompose_ps -> hypotheses
        decomposed = json.loads(self.t_decompose.invoke(
            {"problem_statement": problem_statement}))
        hyps = [SubHypothesis(**h) for h in decomposed["hypotheses"]]
        self._save_hypotheses(hyps)
        print(f"Decomposed into {len(hyps)} sub-hypotheses.")

        # 3) ReAct loop: gather evidence per open hypothesis until coverage holds
        while True:
            for h in hyps:
                if h.status in ("answered", "gap"):
                    continue
                score, answer, citations = self.evidence_provider(h)
                h.best_score = max(h.best_score, score)
                if score >= TAU:
                    h.status, h.answer, h.citations = "answered", answer, citations
                    print(f"  {h.id} answered (score={score:.2f})")
                elif h.retries >= MAX_RETRIES:
                    h.status = "gap"
                    print(f"  {h.id} gave up -> marked gap")
                else:
                    self._broaden(h)
                    print(f"  {h.id} retry {h.retries} (score={score:.2f})")
                self._save_hypotheses(hyps)

            # check_coverage decides whether to stop
            coverage = json.loads(self.t_coverage.invoke({}))
            if coverage["complete"]:
                break

        # 4) compose_report
        report = self.t_report.invoke({})
        return report


# ----------------------------------------------------------------------------- #
# Demo — mock Librarian+Critic so the Orchestrator runs standalone
# ----------------------------------------------------------------------------- #

if __name__ == "__main__":
    import random

    if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        raise SystemExit("Set HUGGINGFACEHUB_API_TOKEN to run the demo.")

    def mock_evidence(h: SubHypothesis) -> tuple[float, str, list[str]]:
        """Stand-in for the real Librarian + Critic.
        Returns a rising score so hypotheses converge after a retry or two."""
        score = min(0.45 + 0.12 * h.retries + random.uniform(0, 0.15), 0.95)
        answer = f"Synthesised finding for '{h.hypothesis}'."
        citations = [f"arxiv:{random.randint(2000, 2500)}.{random.randint(1000, 9999)}"]
        return score, answer, citations

    orchestrator = Orchestrator(evidence_provider=mock_evidence)
    final = orchestrator.run(
        "Build a low-latency system to detect fraudulent transactions in real time."
    )
    print("\n===== FINAL REPORT =====\n")
    print(final)