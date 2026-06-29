"""
Combined RAG pipeline for local PDFs, arXiv results, and GitHub repositories.

The LangGraph agent can discover arXiv papers and GitHub repositories. This
module turns those retrieval outputs into searchable chunks in the same Qdrant
collection used by the PDF RAG pipeline.
"""

from __future__ import annotations

import os
import uuid
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from groq import Groq
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from llama_parse import LlamaParse
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer


def _get_secret(name: str) -> Optional[str]:
    value = os.getenv(name)
    if value:
        return value

    try:
        from google.colab import userdata

        return userdata.get(name)
    except Exception:
        return None


LLAMA_API_KEY = _get_secret("LLAMA_API_KEY")
GROQ_API_KEY = _get_secret("GROQ_API_KEY")
GITHUB_PERSONAL_ACCESS_TOKEN = _get_secret("GITHUB_PERSONAL_ACCESS_TOKEN")

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "research_memory")
QDRANT_PATH = os.getenv("QDRANT_PATH", "qdrant_db")

embed_model = SentenceTransformer("allenai-specter")
VECTOR_DIM = embed_model.get_embedding_dimension()
qdrant = QdrantClient(path=QDRANT_PATH)
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


def ensure_collection(collection_name: str = COLLECTION_NAME) -> None:
    """Create the Qdrant collection when it does not already exist."""
    if qdrant.collection_exists(collection_name):
        return

    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )


ensure_collection()


def _point_id(source_type: str, source_id: str, chunk_index: int) -> str:
    key = f"{source_type}:{source_id}:{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def _split_text(text: str, metadata: Dict[str, Any]) -> List[Any]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.create_documents([text], metadatas=[metadata])
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index
    return chunks


def _upsert_chunks(
    chunks: Sequence[Any],
    collection_name: str = COLLECTION_NAME,
) -> int:
    if not chunks:
        return 0

    texts = [chunk.page_content for chunk in chunks]
    embeddings = embed_model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32,
    ).tolist()

    points = []
    for chunk, text, vector in zip(chunks, texts, embeddings):
        metadata = dict(chunk.metadata)
        source_type = metadata.get("source_type", "document")
        source_id = metadata.get("source_id") or metadata.get("url") or text[:80]
        chunk_index = metadata.get("chunk_index", 0)
        points.append(
            PointStruct(
                id=_point_id(source_type, str(source_id), int(chunk_index)),
                vector=vector,
                payload={"text": text, **metadata},
            )
        )

    qdrant.upsert(collection_name=collection_name, points=points)
    return len(points)


# Step 1: Convert PDF to Markdown via LlamaParse.
def convert_pdf(pdf_path: str) -> str:
    """Takes a path to a PDF and returns parsed Markdown."""
    if not LLAMA_API_KEY:
        raise ValueError("LLAMA_API_KEY is required for PDF parsing.")

    parser = LlamaParse(
        api_key=LLAMA_API_KEY,
        result_type="markdown",
        verbose=True,
    )
    documents = parser.load_data(pdf_path)
    return "\n\n".join(doc.text for doc in documents)


# Step 2: Section-aware chunking for LlamaParse Markdown output.
def chunk_markdown(markdown_text: str, paper_id: str, paper_title: str = "") -> List[Any]:
    """Split Markdown by paper sections, then into embedding-sized chunks."""
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "title"),
            ("##", "section"),
            ("###", "subsection"),
        ],
        strip_headers=False,
    )
    header_chunks = header_splitter.split_text(markdown_text)

    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " "],
    )
    final_chunks = char_splitter.split_documents(header_chunks)

    for index, chunk in enumerate(final_chunks):
        chunk.metadata.update(
            {
                "source_type": "pdf",
                "source_id": paper_id,
                "paper_id": paper_id,
                "paper_title": paper_title,
                "title": paper_title,
                "chunk_index": index,
            }
        )
    return final_chunks


# Step 3: Embed + store local paper PDF chunks.
def index_paper(
    pdf_path: str,
    paper_id: str,
    paper_title: str = "",
    collection_name: str = COLLECTION_NAME,
) -> int:
    print(f"Doc Parsing {paper_title or paper_id}")
    markdown = convert_pdf(pdf_path)

    print("Chunking")
    chunks = chunk_markdown(markdown, paper_id, paper_title)

    print(f"Embedding {len(chunks)} chunks")
    indexed = _upsert_chunks(chunks, collection_name=collection_name)
    print(f"Indexed {indexed} chunks from '{paper_title or paper_id}'")
    return indexed


def index_text_document(
    text: str,
    source_type: str,
    source_id: str,
    title: str = "",
    url: str = "",
    metadata: Optional[Dict[str, Any]] = None,
    collection_name: str = COLLECTION_NAME,
) -> int:
    """Index any agent-retrieved text into the shared research memory."""
    payload = {
        "source_type": source_type,
        "source_id": source_id,
        "title": title,
        "url": url,
        **(metadata or {}),
    }
    chunks = _split_text(text, payload)
    return _upsert_chunks(chunks, collection_name=collection_name)


def arxiv_search(queries: Iterable[str], max_results: int = 3) -> List[Dict[str, Any]]:
    """Run agent-generated arXiv queries and return normalized paper metadata."""
    import arxiv

    client = arxiv.Client()
    papers = []
    seen_urls = set()

    for query in queries:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        for result in client.results(search):
            if result.entry_id in seen_urls:
                continue

            seen_urls.add(result.entry_id)
            papers.append(
                {
                    "title": result.title,
                    "authors": [author.name for author in result.authors],
                    "summary": result.summary.replace("\n", " "),
                    "categories": result.categories,
                    "arxiv_url": result.entry_id,
                    "source_query": query,
                }
            )

    return papers


def index_arxiv_papers(
    papers: Iterable[Dict[str, Any]],
    collection_name: str = COLLECTION_NAME,
) -> int:
    """Index arXiv metadata/abstracts returned by the agent or arXiv API."""
    indexed = 0
    for paper in papers:
        title = paper.get("title", "")
        url = paper.get("arxiv_url", "") or paper.get("url", "")
        source_id = url or title
        summary = paper.get("summary", "") or paper.get("abstract", "")
        text = "\n".join(
            part
            for part in [
                f"Title: {title}",
                f"Authors: {', '.join(paper.get('authors', []))}",
                f"Categories: {', '.join(paper.get('categories', []))}",
                f"Source query: {paper.get('source_query', '')}",
                f"Summary: {summary}",
                f"Relevance reason: {paper.get('relevance_reason', '')}",
            ]
            if part.strip()
        )
        indexed += index_text_document(
            text=text,
            source_type="arxiv",
            source_id=source_id,
            title=title,
            url=url,
            metadata=paper,
            collection_name=collection_name,
        )
    return indexed


def github_search(
    queries: Iterable[str],
    max_results: int = 5,
    token: Optional[str] = GITHUB_PERSONAL_ACCESS_TOKEN,
) -> List[Dict[str, Any]]:
    """Search GitHub repositories using agent-generated GitHub search syntax."""
    repositories = []
    seen_urls = set()

    for query in queries:
        params = urllib.parse.urlencode({"q": query, "per_page": max_results})
        request = urllib.request.Request(
            f"https://api.github.com/search/repositories?{params}",
            headers={
                "Accept": "application/vnd.github+json",
                **({"Authorization": f"Bearer {token}"} if token else {}),
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            print(f"GitHub search failed for '{query}': {exc}")
            continue

        import json

        for item in json.loads(payload).get("items", []):
            url = item.get("html_url", "")
            if url in seen_urls:
                continue

            seen_urls.add(url)
            repositories.append(
                {
                    "name": item.get("full_name", ""),
                    "description": item.get("description", "") or "",
                    "language": item.get("language", "") or "",
                    "topics": item.get("topics", []),
                    "stars": item.get("stargazers_count", 0),
                    "forks": item.get("forks_count", 0),
                    "url": url,
                    "source_query": query,
                }
            )

    return repositories


def index_github_repositories(
    repositories: Iterable[Dict[str, Any]],
    collection_name: str = COLLECTION_NAME,
) -> int:
    """Index GitHub repository metadata returned by MCP or GitHub's API."""
    indexed = 0
    for repo in repositories:
        url = repo.get("html_url", "") or repo.get("url", "")
        name = repo.get("full_name", "") or repo.get("name", "") or url
        topics = repo.get("topics", [])
        if isinstance(topics, str):
            topics = [topics]

        text = "\n".join(
            part
            for part in [
                f"Repository: {name}",
                f"URL: {url}",
                f"Description: {repo.get('description', '')}",
                f"Language: {repo.get('language', '')}",
                f"Topics: {', '.join(topics)}",
                f"Stars: {repo.get('stars', repo.get('stargazers_count', ''))}",
                f"Forks: {repo.get('forks', repo.get('forks_count', ''))}",
                f"Source query: {repo.get('source_query', '')}",
            ]
            if str(part).strip()
        )
        indexed += index_text_document(
            text=text,
            source_type="github",
            source_id=url or name,
            title=name,
            url=url,
            metadata=repo,
            collection_name=collection_name,
        )
    return indexed


def index_agentic_output(
    agent_output: Dict[str, Any],
    collection_name: str = COLLECTION_NAME,
) -> Dict[str, int]:
    """
    Index the final LangGraph output from Agent_Loop/agent.ipynb.

    Supports the existing notebook fields:
    - papers: normalized arXiv papers
    - fetched_papers: raw arXiv API results
    - repo_urls + language: GitHub MCP repository results
    """
    papers = agent_output.get("papers") or agent_output.get("fetched_papers") or []
    repo_urls = agent_output.get("repo_urls", [])
    languages = agent_output.get("language", [])
    repositories = [
        {
            "url": url,
            "name": url.rstrip("/").split("/")[-1],
            "language": languages[index] if index < len(languages) else "",
        }
        for index, url in enumerate(repo_urls)
        if url
    ]

    return {
        "arxiv_chunks": index_arxiv_papers(papers, collection_name=collection_name),
        "github_chunks": index_github_repositories(
            repositories,
            collection_name=collection_name,
        ),
    }


def agentic_retrieve_and_index(
    paper_queries: Iterable[str],
    github_queries: Iterable[str],
    arxiv_max_results: int = 3,
    github_max_results: int = 5,
    collection_name: str = COLLECTION_NAME,
) -> Dict[str, Any]:
    """Run arXiv/GitHub retrieval, index both sources, and return counts."""
    papers = arxiv_search(paper_queries, max_results=arxiv_max_results)
    repositories = github_search(github_queries, max_results=github_max_results)
    return {
        "papers": papers,
        "repositories": repositories,
        "indexed": {
            "arxiv_chunks": index_arxiv_papers(papers, collection_name),
            "github_chunks": index_github_repositories(repositories, collection_name),
        },
    }


# Step 4: Retrieve from combined PDF + arXiv + GitHub memory.
def retrieve(
    query: str,
    top_k: int = 5,
    section_filter: Optional[str] = None,
    source_types: Optional[Sequence[str]] = None,
    collection_name: str = COLLECTION_NAME,
) -> List[Tuple[str, Dict[str, Any], float]]:
    query_vec = embed_model.encode(query).tolist()

    conditions = []
    if section_filter:
        conditions.append(
            FieldCondition(key="section", match=MatchValue(value=section_filter))
        )
    if source_types:
        conditions.append(
            FieldCondition(key="source_type", match=MatchAny(any=list(source_types)))
        )

    query_filter = Filter(must=conditions) if conditions else None
    results = qdrant.query_points(
        collection_name=collection_name,
        query=query_vec,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
    )
    return [
        (point.payload.get("text", ""), point.payload, point.score)
        for point in results.points
    ]


# Step 5: Generate answer from retrieved PDF/arXiv/GitHub context.
def answer(
    query: str,
    section_filter: Optional[str] = None,
    source_types: Optional[Sequence[str]] = None,
    top_k: int = 5,
) -> str:
    if not groq_client:
        raise ValueError("GROQ_API_KEY is required for answer generation.")

    results = retrieve(
        query,
        top_k=top_k,
        section_filter=section_filter,
        source_types=source_types,
    )

    context_parts = []
    for text, meta, score in results:
        title = meta.get("title") or meta.get("paper_title") or meta.get("source_id")
        label = (
            f"[{meta.get('source_type', 'unknown')} | {title} | "
            f"{meta.get('section', 'No Section')} | score: {score:.2f}]"
        )
        url = meta.get("url") or meta.get("arxiv_url", "")
        context_parts.append(f"{label}\nURL: {url}\n{text}")
    context = "\n\n---\n\n".join(context_parts)

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a research and implementation assistant. "
                    "Answer only using the provided PDF, arXiv, and GitHub context. "
                    "Cite the source type, title/repository, and URL when available. "
                    "If the context does not contain the answer, say so clearly."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            },
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content
