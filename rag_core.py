import os
import uuid
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    Language,
    MarkdownTextSplitter
)

print("--> [RAG Core] Initializing Embedder and ChromaDB...")

class OllamaEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self):
        self.embedder = OllamaEmbeddings(model="nomic-embed-text")
        
    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.embedder.embed_documents(input)

nomic_ef = OllamaEmbeddingFunction()

chroma_client = chromadb.PersistentClient(
    path="./qdrant_db", 
    settings=Settings(anonymized_telemetry=False)
)

COLLECTIONS = {
    "papers": chroma_client.get_or_create_collection("papers", embedding_function=nomic_ef),
    "repos": chroma_client.get_or_create_collection("repos", embedding_function=nomic_ef),
    "web": chroma_client.get_or_create_collection("web", embedding_function=nomic_ef),
    "user_docs": chroma_client.get_or_create_collection("user_docs", embedding_function=nomic_ef),
}


def chunk_by_strategy(content: str, source_type: str) -> List[str]:
    """
    Routes the content to the correct structure-aware chunker based on the source.
    """
    if source_type == "paper":
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n## ", "\n### ", "\nAbstract", "\nIntroduction", "\nMethodology", "\n\n", "\n", " "],
            chunk_size=2000,
            chunk_overlap=200
        )
        return splitter.split_text(content)

    elif source_type == "repo":
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, 
            chunk_size=512, 
            chunk_overlap=50
        )
        return splitter.split_text(content)

    elif source_type == "web":
        splitter = MarkdownTextSplitter(
            chunk_size=400,
            chunk_overlap=50
        )
        return splitter.split_text(content)

    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        return splitter.split_text(content)


def rag_ingest(content: str, collection_name: str, metadata: Dict[str, Any]):
    """
    Chunks content based on its origin, generates metadata, and stores it in Chroma.
    """
    if collection_name not in COLLECTIONS:
        raise ValueError(f"Collection {collection_name} does not exist.")
        
    source_type = metadata.get("source_type", "user_docs")
    chunks = chunk_by_strategy(content, source_type)
    
    if not chunks:
        return
        
    ids = [str(uuid.uuid4()) for _ in chunks]
    
    metadatas = [metadata.copy() for _ in chunks]
    
    for i, meta in enumerate(metadatas):
        meta["chunk_index"] = i

    print(f"--> [RAG Core] Ingesting {len(chunks)} chunks into '{collection_name}'...")
    
    COLLECTIONS[collection_name].add(
        documents=chunks,
        metadatas=metadatas,
        ids=ids
    )

def rag_query(query: str, target_collections: List[str] = None, top_k: int = 5) -> str:
    """
    Searches across specified collections and returns a synthesized context string.
    """
    if target_collections is None:
        target_collections = ["user_docs"] 
        
    all_retrieved_chunks = []
    
    for coll_name in target_collections:
        if coll_name in COLLECTIONS:
            results = COLLECTIONS[coll_name].query(
                query_texts=[query],
                n_results=top_k
            )
            
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            
            for doc, meta in zip(documents, metadatas):
                source = meta.get("paper_id") or meta.get("repo_url") or meta.get("file") or "Unknown Source"
                formatted_chunk = f"[Source: {source}]\n{doc}\n"
                all_retrieved_chunks.append(formatted_chunk)

    compiled_context = "\n---\n".join(all_retrieved_chunks)
    
    return compiled_context