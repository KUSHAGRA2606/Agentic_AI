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
# ==========================================
# 1. EMBEDDING & DATABASE INITIALIZATION
# ==========================================

print("--> [RAG Core] Initializing Embedder and ChromaDB...")

# We use Ollama for local nomic-embed-text-v1.5 (8192 token context limit)
# Make sure `ollama serve` is running and you have run `ollama pull nomic-embed-text`
class OllamaEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self):
        self.embedder = OllamaEmbeddings(model="nomic-embed-text")
        
    def __call__(self, input: List[str]) -> List[List[float]]:
        # ChromaDB expects a list of embeddings back
        return self.embedder.embed_documents(input)

# Initialize the embedding function
nomic_ef = OllamaEmbeddingFunction()

# Initialize persistent ChromaDB client
chroma_client = chromadb.PersistentClient(
    path="./qdrant_db", # Reusing your preferred local path name
    settings=Settings(anonymized_telemetry=False)
)

# Initialize the 4 Core Collections
COLLECTIONS = {
    "papers": chroma_client.get_or_create_collection("papers", embedding_function=nomic_ef),
    "repos": chroma_client.get_or_create_collection("repos", embedding_function=nomic_ef),
    "web": chroma_client.get_or_create_collection("web", embedding_function=nomic_ef),
    "user_docs": chroma_client.get_or_create_collection("user_docs", embedding_function=nomic_ef),
}

# ==========================================
# 2. CHUNKING STRATEGIES
# ==========================================

def chunk_by_strategy(content: str, source_type: str) -> List[str]:
    """
    Routes the content to the correct structure-aware chunker based on the source.
    """
    if source_type == "paper":
        # Strategy: Section-aware (Abstract, Intro, Methods) -> Results -> Conclusion
        # Using a larger chunk size to take advantage of Nomic's 8192 context window
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n## ", "\n### ", "\nAbstract", "\nIntroduction", "\nMethodology", "\n\n", "\n", " "],
            chunk_size=2000,
            chunk_overlap=200
        )
        return splitter.split_text(content)

    elif source_type == "repo":
        # Strategy: Split by file, then by function/class.
        # LangChain has a built-in Python code splitter that respects functions/classes
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, 
            chunk_size=512, 
            chunk_overlap=50
        )
        return splitter.split_text(content)

    elif source_type == "web":
        # Strategy: Split by heading level (H1/H2/H3), then by paragraph.
        splitter = MarkdownTextSplitter(
            chunk_size=400,
            chunk_overlap=50
        )
        return splitter.split_text(content)

    else:
        # Default fallback for user_docs (PDFs, txt, generic markdown)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        return splitter.split_text(content)

# ==========================================
# 3. CORE RAG FUNCTIONS
# ==========================================

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
        
    # Generate unique IDs for each chunk
    ids = [str(uuid.uuid4()) for _ in chunks]
    
    # Duplicate the metadata dictionary for each chunk so Chroma accepts it
    metadatas = [metadata.copy() for _ in chunks]
    
    # Add chunk index to metadata for debugging
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
        target_collections = ["user_docs"] # Default to user context
        
    all_retrieved_chunks = []
    
    for coll_name in target_collections:
        if coll_name in COLLECTIONS:
            results = COLLECTIONS[coll_name].query(
                query_texts=[query],
                n_results=top_k
            )
            
            # Chroma returns a list of lists for documents and metadatas
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            
            for doc, meta in zip(documents, metadatas):
                # Format the retrieved chunk with its source for the LLM
                source = meta.get("paper_id") or meta.get("repo_url") or meta.get("file") or "Unknown Source"
                formatted_chunk = f"[Source: {source}]\n{doc}\n"
                all_retrieved_chunks.append(formatted_chunk)

    # Combine all chunks into a single context block
    compiled_context = "\n---\n".join(all_retrieved_chunks)
    
    return compiled_context
    
# ==========================================
# Example Usage (Can be removed in production)
# ==========================================
# ==========================================
# 6. OLLAMA DIAGNOSTIC TEST
# ==========================================

if __name__ == "__main__":
    print("==================================================")
    print("🧪 INITIATING OLLAMA DIAGNOSTIC TEST 🧪")
    print("==================================================")
    
    try:
        # Test connection by attempting to embed a simple string
        test_text = "Testing Ollama embedding connection."
        print(f"[Testing] Attempting to embed: '{test_text}'")
        
        # Instantiate the embedder directly
        from langchain_community.embeddings import OllamaEmbeddings
        embedder = OllamaEmbeddings(model="nomic-embed-text")
        
        embedding = embedder.embed_query(test_text)
        
        if embedding:
            print("✅ Success! Ollama is running and responding to embedding requests.")
            print(f"   Embedding vector dimension: {len(embedding)}")
        else:
            print("❌ Failed: Ollama returned an empty embedding.")
            
    except Exception as e:
        print("❌ Connection Failed.")
        print(f"   Reason: {e}")
        print("\n💡 Troubleshooting Steps:")
        print("   1. Run 'ollama serve' in a separate terminal.")
        print("   2. Run 'ollama pull nomic-embed-text' if you haven't yet.")
        print("   3. Ensure no firewall is blocking port 11434.")

    print("\n==================================================")
    print("🏁 DIAGNOSTIC COMPLETE 🏁")
    print("==================================================")