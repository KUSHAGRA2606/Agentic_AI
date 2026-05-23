

# Dependencies for RAG Pipeline

# `I will be using llama-parse for parsing the Research papers.`

# `For creating Chunks from the markdown obtained, I'll be using LangChain MarkdownHeaderTextSplitter & RecursiveCharacterTextSplitter.`

# `HuggingFace Sentence Transformer Model (allenai-specter), will be used for generating embeddings from chunks.`

# `Qdrant will be used as Vector Database, for storing embeddings.`



from llama_parse import LlamaParse
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter
)
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from groq import Groq

#API Keys
from google.colab import userdata
LLAMA_API_KEY = userdata.get("LLAMA_API_KEY")
GROQ_API_KEY  = userdata.get("GROQ_API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY)

# Models & Clients
embed_model = SentenceTransformer("allenai-specter")
# Using allenai-specter since it is specifically trained on research papers
VECTOR_DIM  = embed_model.get_embedding_dimension()

# Stored in Drive
QDRANT_PATH = "/content/qdrant_db"


if not qdrant.collection_exists("papers"):
    qdrant.create_collection(
        collection_name="papers",
        vectors_config=VectorParams(
            size=VECTOR_DIM,
            distance=Distance.COSINE
        )
    )

# Step 1: Converting PDF to Markdown via LlamaParse
def convert_pdf(pdf_path):
    """
    Takes in a path to a PDF and returns the parsed Markdown.
    """
    parser = LlamaParse(
        api_key=LLAMA_API_KEY,
        result_type="markdown",
        verbose=True
    )
    documents = parser.load_data(pdf_path)
    return "\n\n".join([doc.text for doc in documents])

# Step 2: Section-Aware Chunking (Since llama-parse's output markdown contains division according to section)
def chunk_markdown(markdown_text, paper_id, paper_title=""):
    """
    First splitting is according to section, then in that section using Langchain's RecursiveCharacterTextSplitter
    """
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#",   "title"),
            ("##",  "section"),
            ("###", "subsection"),
        ],
        strip_headers=False
    )
    header_chunks = header_splitter.split_text(markdown_text)

    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " "]
    )
    final_chunks = char_splitter.split_documents(header_chunks)

    for i, chunk in enumerate(final_chunks):
        chunk.metadata.update({
            "paper_id":    paper_id,
            "paper_title": paper_title,
            "chunk_index": i,
        })
    return final_chunks

# Step 3: Embed + Store in Qdrant
def index_paper(pdf_path, paper_id, paper_title=""):
    print(f"Doc Parsing {paper_title or paper_id}")
    markdown = convert_pdf(pdf_path)

    print("Chunking")
    chunks = chunk_markdown(markdown, paper_id, paper_title)

    texts     = [c.page_content for c in chunks]
    metadatas = [c.metadata     for c in chunks]

    print(f"Embedding {len(chunks)} chunks")
    embeddings = embed_model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32
    ).tolist()

    existing = qdrant.count("papers").count
    points = [
        PointStruct(
            id      = existing + i,
            vector  = embeddings[i],
            payload = {"text": texts[i], **metadatas[i]}
        )
        for i in range(len(chunks))
    ]

    qdrant.upsert(collection_name="papers", points=points)
    print(f"Indexed {len(chunks)} chunks from '{paper_title or paper_id}'")

# Step 4: Retrieve
def retrieve(query, top_k=3, section_filter=None):
    query_vec = embed_model.encode(query).tolist()

    query_filter = None
    if section_filter:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        query_filter = Filter(
            must=[FieldCondition(
                key="section",
                match=MatchValue(value=section_filter)
            )]
        )

    results = qdrant.query_points(
    collection_name="papers",
    query=query_vec,
    limit=top_k,
    query_filter=query_filter,
    with_payload=True
    )
    return [join((r.payload["text"], r.payload, r.score) for r in results.points)]

# Step 5: Generate Answer
def answer(query, section_filter=None):
    results = retrieve(query, section_filter=section_filter)

    context_parts = []
    for text, meta, score in results:
        label = (
            f"[{meta.get('paper_title', meta.get('paper_id', 'Unknown'))} | "
            f"{meta.get('section', 'Unknown Section')} | "
            f"score: {score:.2f}]"
        )
        context_parts.append(f"{label}\n{text}")
    context = "\n\n---\n\n".join(context_parts)

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a research paper assistant. "
                    "Answer ONLY using the provided context. "
                    "Always cite the paper and section you draw from. "
                    "If the context doesn't contain the answer, say so clearly."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}"
            }
        ],
        temperature=0.5
    )
    return response.choices[0].message.content

# Usage
index_paper(
    "/content/RAG.pdf",
    paper_id="paper_001",
    paper_title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
)

# Relevant Query : Correctly Fetched & Generated
print(answer("What was the Effect of Retrieving more documents"))

# Irrelevant Query : Model Correctly Flags Missing
print(answer("What were the BLEU scores?"))

