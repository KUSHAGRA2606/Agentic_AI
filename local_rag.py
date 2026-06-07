import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- 1. INGESTION & CHUNKING ---
print("Loading and splitting the document...")
# Load the specific RAG paper you have
loader = PyPDFLoader("2005.11401v4.pdf")
docs = loader.load()

# Split the text so the LLM doesn't get overwhelmed by the whole 19 pages at once
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
print(f"Created {len(splits)} chunks.")

# --- 2. VECTOR DATABASE (NON-PARAMETRIC MEMORY) ---
print("Generating embeddings and building the local vector database...")
# Use a fast, local HuggingFace model to turn text into numbers
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Store those numbers in ChromaDB (this creates a local sqlite database folder)
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./chroma_db")

# Set up the retriever to fetch the top 3 most relevant chunks based on our query
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# --- 3. GENERATOR (PARAMETRIC MEMORY) ---
print("Connecting to local Llama 3 model...")
# Initialize Ollama running locally
llm = Ollama(model="llama3")

# Define how the LLM should behave
system_prompt = (
    "You are an expert AI research assistant. "
    "Use the following pieces of retrieved context from a research paper to answer the question. "
    "If the answer is not in the context, say that you don't know based on the provided text. "
    "Keep the answer concise and accurate.\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# --- 4. THE CHAIN ---
# Combine the LLM, the prompt, and the retriever into one workflow
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# --- 5. EXECUTION ---
print("\n=== Setup Complete! ===")
while True:
    query = input("\nAsk a question about the paper (or type 'exit' to quit): ")
    if query.lower() == 'exit':
        break
    
    print("\nThinking...")
    # Invoke the chain with the user's question
    response = rag_chain.invoke({"input": query})
    print("\nAnswer:")
    print(response["answer"])