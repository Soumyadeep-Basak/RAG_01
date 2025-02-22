import os
import warnings
import numpy as np
import faiss
import diskcache as dc
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader  
from llama_cpp import Llama
import time

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Initialize models
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = Llama(model_path=r"C:\Users\USER\OneDrive\Desktop\hack\rag\phi-2-q4_k_m.gguf", verbose=False, n_ctx=2048, n_batch=32)

# FAISS Index
index = None
docs = []  # Store document chunks

# Initialize DiskCache (persistent storage)
cache = dc.Cache("./llm_cache")  # Stores cached responses in `./llm_cache` folder


def process_text(file_path):
    """Process text files (.txt)"""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_text(text)


def process_pdf(file_path):
    """Process PDFs (.pdf)"""
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    return [doc.page_content for doc in docs]  # Extract text content


def build_faiss_index(document_texts):
    """Store text embeddings in FAISS"""
    global index, docs

    # Convert text to embeddings
    doc_vectors = embedding_model.embed_documents(document_texts)
    docs.extend(document_texts)

    # Store text embeddings
    if index is None:
        index = faiss.IndexFlatL2(len(doc_vectors[0]))
    index.add(np.array(doc_vectors))


def retrieve(query, k=1):
    """Retrieve most relevant documents"""
    if index is None:
        return ["No documents available."]

    query_vector = np.array(embedding_model.embed_query(query)).reshape(1, -1)
    distances, indices = index.search(query_vector, k)
    retrieved_docs = [docs[i] for i in indices[0]]
    return retrieved_docs


def generate_answer(query):
    """Generate answer using LLM with caching and streaming"""
    
    if query in cache:  # Check if response is cached
        cached_response = cache[query]
        for word in cached_response.split():
            print(word, end=" ", flush=True)
            time.sleep(0.2)  # Simulating real-time output
        print()
        return

    context = retrieve(query)
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"

    prompt_tokens = len(prompt.split())  # Count input words
    max_allowed_tokens = 8192 - prompt_tokens  # Prevent exceeding limit

    if max_allowed_tokens <= 0:
        print("Error: Context too long. Reduce retrieved chunks.")
        return

    # Generate LLM output with streaming
    response = ""
    for output in llm(prompt, max_tokens=min(200, max_allowed_tokens), stream=True):
        word = output["choices"][0]["text"]
        print(word, end="", flush=True)
        response += word

    # Cache response
    cache[query] = response
    print()
    return response


# --- Main Execution ---
file_path = "./data/mathematics-11-01130.pdf"  # Change this to test different files

file_ext = os.path.splitext(file_path)[-1].lower()

if file_ext == ".txt":
    text_chunks = process_text(file_path)
    build_faiss_index(text_chunks)

elif file_ext == ".pdf":
    text_chunks = process_pdf(file_path)
    build_faiss_index(text_chunks)

else:
    raise ValueError("Unsupported file type!")

context = retrieve("Explain the attached text")
# print(f"Context Length (words): {len(context)}")

# Test Query
generate_answer("explain the topic of the document in 350 words")  # First time: Computes, caches
