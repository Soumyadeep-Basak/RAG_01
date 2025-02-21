import os
import warnings
import numpy as np
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader  
# from paddleocr import PaddleOCR
from llama_cpp import Llama

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Initialize models
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = Llama(model_path=r"C:\Users\USER\OneDrive\Desktop\hack\rag\tinyllama-1.1b-chat-v1.0.Q2_K.gguf", verbose=False, n_ctx=2048,n_batch=32)


# FAISS Index
index = None
docs = []  # Store document chunks


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
    """Generate answer using LLM"""
    context = retrieve(query)
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    output = llm(prompt, max_tokens=100)
    return output["choices"][0]["text"]

def generate_answer(query):
    """Generate answer using LLM with streaming output"""
    context = retrieve(query)
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"

    prompt_tokens = len(prompt.split())  # Count input words
    max_allowed_tokens = 2048 - prompt_tokens  # Prevent exceeding limit

    if max_allowed_tokens <= 0:
        print("Error: Context too long. Reduce retrieved chunks.")
        return

    # Stream output
    for output in llm(prompt, max_tokens=min(200, max_allowed_tokens), stream=True):
        print(output["choices"][0]["text"], end="", flush=True)  # Print as it generates


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
print(f"Context Length (words): {len(context)}")
# Test Query
print(generate_answer("explain the topic of the document"))


