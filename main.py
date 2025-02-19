from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import faiss
import numpy as np
# Load document
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

with open("./data/a.txt", "r", encoding="utf-8") as f:  # Forward slashes

    text = f.read()

# Split text into smaller chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = splitter.split_text(text)

# Convert text to embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
doc_vectors = embedding_model.embed_documents(docs)

# Store in FAISS
index = faiss.IndexFlatL2(len(doc_vectors[0]))
index.add(np.array(doc_vectors))


def retrieve(query, k=3):
    query_vector = np.array(embedding_model.embed_query(query)).reshape(1, -1)
    distances, indices = index.search(query_vector, k)
    retrieved_docs = [docs[i] for i in indices[0]]
    return retrieved_docs

query = "What is the main topic of the document?"
context = retrieve(query)
print("\n".join(context))
print("Heieieieieiie")

from llama_cpp import Llama

llm = Llama(model_path=r"C:\Users\USER\OneDrive\Desktop\hack\rag\tinyllama-1.1b-chat-v1.0.Q2_K.gguf",verbose=False)

def generate_answer(query):
    context = retrieve(query)
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    output = llm(prompt, max_tokens=100)
    return output["choices"][0]["text"]

print(generate_answer("Expand the document and write in 500 words."))
