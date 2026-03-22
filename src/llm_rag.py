import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama


# Loading embedding model


embed_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# Loading FAISS index


index = faiss.read_index(
    "vectorstore/diabetes_index.faiss"
)


# Loading chunks


chunks = np.load(
    "vectorstore/chunks.npy",
    allow_pickle=True
)


# Loading TinyLlama


llm = Llama(
    model_path="models/tinyllama.gguf",
    n_ctx=2048,
    n_threads=8
)




def retrieve(query, k=5):

    query_embedding = embed_model.encode([query]).astype("float32")

    distances, indices = index.search(query_embedding, k)

    results = []

    for idx in indices[0]:
        results.append(chunks[idx])

    return results




def ask_llm(query):
    # Check for simple social cues
    lower_query = query.lower().strip()
    if lower_query in ["thank you", "thanks", "thank you!", "thanks!"]:
        return "You're welcome! 😊 If you have more health questions, feel free to ask."

    if lower_query in ["hi", "hello", "hey"]:
        return "Hello! I am your AI healthcare assistant. How can I help you today?"

    
    # Normal RAG retrieval
    context_docs = retrieve(query)
    context = "\n".join(context_docs)

    prompt = f"""
### System:
You are a helpful and friendly medical assistant. Answer clearly and concisely using the provided context.
If the answer is not in the context, say you are not sure.

### Context:
{context}

### Question:
{query}

### Answer:
"""
    response = llm(
        prompt,
        max_tokens=256,
        temperature=0.3,
        stop=["###"]
    )

    return response["choices"][0]["text"].strip()

