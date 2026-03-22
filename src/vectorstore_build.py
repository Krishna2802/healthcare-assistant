import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

# Load all knowledge files

knowledge_dir = "knowledge"

print("Loading knowledge base files...")

all_text = ""

for file in os.listdir(knowledge_dir):
    if file.endswith(".txt"):
        path = os.path.join(knowledge_dir, file)

        print(f"Reading: {file}")

        with open(path, "r", encoding="utf-8") as f:
            all_text += f.read() + "\n\n"


# Create text chunks

print("Splitting text into chunks...")

chunks = [chunk.strip() for chunk in all_text.split("\n\n") if chunk.strip()]

print(f"Total chunks created: {len(chunks)}")


# Create embeddings

print("Creating embeddings...")

embeddings = model.encode(chunks)

dimension = embeddings.shape[1]


# Build FAISS index

print("Building FAISS index...")

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))

# Save vector store

os.makedirs("vectorstore", exist_ok=True)

faiss.write_index(index, "vectorstore/diabetes_index.faiss")

np.save("vectorstore/chunks.npy", chunks)

print("Vector database built successfully!")
print(f"Total knowledge chunks stored: {len(chunks)}")