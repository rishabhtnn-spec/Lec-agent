import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Runs locally — no API key needed
model = SentenceTransformer('all-MiniLM-L6-v2')

DOCS_FOLDER = "docs"
INDEX_FILE = "rag_index.json"

def chunk_text(text: str, chunk_size: int = 200) -> list:
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - 20):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def build_index():
    """Index all documents in the docs/ folder"""
    chunks = []
    
    for filename in os.listdir(DOCS_FOLDER):
        if filename.endswith('.txt') or filename.endswith('.md'):
            filepath = os.path.join(DOCS_FOLDER, filename)
            with open(filepath, 'r') as f:
                text = f.read()
            for chunk in chunk_text(text):
                chunks.append({
                    "text": chunk,
                    "source": filename
                })

    print(f"[RAG] Indexing {len(chunks)} chunks from {DOCS_FOLDER}/")
    
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts).tolist()
    
    index = [
        {"text": c["text"], "source": c["source"], "embedding": embeddings[i]}
        for i, c in enumerate(chunks)
    ]
    
    with open(INDEX_FILE, 'w') as f:
        json.dump(index, f)
    
    print(f"[RAG] Index saved to {INDEX_FILE}")
    return index

def load_index() -> list:
    """Load existing index or build a new one"""
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, 'r') as f:
            return json.load(f)
    return build_index()

def search(query: str, top_k: int = 3) -> list:
    """Find the most relevant chunks for a query"""
    index = load_index()
    
    query_embedding = model.encode([query])[0]
    
    scores = []
    for item in index:
        doc_embedding = np.array(item["embedding"])
        # Cosine similarity
        score = np.dot(query_embedding, doc_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
        )
        scores.append((score, item))
    
    scores.sort(key=lambda x: x[0], reverse=True)
    top = scores[:top_k]
    
    return [
        {"text": item["text"], "source": item["source"], "score": round(float(score), 3)}
        for score, item in top
    ]

def format_context(results: list) -> str:
    """Format retrieved chunks for injection into a prompt"""
    if not results:
        return ""
    context = "Relevant information from your knowledge base:\n\n"
    for r in results:
        context += f"[Source: {r['source']}]\n{r['text']}\n\n"
    return context.strip()
    