import os
import re
import numpy as np
import faiss
import pickle
from bs4 import BeautifulSoup
from google import genai
from sentence_transformers import SentenceTransformer

_embedder = None
_index    = None
_metadata = None
_client   = None
_project  = None

GEMINI_MODEL = "gemini-2.5-flash"

def init_rag(models_dir, project, location="us-central1"):
    global _embedder, _index, _metadata, _client, _project

    _project = project

    print("  Loading sentence transformer...")
    _embedder = SentenceTransformer(
        os.path.join(models_dir, "all-MiniLM-L6-v2")
    )

    print("  Loading FAISS index...")
    _index = faiss.read_index(
        os.path.join(models_dir, "resume_index.faiss")
    )

    print("  Loading metadata...")
    with open(os.path.join(models_dir, "metadata.pkl"), "rb") as f:
        _metadata = pickle.load(f)

    print("  Initializing Gemini client...")
    _client = genai.Client(vertexai=True,project=project, location=location)

    print(f"  RAG ready — {_index.ntotal} vectors loaded")


def clean_text_for_embedding(text):
    """Light cleaning for embedding — preserves semantic context."""
    text = str(text)
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text[:2000]


def retrieve(query, k=5):
    """Embed query and retrieve top-k similar resumes from FAISS."""
    vector    = _embedder.encode(query, convert_to_numpy=True)
    vector    = (vector / np.linalg.norm(vector)).reshape(1, -1).astype('float32')
    distances, indices = _index.search(vector, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        r = _metadata[idx].copy()
        r['similarity_score'] = round(float(dist), 4)
        results.append(r)
    return results


def generate_answer(query, chunks):
    """Generate recruiter answer from retrieved resume chunks using Gemini."""
    try:
        context_parts = []
        for i, chunk in enumerate(chunks):
            context_parts.append(
                f"Candidate {i+1} | Category: {chunk['category']} | "
                f"Similarity: {chunk['similarity_score']:.3f}\n"
                f"{chunk['text'][:500]}"
            )
        context = "\n\n---\n\n".join(context_parts)

        prompt = f"""You are an expert recruitment assistant helping a hiring manager.
Based ONLY on the resume excerpts provided below, answer the recruiter's question.
Do not invent or assume any skills not explicitly mentioned in the resumes.
Be specific — refer to candidates as Candidate 1, Candidate 2, etc.
If the resumes do not contain enough relevant information, say so clearly.
Never fabricate candidate details. Accuracy is critical for hiring decisions.

Resume excerpts:
{context}

Recruiter question: {query}

Answer:"""

        response = _client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        return response.text

    except Exception as e:
        # Graceful fallback — never return a 500 to the client
        fallback = [f"Candidate {i+1}: {c['category']}"
                    for i, c in enumerate(chunks)]
        return f"LLM temporarily unavailable. Top matches: {', '.join(fallback)}. Error: {str(e)}"


def rag_query(question, k=5):
    """Full RAG pipeline — retrieve + generate."""
    chunks  = retrieve(question, k=k)
    answer  = generate_answer(question, chunks)
    sources = [
        f"Candidate {i+1} ({c['category']}, score: {c['similarity_score']:.3f})"
        for i, c in enumerate(chunks)
    ]
    return {
        'answer':  answer,
        'sources': sources
    }