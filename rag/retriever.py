# rag/retriever.py

"""
FAISS Retriever
----------------
Given a user's question, finds the most relevant
text chunks from the FAISS index.

Retrieval steps:
  1. Embed the user's question (same model used for indexing)
  2. Normalize the query vector
  3. Search FAISS for top-k nearest vectors
  4. Return the corresponding text chunks + similarity scores

Think of it like:
  - Your notes are filed in a library by "meaning"
  - Your question has a "meaning fingerprint"
  - FAISS finds the files closest to that fingerprint
"""

import faiss
import numpy as np
from embeddings.sentence_embeddings import embed_query as embed_single


def retrieve_relevant_chunks(query: str,
                              index,
                              chunks: list,
                              top_k: int = 3) -> list:
    """
    Retrieves the top-k most relevant chunks for a query.

    Args:
        query (str): User's question
        index: Loaded FAISS index
        chunks (list): Original text chunks
        top_k (int): How many chunks to retrieve

    Returns:
        list: Top-k (chunk_text, similarity_score) tuples
              Sorted by relevance (highest first)
    """

    if index is None or not chunks:
        return []

    # Step 1: Embed the query using the SAME model used for indexing
    query_vec = embed_single(query).astype(np.float32)

    # Step 2: Reshape to 2D (FAISS expects batch input)
    query_vec = query_vec.reshape(1, -1)           # Shape: (1, 384)

    # Step 3: Normalize (same as we did for the index vectors)
    faiss.normalize_L2(query_vec)

    # Step 4: Search FAISS
    # Returns: distances (similarity scores) and indices of top-k matches
    # D = distances matrix (1, top_k)
    # I = indices matrix  (1, top_k)
    D, I = index.search(query_vec, top_k)

    # Step 5: Map indices back to original text chunks
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1:           # FAISS returns -1 for empty slots
            continue
        if idx >= len(chunks):  # Safety check
            continue

        chunk_text  = chunks[idx]
        similarity  = float(score)  # Higher = more relevant (cosine sim)

        results.append((chunk_text, round(similarity, 4)))

    # Sort by similarity (highest first)
    results.sort(key=lambda x: x[1], reverse=True)

    return results


def build_context_string(retrieved_chunks: list,
                          max_chars: int = 1200) -> str:
    """
    Combines retrieved chunks into a single context string
    to pass to the answer generator.

    We limit total length to avoid overflowing the model's
    context window.

    Args:
        retrieved_chunks (list): Output from retrieve_relevant_chunks()
        max_chars (int): Max total characters in context

    Returns:
        str: Combined context string with chunk labels
    """

    context_parts = []
    total_chars   = 0

    for i, (chunk, score) in enumerate(retrieved_chunks):
        # Label each chunk so the model knows they're separate sources
        labeled = f"[Source {i+1} | Relevance: {score:.2f}]\n{chunk}"

        if total_chars + len(labeled) > max_chars:
            break  # Stop if we'd exceed the limit

        context_parts.append(labeled)
        total_chars += len(labeled)

    return "\n\n".join(context_parts)


def is_query_answerable(retrieved_chunks: list,
                         threshold: float = 0.30) -> bool:
    """
    Checks if the retrieved chunks are relevant enough
    to answer the query.

    If the best match has a low similarity score, the question
    is probably not covered in the uploaded notes.

    Args:
        retrieved_chunks (list): Output from retrieve_relevant_chunks()
        threshold (float): Minimum acceptable similarity score

    Returns:
        bool: True if query can be answered, False otherwise
    """

    if not retrieved_chunks:
        return False

    best_score = retrieved_chunks[0][1]  # Highest similarity score
    return best_score >= threshold