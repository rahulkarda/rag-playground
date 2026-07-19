"""
Embedding utilities for retrieval-augmented generation (RAG).

This module provides:
- Wrapper for sentence-transformers models (SentenceTransformerEmbedder)
- Single and batch embedding functions
- Progress-aware batch embedding for large corpora
- Cosine similarity utilities for comparing embeddings
- Factory for generic embed_fn (load_embedder)

Usage:
    from src.embedder import SentenceTransformerEmbedder
    embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
    emb = embedder.embed("Some text")
    batch_emb = embedder.embed_batch(["Doc 1", "Doc 2"])
    sims = embedder.embed_batch_cosine_similarity(["Q1"], ["D1", "D2"])

The embedder class is designed for quick experimentation with different models and embedding strategies
in RAG pipelines. It returns numpy arrays for easy integration with FAISS and retrieval modules.
"""
import numpy as np
from typing import List, Union, Callable, Optional

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    Args:
        a: numpy array
        b: numpy array
    Returns:
        float: cosine similarity in [-1, 1]
    """
    a = np.asarray(a)
    b = np.asarray(b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    denom = norm_a * norm_b
    # Avoid division by zero: if either vector is zero, return 0.0
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

class SentenceTransformerEmbedder:
    """
    Wrapper for sentence-transformers models.
    Provides embed(text), embed_batch(texts), and embed_batch_with_progress(texts) for generating embeddings.
    
    - Use embed(text) for a single string, returns a numpy array of shape (dim,)
    - Use embed_batch(texts) for a list of strings, returns array of shape (len(texts), dim)
    - Use embed_batch_with_progress(texts, batch_size=128, progress_fn=None) to embed large corpora with progress reporting
    
    Model selection: pass model_name to constructor (e.g. "all-MiniLM-L6-v2").
    Batch encoding is recommended for efficiency on large corpora.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> np.ndarray:
        """
        Embed a single text. Returns a numpy array.
        """
        emb = self.model.encode([text], convert_to_numpy=True)[0]
        return emb

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Embed a batch of texts. Returns array of shape (len(texts), dim).
        """
        return self.model.encode(texts, convert_to_numpy=True)

    def embed_batch_with_progress(
        self,
        texts: List[str],
        batch_size: int = 128,
        progress_fn: Optional[Callable[[int, int], None]] = None
    ) -> np.ndarray:
        """
        Embed a batch of texts in batches, reporting progress.
        Args:
            texts: List of input strings
            batch_size: Number of texts per batch (default: 128)
            progress_fn: Optional function(current, total) called after each batch
        Returns:
            Numpy array of shape (len(texts), dim)
        """
        total = len(texts)
        embeddings = []
        for i in range(0, total, batch_size):
            batch = texts[i:i+batch_size]
            batch_emb = self.model.encode(batch, convert_to_numpy=True)
            embeddings.append(batch_emb)
            if progress_fn:
                progress_fn(min(i+batch_size, total), total)
        return np.vstack(embeddings)

    def embed_batch_cosine_similarity(self, texts_a: List[str], texts_b: List[str]) -> np.ndarray:
        """
        Compute cosine similarities between two batches of texts.
        Returns an array of shape (len(texts_a), len(texts_b)) with similarities.
        Useful for comparing queries to documents, etc.
        """
        emb_a = self.embed_batch(texts_a)
        emb_b = self.embed_batch(texts_b)
        # Normalize
        emb_a_norm = emb_a / (np.linalg.norm(emb_a, axis=1, keepdims=True) + 1e-8)
        emb_b_norm = emb_b / (np.linalg.norm(emb_b, axis=1, keepdims=True) + 1e-8)
        # Compute cosine similarity matrix
        sims = np.dot(emb_a_norm, emb_b_norm.T)
        return sims

# Factory for generic embed_fn


def load_embedder(model_name: str = "all-MiniLM-L6-v2") -> Callable[[str], np.ndarray]:
    """
    Returns a function text -> embedding using the chosen model.
    """
    embedder = SentenceTransformerEmbedder(model_name)
    return embedder.embed
