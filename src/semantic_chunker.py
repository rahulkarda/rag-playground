"""
Semantic chunking utilities for retrieval-augmented generation (RAG).

This module implements chunking based on embedding similarity, allowing for splitting documents into semantically coherent chunks.

Features:
- Semantic chunker: splits text where embedding similarity drops, producing chunks aligned with semantic boundaries.
- Comparison utilities: compare fixed-size, recursive, and semantic chunking strategies.
- Useful for experiments in RAG where chunk granularity and semantic coherence affect retrieval and answer quality.

Typical usage:
    from src.semantic_chunker import semantic_chunker, compare_chunkers
    chunks = semantic_chunker(text, embed_fn)
    # embed_fn: function that maps text -> embedding vector (e.g. from src.embedder)

See also:
    - src.chunker.py: basic chunking and statistics
    - src.recursive_chunker.py: recursive chunking respecting structure
    - src.embedder.py: embedding models
"""

import numpy as np
from typing import List, Callable, Optional
from src.chunker import Chunk


def semantic_chunker(
    text: str,
    embed_fn: Callable[[str], np.ndarray],
    min_chunk_size: int = 128,
    max_chunk_size: int = 2048,
    similarity_threshold: float = 0.85,
    sliding_window: int = 32
) -> List[Chunk]:
    """
    Split text into semantically coherent chunks using embedding similarity.
    Chunks are created where similarity between adjacent windows drops below a threshold.

    Args:
        text (str): Input document.
        embed_fn (Callable): Function mapping text -> embedding vector.
        min_chunk_size (int): Minimum chunk size (characters).
        max_chunk_size (int): Maximum chunk size (characters).
        similarity_threshold (float): Cosine similarity threshold to split.
        sliding_window (int): Window size (characters) for embedding comparison.

    Returns:
        List[Chunk]: List of semantic chunks.
    """
    text_len = len(text)
    if text_len <= min_chunk_size:
        return [Chunk(text=text, start=0, end=text_len)]

    # Generate windows
    window_starts = list(range(0, text_len - sliding_window + 1, sliding_window))
    window_embs = [embed_fn(text[s:s+sliding_window]) for s in window_starts]

    # Compute similarity between adjacent windows
    sims = []
    for i in range(len(window_embs) - 1):
        a = window_embs[i]
        b = window_embs[i+1]
        sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
        sims.append(sim)

    # Identify split points
    splits = [0]
    for i, sim in enumerate(sims):
        if sim < similarity_threshold:
            split_pos = window_starts[i+1]
            if split_pos - splits[-1] >= min_chunk_size:
                splits.append(split_pos)
    splits.append(text_len)

    # Merge chunks exceeding max_chunk_size
    chunks = []
    for i in range(len(splits)-1):
        start = splits[i]
        end = splits[i+1]
        chunk_text = text[start:end]
        if len(chunk_text) > max_chunk_size:
            # Split further
            for j in range(start, end, max_chunk_size):
                sub_end = min(j + max_chunk_size, end)
                chunks.append(Chunk(text=text[j:sub_end], start=j, end=sub_end))
        else:
            chunks.append(Chunk(text=chunk_text, start=start, end=end))
    return chunks


def compare_chunkers(
    text: str,
    embed_fn: Callable[[str], np.ndarray],
    fixed_size: int = 512,
    recursive_fn: Optional[Callable[[str], List[Chunk]]] = None
) -> dict:
    """
    Compare fixed-size, semantic, and recursive chunking on a single document.

    Args:
        text (str): Input document
        embed_fn (Callable): Embedding function
        fixed_size (int): Fixed chunk size for baseline
        recursive_fn (Optional[Callable]): Recursive chunker function

    Returns:
        dict: Chunk stats for each method
    """
    from src.chunker import fixed_size_chunks, chunk_stats
    results = {}
    # Fixed-size
    fixed_chunks = list(fixed_size_chunks(text, size=fixed_size, overlap=0))
    results['fixed'] = chunk_stats(fixed_chunks)
    # Semantic
    semantic_chunks = semantic_chunker(text, embed_fn, min_chunk_size=128, max_chunk_size=fixed_size, similarity_threshold=0.85)
    results['semantic'] = chunk_stats(semantic_chunks)
    # Recursive (if provided)
    if recursive_fn:
        rec_chunks = recursive_fn(text)
        results['recursive'] = chunk_stats(rec_chunks)
    return results
