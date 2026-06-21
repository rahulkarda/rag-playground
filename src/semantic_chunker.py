"""
Semantic chunker: splits text into chunks where adjacent segments are semantically similar,
using embedding similarity threshold.

How it works:
- The text is split into segments (default: sentences).
- Embeddings are computed for each segment via embed_fn.
- Adjacent segments are greedily merged if their cosine similarity exceeds the threshold,
  and the merged chunk does not exceed max_size.
- If a chunk is below min_size, it tries to merge further, even if similarity is low.
- Chunk boundaries are thus determined by semantic similarity, not just syntax.

Limitations:
- Boundaries depend on the initial splitter (default: sentences).
- Quality depends on embed_fn (embedding model).
- Does not guarantee all chunks are within [min_size, max_size] but tries to enforce bounds.

Example usage:

    from src.semantic_chunker import semantic_chunks
    import numpy as np
    # Dummy embed function (replace with real model)
    def embed_fn(text):
        np.random.seed(hash(text) % 2**32)
        return np.random.rand(384)
    text = "Sentence one. Sentence two. Sentence three."
    chunks = list(semantic_chunks(text, embed_fn=embed_fn))
    for chunk in chunks:
        print(f"[{chunk.start}:{chunk.end}] {chunk.text}")

"""
from dataclasses import dataclass
from typing import Iterator, List, Optional
import numpy as np

@dataclass
class Chunk:
    text: str
    start: int
    end: int


def semantic_chunks(
    text: str,
    embed_fn,
    similarity_threshold: float = 0.72,
    min_size: int = 64,
    max_size: int = 512,
    split_fn: Optional = None
) -> Iterator[Chunk]:
    """
    Split text into semantic chunks using embedding similarity.
    - embed_fn: function(text: str) -> np.ndarray (embedding)
    - similarity_threshold: cosine similarity threshold for merging
    - min_size, max_size: chunk size bounds (characters)
    - split_fn: custom splitter (defaults to sentences)
    """
    if split_fn is None:
        split_fn = _split_sentences
    segments = split_fn(text)
    starts, ends = _segment_indices(text, segments)
    embeddings = [embed_fn(seg) for seg in segments]
    i = 0
    while i < len(segments):
        curr_text = segments[i]
        curr_start = starts[i]
        curr_end = ends[i]
        curr_emb = embeddings[i]
        j = i + 1
        while j < len(segments):
            next_emb = embeddings[j]
            sim = _cosine(curr_emb, next_emb)
            next_len = len(curr_text) + len(segments[j])
            if sim >= similarity_threshold and next_len <= max_size:
                # Merge
                curr_text += segments[j]
                curr_end = ends[j]
                # Update embedding (average)
                curr_emb = (curr_emb + next_emb) / 2
                j += 1
            else:
                break
        # If chunk too small, try to merge forcibly (unless at end)
        if (curr_end - curr_start) < min_size and j < len(segments):
            curr_text += segments[j]
            curr_end = ends[j]
            curr_emb = (curr_emb + embeddings[j]) / 2
            j += 1
        yield Chunk(text=curr_text, start=curr_start, end=curr_end)
        i = j


def _split_sentences(text: str) -> List[str]:
    import re
    # Simple sentence splitter
    sentences = re.findall(r'[^.!?]+[.!?]', text)
    if not sentences:
        return [text]
    return sentences

def _segment_indices(text: str, segments: List[str]) -> (List[int], List[int]):
    """
    Returns lists of start and end indices for each segment in original text.
    """
    idx = 0
    starts = []
    ends = []
    for seg in segments:
        start = text.find(seg, idx)
        end = start + len(seg)
        starts.append(start)
        ends.append(end)
        idx = end
    return starts, ends

def _cosine(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
