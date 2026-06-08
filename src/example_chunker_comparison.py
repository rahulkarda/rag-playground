from src.chunker import fixed_size_chunks, chunk_stats, normalize_text
from src.recursive_chunker import recursive_chunks
from src.semantic_chunker import semantic_chunks
import numpy as np

# Dummy embed function for semantic chunker

def dummy_embed_fn(text):
    np.random.seed(hash(text) % 2**32)
    return np.random.rand(384)

if __name__ == "__main__":
    # Example corpus: short markdown doc
    corpus = """
# Intro
Retrieval-augmented generation (RAG) combines search and language models.

## Why chunking?
Chunking splits documents to improve retrieval granularity.

```python
def foo():
    pass
```

Semantic chunking uses embedding similarity.
"""
    corpus = normalize_text(corpus)
    chunkers = {
        "fixed": lambda t: list(fixed_size_chunks(t, size=64, overlap=16)),
        "recursive": lambda t: list(recursive_chunks(t, max_size=64, min_size=16)),
        "semantic": lambda t: list(semantic_chunks(
            t,
            embed_fn=dummy_embed_fn,
            similarity_threshold=0.5,
            min_size=24,
            max_size=64
        ))
    }
    print("=== Chunking Comparison ===\n")
    for name, fn in chunkers.items():
        print(f"-- {name.upper()} --")
        chunks = fn(corpus)
        stats = chunk_stats(chunks)
        for i, chunk in enumerate(chunks):
            preview = chunk.text.strip().replace('\n', ' ')[:40]
            print(f"Chunk {i}: [{chunk.start}:{chunk.end}] '{preview}'")
        print("Stats:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        print()
