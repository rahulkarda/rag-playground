# rag-playground

Building a retrieval-augmented generation system from scratch. Exploring chunking strategies, embedding models, retrieval methods, and evaluation harnesses.

## Goals

- Implement a minimal RAG pipeline end-to-end without high-level frameworks
- Compare chunking strategies (fixed-size, semantic, recursive)
- Try different retrievers: dense, sparse, hybrid
- Build a small eval harness with answer-relevance + faithfulness metrics
- Keep it readable — small files, clear interfaces

## Status

Early — see [ROADMAP.md](ROADMAP.md) for the plan.

## Setup

```bash
pip install -r requirements.txt
```

## Chunker Usage

You can use the `fixed_size_chunks` utility for quick chunking of large texts:

```python
from src.chunker import fixed_size_chunks, chunk_stats

text = "Your document..."
chunks = list(fixed_size_chunks(text, size=512, overlap=128))

for chunk in chunks:
    print(f"Chunk from {chunk.start} to {chunk.end}: \n{chunk.text}\n")
```

## Chunk Statistics

The `chunk_stats` function summarizes chunk statistics (number, sizes, word counts):

```python
stats = chunk_stats(chunks)
print(stats)
# Example output:
# {'num_chunks': 5, 'avg_chunk_size_chars': 510.4, 'min_chunk_size_chars': 128, ...}
```

See `src/chunker.py` for additional utilities: word/sentence/paragraph/line/token counters.
