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

## Quickstart

- Chunk a document:

```python
from src.chunker import fixed_size_chunks
text = "Your document..."
chunks = list(fixed_size_chunks(text, size=512, overlap=128))
```

- Summarize chunk statistics:

```python
from src.chunker import chunk_stats
stats = chunk_stats(chunks)
print(stats)
```

- Embed a batch (requires sentence-transformers):

```python
from src.embedder import SentenceTransformerEmbedder
embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
embs = embedder.embed_batch([c.text for c in chunks])
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

## FAQ

**Q:** Why do I get `ModuleNotFoundError: No module named 'sentence_transformers'`?

**A:** Install `sentence-transformers`:

```bash
pip install sentence-transformers
```

**Q:** How do I run the evaluation CLI?

**A:**

```bash
python -m src.main --input questions.jsonl --output results.jsonl --faithfulness
```

**Q:** Can I use my own chunker?

**A:** Yes, see `src/recursive_chunker.py` and `src/semantic_chunker.py` for extensible chunking strategies.
