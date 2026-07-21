# Chunker Documentation

## Overview

The chunker module provides utilities for splitting text into chunks for retrieval-augmented generation (RAG) pipelines. It includes:
- Fixed-size chunking (with overlap)
- Whitespace chunking
- Statistics summarization for chunk sets
- Various text counters (words, sentences, paragraphs, lines, tokens)

## Fixed-Size Chunking

```python
from src.chunker import fixed_size_chunks
text = "..."
chunks = list(fixed_size_chunks(text, size=512, overlap=128))
```
- `size`: maximum chunk length (in chars)
- `overlap`: overlap between consecutive chunks

Each chunk is a `Chunk(text, start, end)` dataclass, with the chunk text and its position.

## Chunk Statistics

```python
from src.chunker import chunk_stats
stats = chunk_stats(chunks)
print(stats)
# Example:
# {'num_chunks': 5, 'avg_chunk_size_chars': 510.4, ...}
```

Returned fields:
- `num_chunks`: number of chunks
- `avg_chunk_size_chars`: average chunk size (chars)
- `min_chunk_size_chars`, `max_chunk_size_chars`: min/max chunk sizes
- `avg_chunk_size_words`: average chunk size (words)
- `min_chunk_size_words`, `max_chunk_size_words`: min/max chunk sizes (words)

## Word, Sentence, Paragraph, Line, Token Counters

Examples:
```python
from src.chunker import count_words, count_sentences, count_paragraphs, count_lines, count_tokens
text = "This is an example."
print(count_words(text))       # 5
print(count_sentences(text))   # 1
print(count_paragraphs(text))  # 1
print(count_lines(text))       # 1
print(count_tokens(text))      # 5
```

## Whitespace Chunking

Splits text on whitespace blocks:
```python
from src.chunker import chunk_whitespace
chunks = chunk_whitespace(text)
```

## Batch Token Counting

```python
from src.chunker import batch_count_tokens
counts = batch_count_tokens(["a b c", "one two three four"])
```

## Notes
- All chunkers return start/end character indices for easy mapping to original text.
- For more advanced chunking, see `recursive_chunker.py` and `semantic_chunker.py`.
