"""
Recursive chunker for document splitting that respects markdown and code boundaries.

This module implements chunking logic that attempts to split documents at semantic boundaries:
- Markdown headers (e.g. #, ##, ###)
- Code blocks (triple backticks)
- Paragraphs (blank lines)
If a segment is still too large, it falls back to splitting by sentences, and then by fixed-size chunks.

Boundary detection details:
- Code blocks: detected by lines starting/ending with triple backticks (```). All content inside is treated as one chunk.
- Markdown headers: lines beginning with one or more # (e.g. # Header) start new chunks.
- Paragraphs: separated by two or more newlines (\n\n), each treated as a chunk boundary.
- Fallback: if any chunk exceeds max_size, it is further split by sentences or fixed-size chars.

Designed for use in retrieval-augmented generation pipelines where preserving context structure improves retrieval and answer generation.
"""
from dataclasses import dataclass
from typing import Iterator, List
import re

@dataclass
class Chunk:
    text: str
    start: int
    end: int


def recursive_chunks(text: str, max_size: int = 512, min_size: int = 128) -> Iterator[Chunk]:
    """
    Chunk text recursively, respecting markdown and code boundaries.
    Tries to split at code blocks, headers, and paragraphs, falling back to smaller splits if needed.

    Args:
        text (str): Input text to chunk.
        max_size (int): Maximum chunk size (characters).
        min_size (int): Minimum chunk size before fallback splitting.

    Yields:
        Chunk: Chunked text and start/end indices.
    """
    boundaries = _find_boundaries(text)
    start = 0
    for boundary in boundaries:
        end = boundary
        chunk_text = text[start:end]
        if len(chunk_text) > max_size:
            # Further split this chunk
            for sub_chunk in _split_fallback(chunk_text, start, max_size, min_size):
                yield sub_chunk
        else:
            if chunk_text.strip():
                yield Chunk(text=chunk_text, start=start, end=end)
        start = end
    # tail
    if start < len(text):
        chunk_text = text[start:]
        if len(chunk_text) > max_size:
            for sub_chunk in _split_fallback(chunk_text, start, max_size, min_size):
                yield sub_chunk
        else:
            if chunk_text.strip():
                yield Chunk(text=chunk_text, start=start, end=len(text))


def _find_boundaries(text: str) -> List[int]:
    """
    Find boundary indices for markdown/code structure:
    - Code blocks (```)
    - Headers (#, ##, ###)
    - Blank lines (paragraphs)
    Returns list of indices where a chunk could end.
    """
    boundaries = []
    code_block_matches = list(re.finditer(r'(^```.*?$)(.*?)(^```.*?$)', text, re.MULTILINE | re.DOTALL))
    for m in code_block_matches:
        boundaries.append(m.end())
    header_matches = list(re.finditer(r'^#{1,6} .*$', text, re.MULTILINE))
    for m in header_matches:
        boundaries.append(m.start())
    paragraph_matches = list(re.finditer(r'\n\n+', text))
    for m in paragraph_matches:
        boundaries.append(m.end())
    boundaries = sorted(set(boundaries))
    return boundaries


def _split_fallback(chunk_text: str, offset: int, max_size: int, min_size: int) -> Iterator[Chunk]:
    """
    Fallback splitter: splits chunk_text into smaller pieces by sentences, then by fixed chars.
    """
    # Try splitting by sentences
    import re
    sentences = re.findall(r'[^.!?]+[.!?]', chunk_text)
    curr = 0
    acc = ''
    acc_start = offset
    for sent in sentences:
        if len(acc) + len(sent) > max_size:
            # yield chunk
            yield Chunk(text=acc, start=acc_start, end=acc_start+len(acc))
            acc = sent
            acc_start = offset + curr
        else:
            acc += sent
        curr += len(sent)
    if acc.strip():
        yield Chunk(text=acc, start=acc_start, end=acc_start+len(acc))
    # If still too large, fall back to fixed-size
    remaining = chunk_text[curr:]
    if len(remaining) > min_size:
        pos = 0
        while pos < len(remaining):
            end = min(pos + max_size, len(remaining))
            yield Chunk(
                text=remaining[pos:end],
                start=offset + curr + pos,
                end=offset + curr + end
            )
            pos += max_size
