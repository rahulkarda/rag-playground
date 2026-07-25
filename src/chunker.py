"""
A collection of chunking and text statistics utilities for retrieval-augmented generation (RAG).

Includes:
- Fixed-size chunker with configurable overlap
- Whitespace chunker
- Word, sentence, paragraph, line, character, token counters
- Chunk statistics summarizer

This module serves as the base for more advanced chunking strategies (recursive, semantic) and is designed for simple, readable experimentation on text splitting approaches.

Usage examples:
    from src.chunker import fixed_size_chunks, chunk_stats
    text = "Your document..."
    chunks = list(fixed_size_chunks(text, size=512, overlap=128))
    stats = chunk_stats(chunks)
    print(stats)

See also:
    - count_words, count_sentences, count_paragraphs, count_lines, count_tokens, count_tiktoken_tokens
    - batch_count_tokens for batch processing
"""
from dataclasses import dataclass
from typing import Iterator, List, Dict

@dataclass
class Chunk:
    text: str
    start: int
    end: int


def fixed_size_chunks(text: str, size: int = 1024, overlap: int = 128) -> Iterator[Chunk]:
    """
    Split text into fixed-size chunks with optional overlap.

    Args:
        text (str): Input text to chunk.
        size (int, optional): Maximum length of each chunk (in characters). Default is 1024.
        overlap (int, optional): Number of characters to overlap between consecutive chunks. Default is 128.

    Yields:
        Chunk: A dataclass containing chunk text and its start/end character indices.

    Raises:
        ValueError: If size <= 0, overlap < 0, or overlap >= size.

    Example:
        >>> text = 'abcdefghi'
        >>> list(fixed_size_chunks(text, size=4, overlap=1))
        [Chunk(text='abcd', start=0, end=4), Chunk(text='defg', start=3, end=7), Chunk(text='ghi', start=6, end=9)]
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    step = size - overlap
    pos = 0
    text_len = len(text)
    while pos < text_len:
        end = min(pos + size, text_len)
        yield Chunk(text=text[pos:end], start=pos, end=end)
        pos += step
        if pos >= text_len:
            break


def chunk_whitespace(text: str) -> List[Chunk]:
    """
    Split text into chunks by contiguous non-whitespace sequences (words or blocks).
    Each chunk consists of a segment of text (no whitespace), with start/end character indices in the original text.

    This function splits the text on runs of whitespace (spaces, tabs, newlines) and returns a list of Chunk objects,
    where each chunk contains a non-whitespace segment and its position in the original text.

    Args:
        text (str): Input text to chunk.
    Returns:
        List[Chunk]: List of Chunk objects.
    """
    import re
    segments = [s for s in re.split(r'\s+', text.strip()) if s]
    chunks = []
    idx = 0
    for seg in segments:
        # Find the segment's start index after the previous segment
        start = text.find(seg, idx)
        end = start + len(seg)
        chunks.append(Chunk(text=seg, start=start, end=end))
        idx = end if end > idx else start + len(seg)
    return chunks


def count_words(text: str) -> int:
    """
    Count the number of words in a text string.
    """
    if text is None:
        return 0
    return len(text.split())


def count_sentences(text: str) -> int:
    """
    Count the number of sentences in a text string.
    A sentence is defined as ending with '.', '!', or '?'.
    Handles edge case where text lacks terminal punctuation by counting trailing content as one sentence.
    """
    import re
    if not text or not text.strip():
        return 0
    sentences = re.findall(r'[^.!?]+[.!?]', text)
    remainder = text.strip()
    if sentences:
        matched_len = sum(len(s) for s in sentences)
        leftover = remainder[matched_len:]
        if leftover and leftover.strip():
            return len(sentences) + 1
        else:
            return len(sentences)
    else:
        return 1 if remainder else 0


def count_characters(text: str) -> int:
    """
    Count the number of characters in a text string.
    """
    if text is None:
        return 0
    return len(text)


def count_paragraphs(text: str) -> int:
    """
    Count the number of paragraphs in a text string.
    A paragraph is defined as a block of text separated by one or more blank lines.
    """
    paragraphs = [p for p in text.split('\n\n') if p.strip()]
    return len(paragraphs)


def count_lines(text: str) -> int:
    """
    Count the number of lines in a text string.
    A line is any sequence of characters separated by a newline ('\n').
    """
    if not text:
        return 0
    return len(text.splitlines())


def count_tokens(text: str) -> int:
    """
    Count the number of tokens in a text string using whitespace splitting.
    This is a rough proxy for true tokenization.
    """
    if not text:
        return 0
    return len(text.split())


def batch_count_tokens(texts: List[str]) -> List[int]:
    """
    Count tokens for each text in a batch using whitespace splitting.
    """
    return [count_tokens(t) if t is not None else 0 for t in texts]


def count_tiktoken_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """
    Count the number of tokens in a string using tiktoken (OpenAI tokenizer).
    Args:
        text (str): Input string.
        encoding_name (str): tiktoken encoding name (default: cl100k_base for OpenAI models).
    Returns:
        int: Number of tokens.
    """
    try:
        import tiktoken
    except ImportError:
        raise ImportError("tiktoken is required for count_tiktoken_tokens")
    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(text))


def batch_count_tiktoken_tokens(texts: List[str], encoding_name: str = "cl100k_base") -> List[int]:
    """
    Count tiktoken tokens for each text in a batch.
    """
    return [count_tiktoken_tokens(t, encoding_name) if t is not None else 0 for t in texts]


def chunk_stats(chunks: List[Chunk]) -> Dict[str, float]:
    """
    Summarize chunk statistics: number, average/min/max chunk size (chars, words).

    Args:
        chunks (List[Chunk]): List of Chunk objects (with .text field).
    Returns:
        Dict[str, float]: Summary statistics.

    Example output:
        {'num_chunks': 5, 'avg_chunk_size_chars': 510.4, 'min_chunk_size_chars': 128, ...}
    """
    if not chunks:
        return {
            'num_chunks': 0,
            'avg_chunk_size_chars': 0.0,
            'min_chunk_size_chars': 0,
            'max_chunk_size_chars': 0,
            'avg_chunk_size_words': 0.0,
            'min_chunk_size_words': 0,
            'max_chunk_size_words': 0,
        }
    # Compute character sizes for each chunk
    chunk_sizes_chars = [len(chunk.text) for chunk in chunks]
    # Compute word counts for each chunk
    chunk_sizes_words = [count_words(chunk.text) for chunk in chunks]
    num_chunks = len(chunks)
    avg_chunk_size_chars = sum(chunk_sizes_chars) / num_chunks
    min_chunk_size_chars = min(chunk_sizes_chars)
    max_chunk_size_chars = max(chunk_sizes_chars)
    avg_chunk_size_words = sum(chunk_sizes_words) / num_chunks
    min_chunk_size_words = min(chunk_sizes_words)
    max_chunk_size_words = max(chunk_sizes_words)
    return {
        'num_chunks': num_chunks,
        'avg_chunk_size_chars': avg_chunk_size_chars,
        'min_chunk_size_chars': min_chunk_size_chars,
        'max_chunk_size_chars': max_chunk_size_chars,
        'avg_chunk_size_words': avg_chunk_size_words,
        'min_chunk_size_words': min_chunk_size_words,
        'max_chunk_size_words': max_chunk_size_words,
    }
