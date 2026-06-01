"""
A collection of chunking and text statistics utilities for retrieval-augmented generation (RAG).

Includes:
- Fixed-size chunker with configurable overlap
- Whitespace chunker
- Word, sentence, paragraph, line, character, token counters
- Chunk statistics summarizer

This module serves as the base for more advanced chunking strategies (recursive, semantic) and is designed for simple, readable experimentation on text splitting approaches.
"""
from dataclasses import dataclass
from typing import Iterator, List, Dict

@dataclass
class Chunk:
    text: str
    start: int
    end: int


def fixed_size_chunks(text: str, size: int = 512, overlap: int = 128) -> Iterator[Chunk]:
    """
    Split text into fixed-size chunks with optional overlap.

    Args:
        text (str): Input text to chunk.
        size (int, optional): Maximum length of each chunk (in characters). Default is 512.
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
    Split text into chunks by contiguous whitespace (e.g. paragraphs, blocks).
    Returns list of Chunk objects with text and character indices.

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
        start = text.find(seg, idx)
        end = start + len(seg)
        chunks.append(Chunk(text=seg, start=start, end=end))
        idx = end
    return chunks


def count_words(text: str) -> int:
    """
    Count the number of words in a text string.
    """
    return len(text.split())


def count_sentences(text: str) -> int:
    """
    Count the number of sentences in a text string.
    A sentence is defined as ending with '.', '!', or '?'.
    """
    import re
    if not text.strip():
        return 0
    # Matches sentences ending with ., !, or ?
    sentences = re.findall(r'[^.!?]+[.!?]', text)
    return len(sentences)


def count_characters(text: str) -> int:
    """
    Count the number of characters in a text string.
    """
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
    This is a rough proxy for true model tokens.
    """
    return len(text.strip().split()) if text.strip() else 0


def chunk_stats(chunks: List[Chunk]) -> Dict[str, float]:
    """
    Compute summary statistics for a list of chunks:
    - num_chunks
    - avg_chunk_size_chars
    - min_chunk_size_chars
    - max_chunk_size_chars
    - avg_chunk_word_count
    - min_chunk_word_count
    - max_chunk_word_count

    Args:
        chunks (List[Chunk]): List of chunk objects
    Returns:
        Dict[str, float]: Statistics summary
    """
    if not chunks:
        return {
            'num_chunks': 0,
            'avg_chunk_size_chars': 0,
            'min_chunk_size_chars': 0,
            'max_chunk_size_chars': 0,
            'avg_chunk_word_count': 0,
            'min_chunk_word_count': 0,
            'max_chunk_word_count': 0,
        }
    sizes = [len(c.text) for c in chunks]
    word_counts = [count_words(c.text) for c in chunks]
    return {
        'num_chunks': len(chunks),
        'avg_chunk_size_chars': sum(sizes) / len(sizes),
        'min_chunk_size_chars': min(sizes),
        'max_chunk_size_chars': max(sizes),
        'avg_chunk_word_count': sum(word_counts) / len(word_counts),
        'min_chunk_word_count': min(word_counts),
        'max_chunk_word_count': max(word_counts),
    }


def split_whitespace(text: str) -> List[str]:
    """
    Split text into segments by contiguous whitespace.
    Returns list of non-empty segments.
    """
    import re
    return [s for s in re.split(r'\s+', text.strip()) if s]


def word_frequencies(text: str) -> Dict[str, int]:
    """
    Count frequency of each word in the text (case-insensitive).
    Returns a dict mapping words to their counts.
    """
    import re
    words = re.findall(r'\w+', text.lower())
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    return freq
