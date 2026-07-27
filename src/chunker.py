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


def fixed_size_chunks(text: str, size: int = 2048, overlap: int = 128) -> Iterator[Chunk]:
    """
    Split text into fixed-size chunks with optional overlap.

    Args:
        text (str): Input text to chunk.
        size (int, optional): Maximum length of each chunk (in characters). Default is 2048.
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


def batch_count_words(texts: List[str]) -> List[int]:
    """
    Count the number of words in each string in a batch.
    Args:
        texts (list of str): List of input strings.
    Returns:
        list of int: Number of words for each string.
    Example:
        >>> batch_count_words(["This is one.", "Two words", ""])
        [3, 2, 0]
    """
    return [count_words(t) if t is not None else 0 for t in texts]


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


def batch_count_sentences(texts: List[str]) -> List[int]:
    """
    Count the number of sentences in each string in a batch.
    Args:
        texts (list of str): List of input strings.
    Returns:
        list of int: Number of sentences for each string.
    Example:
        >>> batch_count_sentences(["This is one. This is two!", "No punctuation", "Question?"])
        [2, 1, 1]
    """
    return [count_sentences(t) if t is not None else 0 for t in texts]


def count_paragraphs(text: str) -> int:
    """
    Count the number of paragraphs (split by blank lines).
    """
    if not text or not text.strip():
        return 0
    paras = [p for p in text.split('\n\n') if p.strip()]
    return len(paras)


def batch_count_paragraphs(texts: List[str]) -> List[int]:
    """
    Count the number of paragraphs in each string in a batch.
    """
    return [count_paragraphs(t) if t is not None else 0 for t in texts]


def count_lines(text: str) -> int:
    """
    Count the number of lines in a text string.
    """
    if not text:
        return 0
    return len(text.split('\n'))


def batch_count_lines(texts: List[str]) -> List[int]:
    """
    Count the number of lines in each string in a batch.
    """
    return [count_lines(t) if t is not None else 0 for t in texts]


def count_tokens(text: str) -> int:
    """
    Count the number of tokens (space-separated) in a text string.
    """
    if not text:
        return 0
    return len(text.split())


def batch_count_tokens(texts: List[str]) -> List[int]:
    """
    Count the number of tokens in each string in a batch.
    """
    return [count_tokens(t) if t is not None else 0 for t in texts]


def count_tiktoken_tokens(text: str, model: str = "gpt2") -> int:
    """
    Count the number of tokens using tiktoken tokenizer for a given model.
    Args:
        text (str): Input text
        model (str): Model name (default: "gpt2")
    Returns:
        int: Number of tokens
    """
    import tiktoken
    enc = tiktoken.get_encoding(model)
    return len(enc.encode(text))


def batch_count_tiktoken_tokens(texts: List[str], model: str = "gpt2") -> List[int]:
    """
    Count tiktoken tokens for each string in a batch.
    """
    import tiktoken
    enc = tiktoken.get_encoding(model)
    return [len(enc.encode(t)) if t is not None else 0 for t in texts]


def chunk_stats(chunks: List[Chunk]) -> Dict[str, float]:
    """
    Summarize statistics for a set of chunks.
    Returns:
        dict with num_chunks, avg/min/max chunk size (chars, words)
    """
    if not chunks:
        return {
            'num_chunks': 0,
            'avg_chunk_size_chars': 0,
            'min_chunk_size_chars': 0,
            'max_chunk_size_chars': 0,
            'avg_chunk_size_words': 0,
            'min_chunk_size_words': 0,
            'max_chunk_size_words': 0,
        }
    num_chunks = len(chunks)
    chunk_sizes_chars = [len(c.text) for c in chunks]
    chunk_sizes_words = [count_words(c.text) for c in chunks]
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
