"""Fixed-size text chunker. Starting point  ð recursive and semantic chunkers come next."""
from dataclasses import dataclass
from typing import Iterator


@dataclass
class Chunk:
    text: str
    start: int
    end: int


def fixed_size_chunks(text: str, size: int = 512, overlap: int = 64) -> Iterator[Chunk]:
    """
    Split text into fixed-size chunks with optional overlap.

    Args:
        text (str): Input text to chunk.
        size (int, optional): Maximum length of each chunk (in characters). Default is 512.
        overlap (int, optional): Number of characters to overlap between consecutive chunks. Default is 64.

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
