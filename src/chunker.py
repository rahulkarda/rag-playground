"""Fixed-size text chunker. Starting point — recursive and semantic chunkers come next."""
from dataclasses import dataclass
from typing import Iterator


@dataclass
class Chunk:
    text: str
    start: int
    end: int


def fixed_size_chunks(text: str, size: int = 512, overlap: int = 64) -> Iterator[Chunk]:
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    step = size - overlap
    pos = 0
    while pos < len(text):
        end = min(pos + size, len(text))
        yield Chunk(text=text[pos:end], start=pos, end=end)
        if end == len(text):
            break
        pos += step
