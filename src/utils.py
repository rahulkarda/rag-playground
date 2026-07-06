"""
Utilities for chunking, retrieval, and text normalization.

Provides:
- flatten: flatten nested lists
- batch_flatten: flatten each element in a batch (list of lists)
- normalize_text: lowercase, strip, collapse whitespace
- batch_normalize_text: batch text normalization
- count_tokens: whitespace token counter
- batch_count_tokens: batch token counting
- batch_strip: batch whitespace removal

Designed to support chunking and retrieval pipelines in RAG experiments.
"""
def flatten(lst):
    """
    Flatten a nested list (one level).

    Args:
        lst: list of lists or elements
    Returns:
        flat list
    Example:
        >>> flatten([[1, 2], [3], 4])
        [1, 2, 3, 4]
    """
    flat = []
    for el in lst:
        if isinstance(el, list):
            flat.extend(el)
        else:
            flat.append(el)
    return flat


def batch_flatten(lists):
    """
    Flatten each element in a batch (list of lists or elements).
    Args:
        lists: Iterable of lists or elements
    Returns:
        List: Each element flattened one level if it's a list, else unchanged
    Example:
        >>> batch_flatten([[1, 2], [3], 4])
        [[1, 2], [3], 4]  # only flattens each element
        >>> batch_flatten([[1, 2], [3, 4], [5]])
        [[1, 2], [3, 4], [5]]
    """
    return [flatten(el) if isinstance(el, list) else el for el in lists]


def normalize_text(text: str) -> str:
    """
    Normalize text for chunking/retrieval:
    - Lowercase
    - Strip leading/trailing whitespace
    - Collapse runs of whitespace to single spaces
    Args:
        text (str): Input string
    Returns:
        str: Normalized string
    """
    import re
    text = text.lower()
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def batch_normalize_text(texts):
    """
    Normalize a batch of texts for chunking/retrieval.
    Args:
        texts (List[str]): List of input strings
    Returns:
        List[str]: Normalized strings
    """
    return [normalize_text(t) for t in texts]


def count_tokens(text: str) -> int:
    """
    Count the number of tokens in a text string using whitespace splitting.
    Args:
        text (str): Input string
    Returns:
        int: Number of tokens
    """
    if not text or not text.strip():
        return 0
    return len(text.strip().split())


def batch_count_tokens(texts):
    """
    Count tokens for a batch of texts using simple whitespace splitting.
    Args:
        texts (List[str]): List of input strings.
    Returns:
        List[int]: Token counts for each text.
    """
    return [count_tokens(t) for t in texts]


def batch_strip(texts):
    """
    Remove leading/trailing whitespace from each text in a batch.
    Args:
        texts (List[str]): List of input strings.
    Returns:
        List[str]: Stripped strings.
    """
    return [t.strip() if isinstance(t, str) and t is not None else t for t in texts]
