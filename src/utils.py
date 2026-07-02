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


def normalize_text(text: str) -> str:
    """
    Normalize text for chunking/retrieval: lowercase, strip, collapse whitespace.
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


def count_tokens(text: str) -> int:
    """
    Count the number of tokens in a text string using whitespace splitting.
    Args:
        text (str): Input string
    Returns:
        int: Number of tokens
    """
    return len(text.strip().split()) if text.strip() else 0


def batch_count_tokens(texts):
    """
    Count tokens for a batch of texts using simple whitespace splitting.
    Args:
        texts (List[str]): List of input strings.
    Returns:
        List[int]: Token counts for each text.
    """
    return [count_tokens(t) for t in texts]
