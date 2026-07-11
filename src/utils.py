"""
Utilities for chunking, retrieval, and text normalization.

Provides:
- flatten: flatten nested lists (one level)
- batch_flatten: flatten each element in a batch (list of lists)
- normalize_text: lowercase, strip, collapse whitespace
- batch_normalize_text: batch text normalization
- count_tokens: whitespace token counter
- batch_count_tokens: batch token counting
- batch_strip: batch whitespace removal
- batch_is_empty: batch empty-checking utility
- batch_count_words: batch word counting utility
- batch_count_sentences: batch sentence counting utility
- batch_count_characters: batch character counting utility
- batch_count_paragraphs: batch paragraph counting utility
- batch_count_lines: batch line counting utility

Batch utilities:
- All batch_* functions operate on lists and return lists, for easy mapping in chunking/retrieval pipelines.
- batch_flatten: applies flatten to each element (not full batch flatten)
- batch_normalize_text: maps normalize_text
- batch_count_tokens: maps count_tokens
- batch_count_words: maps count_words
- batch_count_sentences: maps count_sentences
- batch_count_characters: maps count_characters
- batch_count_paragraphs: maps count_paragraphs
- batch_count_lines: maps count_lines
- batch_strip: removes whitespace from each string
- batch_is_empty: checks if each string in batch is empty or whitespace

Designed to support chunking and retrieval pipelines in RAG experiments.
"""

def flatten(lst):
    """
    Flatten a nested list (one level).

    Args:
        lst (list): list of lists or elements
    Returns:
        list: flattened list
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
    Flatten each element in a batch (one level if it's a list, else unchanged).
    Args:
        lists (Iterable): Iterable of lists or elements
    Returns:
        list: Each element flattened one level if it's a list, else unchanged
    Example:
        >>> batch_flatten([[1, 2], [3], 4])
        [[1, 2], [3], 4]  # only flattens each element
        >>> batch_flatten([[1, 2], [3, 4], [5]])
        [[1, 2], [3, 4], [5]]
    """
    return [flatten(el) if isinstance(el, list) else el for el in lists]


def normalize_text(text):
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
        texts (list of str): List of input strings
    Returns:
        list of str: Normalized strings
    """
    return [normalize_text(t) for t in texts]


def batch_strip(texts):
    """
    Remove leading/trailing whitespace from each text in a batch.
    Args:
        texts (list of str): List of input strings.
    Returns:
        list of str: Stripped strings.
    """
    return [t.strip() if isinstance(t, str) and t is not None else t for t in texts]


def batch_is_empty(texts):
    """
    Check if each text in a batch is empty or only whitespace.
    Args:
        texts (list of str): List of input strings.
    Returns:
        list of bool: True if empty or only whitespace, else False for each text.
    Example:
        >>> batch_is_empty(["", "   ", "hello", None])
        [True, True, False, True]
    """
    return [not (t and str(t).strip()) for t in texts]


def count_tokens(text):
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
        texts (list of str): List of input strings.
    Returns:
        list of int: Token counts for each text.
    """
    return [count_tokens(t) if t is not None else 0 for t in texts]


def count_words(text):
    """
    Count the number of words in a text string using whitespace splitting.
    Args:
        text (str): Input string
    Returns:
        int: Number of words
    """
    if not text or not text.strip():
        return 0
    return len(text.strip().split())


def batch_count_words(texts):
    """
    Count words for a batch of texts using simple whitespace splitting.
    Args:
        texts (list of str): List of input strings.
    Returns:
        list of int: Word counts for each text.
    Example:
        >>> batch_count_words(["hello world", "one two three", "   "])
        [2, 3, 0]
    """
    return [count_words(t) if t is not None else 0 for t in texts]


def count_sentences(text):
    """
    Count the number of sentences in a text string.
    A sentence ends with '.', '!', or '?'. Trailing content without terminal punctuation is counted as one sentence.
    Args:
        text (str): Input string
    Returns:
        int: Number of sentences
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


def batch_count_sentences(texts):
    """
    Count sentences for a batch of texts.
    Args:
        texts (list of str): List of input strings.
    Returns:
        list of int: Sentence counts for each text.
    Example:
        >>> batch_count_sentences(["This is one. That is two!", "Sentence", "", None])
        [2, 1, 0, 0]
    """
    return [count_sentences(t) if t is not None else 0 for t in texts]


def count_characters(text):
    """
    Count the number of characters in a text string.
    Args:
        text (str): Input string
    Returns:
        int: Number of characters
    """
    if not text:
        return 0
    return len(text)


def batch_count_characters(texts):
    """
    Count characters for a batch of texts.
    Args:
        texts (list of str): List of input strings.
    Returns:
        list of int: Character counts for each text.
    Example:
        >>> batch_count_characters(["abc", "", None, " "])
        [3, 0, 0, 1]
    """
    return [count_characters(t) if t is not None else 0 for t in texts]


def count_paragraphs(text):
    """
    Count the number of paragraphs in a text string.
    A paragraph is a block separated by one or more blank lines.
    Args:
        text (str): Input string
    Returns:
        int: Number of paragraphs
    """
    if not text or not str(text).strip():
        return 0
    paragraphs = [p for p in str(text).split('\n\n') if p.strip()]
    return len(paragraphs)


def batch_count_paragraphs(texts):
    """
    Count paragraphs for a batch of texts.
    Args:
        texts (list of str): List of input strings.
    Returns:
        list of int: Paragraph counts for each text.
    Example:
        >>> batch_count_paragraphs(["Para 1\n\nPara 2", "One paragraph", "\n\n\n", ""])
        [2, 1, 0, 0]
    """
    return [count_paragraphs(t) if t is not None else 0 for t in texts]


def count_lines(text):
    """
    Count the number of lines in a text string.
    A line is any sequence of characters separated by a newline ('\n').
    Args:
        text (str): Input string
    Returns:
        int: Number of lines
    Example:
        >>> count_lines("a\nb\nc")
        3
        >>> count_lines("")
        0
    """
    if not text or not str(text).strip():
        return 0
    return len(str(text).splitlines())


def batch_count_lines(texts):
    """
    Count lines for a batch of texts.
    Args:
        texts (list of str): List of input strings.
    Returns:
        list of int: Line counts for each text.
    Example:
        >>> batch_count_lines(["a\nb\nc", "one line", "", None])
        [3, 1, 0, 0]
    """
    return [count_lines(t) if t is not None else 0 for t in texts]
