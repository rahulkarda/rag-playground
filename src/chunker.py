...
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
...
