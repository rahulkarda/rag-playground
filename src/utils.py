...

def count_substring(text, substring):
    """
    Count the number of times a substring appears in a text string.
    Args:
        text (str): Input string
        substring (str): Substring to count
    Returns:
        int: Number of occurrences
    Example:
        >>> count_substring('banana', 'an')
        2
    """
    if not text or not substring:
        return 0
    return str(text).count(substring)


def batch_count_substring(texts, substring):
    """
    Count occurrences of a substring for each text in a batch.
    Args:
        texts (list of str): List of input strings.
        substring (str): Substring to count.
    Returns:
        list of int: Number of occurrences for each text.
    Example:
        >>> batch_count_substring(['banana', 'bandana', 'an'], 'an')
        [2, 2, 1]
    """
    return [count_substring(t, substring) if t is not None else 0 for t in texts]

...