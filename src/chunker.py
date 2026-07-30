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

def batch_count_uppercase(texts: List[str]) -> List[int]:
    """
    Count the number of uppercase letters in each string in a batch.
    Args:
        texts (list of str): List of input strings.
    Returns:
        list of int: Number of uppercase letters for each string.
    Example:
        >>> batch_count_uppercase(['abcDEF', 'ALLUP', 'mixedCase', '', None])
        [3, 5, 1, 0, 0]
    """
    from src.utils import count_uppercase
    return [count_uppercase(t) if t is not None else 0 for t in texts]

def batch_count_lowercase(texts: List[str]) -> List[int]:
    """
    Count the number of lowercase letters in each string in a batch.
    Args:
        texts (list of str): List of input strings.
    Returns:
        list of int: Number of lowercase letters for each string.
    Example:
        >>> batch_count_lowercase(['abcDEF', 'ALLUP', 'mixedCase', '', None])
        [3, 0, 5, 0, 0]
    """
    from src.utils import count_lowercase
    return [count_lowercase(t) if t is not None else 0 for t in texts]
...