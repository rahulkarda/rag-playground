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


def count_lowercase(text):
    """
    Count the number of lowercase letters in a string.
    Args:
        text (str): Input string
    Returns:
        int: Number of lowercase letters
    Example:
        >>> count_lowercase('abcDEF')
        3
    """
    if not text:
        return 0
    return sum(1 for c in text if c.islower())


def batch_count_lowercase(texts):
    """
    Count the number of lowercase letters in each string in a batch.
    Args:
        texts (list of str): List of input strings.
    Returns:
        list of int: Number of lowercase letters for each string.
    Example:
        >>> batch_count_lowercase(['abcDEF', 'ALLUP', 'mixedCase'])
        [3, 0, 5]
    """
    return [count_lowercase(t) if t is not None else 0 for t in texts]


def count_uppercase(text):
    """
    Count the number of uppercase letters in a string.
    Args:
        text (str): Input string
    Returns:
        int: Number of uppercase letters
    Example:
        >>> count_uppercase('abcDEF')
        3
    """
    if not text:
        return 0
    return sum(1 for c in text if c.isupper())


def batch_count_uppercase(texts):
    """
    Count the number of uppercase letters in each string in a batch.
    Args:
        texts (list of str): List of input strings.
    Returns:
        list of int: Number of uppercase letters for each string.
    Example:
        >>> batch_count_uppercase(['abcDEF', 'ALLUP', 'mixedCase'])
        [3, 5, 1]
    """
    return [count_uppercase(t) if t is not None else 0 for t in texts]


def count_whitespace(text):
    """
    Count the number of whitespace characters in a string.
    Args:
        text (str): Input string
    Returns:
        int: Number of whitespace characters (spaces, tabs, newlines, etc)
    Example:
        >>> count_whitespace('a b\tc\n')
        3
    """
    if not text:
        return 0
    return sum(1 for c in text if c.isspace())


def batch_count_whitespace(texts):
    """
    Count the number of whitespace characters in each string in a batch.
    Args:
        texts (list of str): List of input strings.
    Returns:
        list of int: Number of whitespace characters for each string.
    Example:
        >>> batch_count_whitespace(['a b', '\t\n ', 'None'])
        [1, 3, 0]

    Each element in the returned list corresponds to the count of whitespace characters
    (spaces, tabs, newlines, etc.) in each input string. If an input is None, returns 0.
    """
    return [count_whitespace(t) if t is not None else 0 for t in texts]

...
