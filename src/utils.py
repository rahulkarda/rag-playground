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
