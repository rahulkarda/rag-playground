from src.utils import flatten, normalize_text, count_tokens, batch_count_tokens, batch_strip, batch_is_empty

if __name__ == "__main__":
    # Test flatten
    nested = [[1, 2], [3], 4]
    flat = flatten(nested)
    print(f"flatten([[1, 2], [3], 4]) -> {flat}")

    # Test normalize_text
    messy = "  This\tIS   Some  TEXT\n"
    normalized = normalize_text(messy)
    print(f"normalize_text('{messy}') -> '{normalized}'")

    # Test count_tokens
    text = "hello world, this is a test"
    num_tokens = count_tokens(text)
    print(f"count_tokens('{text}') -> {num_tokens}")

    # Test batch_count_tokens
    texts = ["first line", "second line with more words", "   "]
    batch_counts = batch_count_tokens(texts)
    print(f"batch_count_tokens({texts}) -> {batch_counts}")

    # Test batch_strip
    batch_texts = ["  leading", "trailing  ", " both ", "no_whitespace", "\t tabbed\n"]
    stripped = batch_strip(batch_texts)
    print(f"batch_strip({batch_texts}) -> {stripped}")

    # Test batch_is_empty
    empty_batch = ["", "   ", "not empty", None, "\n"]
    is_empty = batch_is_empty(empty_batch)
    print(f"batch_is_empty({empty_batch}) -> {is_empty}")
