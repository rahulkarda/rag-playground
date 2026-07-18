from src.utils import batch_count_newlines

if __name__ == "__main__":
    texts = [
        "abc",
        "a\nb\nc",
        "\n",
        "",
        "two\nlines\n",
        None,
    ]
    counts = batch_count_newlines(texts)
    print(f"batch_count_newlines({texts}) -> {counts}")
