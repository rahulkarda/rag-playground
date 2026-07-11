from src.utils import batch_count_lines

if __name__ == "__main__":
    texts = [
        "a\nb\nc",               # 3 lines
        "one line",             # 1 line
        "",                    # 0 lines
        None,                   # 0 lines (None)
        "first\nsecond\n",    # 2 lines
        "\n\n",               # 2 empty lines
    ]
    counts = batch_count_lines(texts)
    print(f"batch_count_lines({texts}) -> {counts}")
