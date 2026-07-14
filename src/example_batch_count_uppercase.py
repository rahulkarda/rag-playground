from src.utils import batch_count_uppercase

if __name__ == "__main__":
    texts = ["Hello World!", "UPPER lower", "abc", "", None]
    counts = batch_count_uppercase(texts)
    print(f"batch_count_uppercase({texts}) -> {counts}")
