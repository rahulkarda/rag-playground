from src.chunker import normalize_text

if __name__ == "__main__":
    text = "  This   is a Test.\n\nSecond   Line!\t  "
    print("Original:")
    print(repr(text))
    print("\nNormalized:")
    norm = normalize_text(text)
    print(repr(norm))
