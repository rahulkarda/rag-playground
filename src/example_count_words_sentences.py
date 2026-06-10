from src.chunker import count_words, count_sentences

if __name__ == "__main__":
    text = "This is a test. Another sentence! Is this counted? Yes."
    print(f"Text: {repr(text)}")
    print(f"Word count: {count_words(text)}")
    print(f"Sentence count: {count_sentences(text)}")
