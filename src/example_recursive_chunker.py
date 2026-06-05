from src.recursive_chunker import recursive_chunks

if __name__ == "__main__":
    text = (
        "# Header 1\n"
        "Paragraph one. Sentence two!\n\n"
        "## Header 2\n"
        "Code block:\n"
        "```python\nprint('Hello')\n```\n\n"
        "Another paragraph. With more sentences. And yet more.\n"
    )
    chunks = list(recursive_chunks(text, max_size=50, min_size=10))
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: [{chunk.start}:{chunk.end}]\n'{chunk.text.strip()}'\n")
