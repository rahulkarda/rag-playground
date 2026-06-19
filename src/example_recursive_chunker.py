from src.recursive_chunker import recursive_chunks

if __name__ == "__main__":
    text = (
        "# Introduction\n"
        "This is the first paragraph.\n\n"
        "## Methods\n"
        "Here we describe our approach.\n\n"
        "```python\ndef foo():\n    return 'bar'\n```\n\n"
        "Results are discussed here.\n\n"
        "Conclusions and future work."
    )
    chunks = list(recursive_chunks(text, max_size=80, min_size=20))
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: [{chunk.start}:{chunk.end}]\n'{chunk.text.strip()}'\n")
