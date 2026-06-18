from src.chunker import chunk_whitespace

if __name__ == "__main__":
    text = "This   is a test.\nNewline here.\tTab here."
    chunks = chunk_whitespace(text)
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: [{chunk.start}:{chunk.end}] '{chunk.text}'")
