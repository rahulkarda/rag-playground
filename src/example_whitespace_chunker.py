from src.chunker import chunk_whitespace

if __name__ == "__main__":
    text = "RAG combines retrieval\n\nwith language models.\tWhitespace\nchunking!"
    chunks = chunk_whitespace(text)
    print(f"Input: '{text}'\n")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: [{chunk.start}:{chunk.end}] '{chunk.text}'")
