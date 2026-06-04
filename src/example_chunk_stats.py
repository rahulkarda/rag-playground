from src.chunker import fixed_size_chunks, chunk_stats

if __name__ == "__main__":
    text = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    chunks = list(fixed_size_chunks(text, size=10, overlap=2))
    stats = chunk_stats(chunks)
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: [{chunk.start}:{chunk.end}] '{chunk.text}'")
    print("\nChunk stats:")
    for k, v in stats.items():
        print(f"{k}: {v}")
