from src.chunker import fixed_size_chunks

if __name__ == "__main__":
    text = "abcdefghijklmnopqrstuvwxyz"
    chunks = list(fixed_size_chunks(text, size=6, overlap=2))
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: [{chunk.start}:{chunk.end}] '{chunk.text}'")
