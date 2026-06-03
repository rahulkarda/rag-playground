from src.semantic_chunker import semantic_chunks
import numpy as np

# Dummy embed function: random embedding for demonstration

def dummy_embed_fn(text):
    np.random.seed(hash(text) % 2**32)  # deterministic per text
    return np.random.rand(384)

if __name__ == "__main__":
    text = (
        "The quick brown fox jumps over the lazy dog. "
        "This sentence is similar to the previous one. "
        "However, the next sentence talks about something else. "
        "Deep learning models have transformed natural language processing. "
        "Chunking is important in retrieval augmented generation."
    )
    chunks = list(semantic_chunks(
        text,
        embed_fn=dummy_embed_fn,
        similarity_threshold=0.5,
        min_size=40,
        max_size=120
    ))
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: [{chunk.start}:{chunk.end}]\n'{chunk.text.strip()}'\n")
