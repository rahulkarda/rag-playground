from src.embedder import SentenceTransformerEmbedder

if __name__ == "__main__":
    text = "Retrieval-augmented generation is a powerful technique."
    embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
    emb = embedder.embed(text)
    print(f"Embedding shape: {emb.shape}")
    print(f"First 5 values: {emb[:5]}")

    batch = [
        "RAG combines search and language models.",
        "Chunking improves retrieval granularity.",
        "Embedding models capture semantic meaning."
    ]
    batch_emb = embedder.embed_batch(batch)
    print(f"Batch embedding shape: {batch_emb.shape}")
    print(f"First row first 5 values: {batch_emb[0][:5]}")
