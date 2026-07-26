from src.embedder import SentenceTransformerEmbedder

if __name__ == "__main__":
    embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
    queries = [
        "What is retrieval-augmented generation?",
        "How do chunkers improve retrieval?"
    ]
    docs = [
        "Retrieval-augmented generation combines search with language models.",
        "Chunking splits documents into smaller pieces for better retrieval.",
        "Dense retrievers use embeddings to find similar texts."
    ]
    sims = embedder.embed_batch_cosine_similarity(queries, docs)
    print("Cosine similarity matrix:")
    print(sims)
    # Print which document is most similar to each query
    for i, q in enumerate(queries):
        best_idx = sims[i].argmax()
        print(f"Query: '{q}'\nMost similar doc: '{docs[best_idx]}'\nScore: {sims[i][best_idx]:.4f}\n")
