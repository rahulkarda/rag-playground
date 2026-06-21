from src.embedder import SentenceTransformerEmbedder

if __name__ == "__main__":
    embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
    queries = [
        "What is retrieval-augmented generation?",
        "How do embeddings work?",
    ]
    docs = [
        "Retrieval-augmented generation (RAG) combines search and language models.",
        "Embeddings are numeric representations of text.",
        "Chunking strategies affect retrieval quality."
    ]
    sims = embedder.embed_batch_cosine_similarity(queries, docs)
    print("Cosine similarity matrix:")
    for i, query in enumerate(queries):
        for j, doc in enumerate(docs):
            print(f"Query {i} vs Doc {j}: {sims[i, j]:.4f}")
