from src.bm25_retriever import BM25Retriever

if __name__ == "__main__":
    corpus = [
        "Retrieval-augmented generation uses search and LLMs.",
        "BM25 is a strong keyword-based baseline.",
        "Dense retrieval relies on embeddings.",
        "Chunking affects retrieval granularity.",
    ]
    retriever = BM25Retriever(corpus)
    query = "keyword search baseline"
    results = retriever.search(query, k=2)
    print(f"Query: '{query}'\n")
    for i, res in enumerate(results):
        print(f"Rank {i+1} | Score: {res['score']:.2f}\n{res['text']}\n")
