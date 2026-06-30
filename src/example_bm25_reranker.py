from src.reranker import BM25Reranker

if __name__ == "__main__":
    # Example candidates
    candidates = [
        {"text": "Retrieval-augmented generation leverages external knowledge."},
        {"text": "BM25 is a sparse keyword matching algorithm."},
        {"text": "Dense retrievers use embeddings for semantic search."},
        {"text": "Chunking text improves retrieval granularity."},
    ]
    query = "What is BM25?"
    reranker = BM25Reranker(k=2)
    results = reranker.rerank(query, candidates)
    print("Top reranked candidates:")
    for item in results:
        print(f"Score: {item['bm25_score']:.2f} | Text: {item['text']}")
