from src.faiss_index import FaissIndex
import numpy as np

if __name__ == "__main__":
    # Create dummy embeddings
    dim = 8
    vectors = np.random.rand(5, dim).astype(np.float32)
    # Build FAISS index
    index = FaissIndex(dim, index_factory="Flat")
    index.add(vectors)
    # Query with one of the vectors
    query = vectors[2]
    D, I = index.search(query, k=3)
    print("Query vector:", query)
    print("Distances:", D)
    print("Indices:", I)
