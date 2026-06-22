from src.faiss_index import FaissIndex
import numpy as np
import os

if __name__ == "__main__":
    # Create dummy embeddings
    dim = 12
    num_vecs = 6
    vectors = np.random.rand(num_vecs, dim).astype(np.float32)
    # Build FAISS index
    index = FaissIndex(dim, index_factory="Flat")
    index.add(vectors)
    # Save index to disk
    path = "faiss_test.index"
    index.save(path)
    print(f"Index saved to {path}")
    # Load index back
    loaded_index = FaissIndex.load(path)
    print(f"Loaded index dim: {loaded_index.dim}")
    # Query with one of the vectors
    query = vectors[0]
    D, I = loaded_index.search(query, k=3)
    print("Query vector:", query)
    print("Distances:", D)
    print("Indices:", I)
    # Clean up
    os.remove(path)
    print(f"Deleted {path}")
