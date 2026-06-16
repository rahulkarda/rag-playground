import faiss
import numpy as np
from typing import Optional, List

class FaissIndex:
    """
    Wrapper for a local FAISS index for dense vector retrieval.
    Supports adding vectors, searching for nearest neighbors, and persistence.
    """
    def __init__(self, dim: int, index_factory: str = "Flat", use_gpu: bool = False):
        """
        Args:
            dim: embedding dimension
            index_factory: FAISS index type (e.g. "Flat", "IVF32,Flat")
            use_gpu: whether to use GPU (default False)
        """
        self.dim = dim
        self.index_factory = index_factory
        self.use_gpu = use_gpu
        self.index = self._create_index()
        self.ids = []  # track ids for lookup

    def _create_index(self):
        index = faiss.index_factory(self.dim, self.index_factory)
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        return index

    def add(self, vectors: np.ndarray, ids: Optional[List[int]] = None):
        """
        Add vectors to the index.
        Args:
            vectors: numpy array of shape (num_vectors, dim)
            ids: optional list of ids (must be int)
        """
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if ids is not None:
            ids_arr = np.array(ids, dtype=np.int64)
            self.index.add_with_ids(vectors, ids_arr)
            self.ids.extend(ids)
        else:
            self.index.add(vectors)
            self.ids.extend([len(self.ids) + i for i in range(vectors.shape[0])])

    def search(self, query: np.ndarray, k: int = 5):
        """
        Search for k nearest neighbors for query vector(s).
        Args:
            query: numpy array of shape (dim,) or (num_queries, dim)
            k: number of neighbors
        Returns:
            (distances, indices): arrays of shape (num_queries, k)
        """
        query = np.ascontiguousarray(query, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        D, I = self.index.search(query, k)
        return D, I

    def save(self, path: str):
        """
        Save the FAISS index to disk.
        """
        faiss.write_index(self.index, path)

    @classmethod
    def load(cls, path: str, use_gpu: bool = False):
        """
        Load a FAISS index from disk.
        """
        index = faiss.read_index(path)
        if use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        dim = index.d
        instance = cls(dim, use_gpu=use_gpu)
        instance.index = index
        return instance
