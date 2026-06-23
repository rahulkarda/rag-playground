import numpy as np
from src.faiss_index import FaissIndex
from typing import List, Dict, Any, Optional

class DenseRetriever:
    """
    Dense retriever using FAISS for embedding-based nearest neighbor search.
    Supports fitting on a corpus of embeddings and retrieving top-k items for a query embedding.
    """
    def __init__(
        self,
        embeddings: np.ndarray,
        ids: Optional[List[int]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        index_factory: str = "IVF32,Flat",
        use_gpu: bool = False
    ):
        """
        Args:
            embeddings: numpy array of shape (num_items, dim)
            ids: optional list of ids (int)
            metadata: optional list of dicts
            index_factory: FAISS index type
            use_gpu: whether to use GPU
        """
        dim = embeddings.shape[1]
        self.index = FaissIndex(dim, index_factory=index_factory, use_gpu=use_gpu)
        self.index.add(embeddings, ids=ids, metadata=metadata)

    def search(self, query_emb: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top-k nearest items for a query embedding.
        Returns: list of dicts: {id, score, metadata}
        """
        return self.index.dense_retrieve(query_emb, k=k)

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embed_fn,
        ids: Optional[List[int]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        index_factory: str = "IVF32,Flat",
        use_gpu: bool = False,
        batch_size: int = 32
    ):
        """
        Build retriever from raw texts and embedding function.
        Args:
            texts: list of strings
            embed_fn: function(texts: List[str]) -> np.ndarray
            ids: optional list of ids
            metadata: optional list of dicts
            index_factory: FAISS index type
            use_gpu: GPU flag
            batch_size: embedding batch size
        Returns: DenseRetriever
        """
        if hasattr(embed_fn, "embed_batch"):
            embeddings = embed_fn.embed_batch(texts)
        else:
            # fallback: assume embed_fn(text: str) -> np.ndarray
            embeddings = np.vstack([embed_fn(t) for t in texts])
        return cls(
            embeddings,
            ids=ids,
            metadata=metadata,
            index_factory=index_factory,
            use_gpu=use_gpu
        )
