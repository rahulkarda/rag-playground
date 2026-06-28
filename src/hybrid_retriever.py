"""
Hybrid dense+sparse retriever using Reciprocal Rank Fusion (RRF).

This module combines dense (embedding-based) and sparse (BM25/keyword) retrieval results
by fusing their rankings. It allows flexible integration of different retriever types:
- dense_retriever: instance with .search(query_emb, k) -> List[Dict] (e.g. FAISS)
- sparse_retriever: instance with .search(query, k) -> List[Dict] (e.g. BM25)

RRF fusion scores each candidate as:
    score = sum(1 / (rrf_k + rank)) for each source
Where rank is the position in the result list. Top-k fused results are returned.

Useful for hybrid RAG pipelines to combine semantic and keyword search signals.
"""
from typing import List, Dict, Any, Optional
import numpy as np

class HybridRetriever:
    """
    Hybrid retriever using Reciprocal Rank Fusion (RRF) to combine dense and sparse retrieval results.
    - dense_retriever: instance with .search(query_emb, k) -> List[Dict]
    - sparse_retriever: instance with .search(query, k) -> List[Dict]
    - rrf_k: fusion parameter (default 60)
    - top_k: number of final results to return
    """
    def __init__(
        self,
        dense_retriever,
        sparse_retriever,
        rrf_k: int = 60,
        top_k: int = 5
    ):
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.rrf_k = rrf_k
        self.top_k = top_k

    def search(self, query: str, query_emb: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """
        Hybrid search: fuse dense and sparse results using RRF.
        Args:
            query: string (for sparse)
            query_emb: np.ndarray (for dense); if None, dense search skipped
        Returns:
            List[Dict]: fused top-k results with {id, rrf_score, metadata, source_rank_dense, source_rank_sparse}
        """
        dense_results = []
        sparse_results = []
        if query_emb is not None:
            dense_results = self.dense.search(query_emb, k=self.rrf_k)
        if query:
            sparse_results = self.sparse.search(query, k=self.rrf_k)
        all_ids = set([r["id"] for r in dense_results] + [r["id"] for r in sparse_results])
        rrf_scores = {}
        rank_dense = {r["id"]: i for i, r in enumerate(dense_results)}
        rank_sparse = {r["id"]: i for i, r in enumerate(sparse_results)}
        metadata = {}
        for r in dense_results:
            metadata[r["id"]] = r.get("metadata", None)
        for r in sparse_results:
            metadata[r["id"]] = r.get("metadata", None)
        for id in all_ids:
            rrf_score = 0.0
            if id in rank_dense:
                rrf_score += 1.0 / (self.rrf_k + rank_dense[id])
            if id in rank_sparse:
                rrf_score += 1.0 / (self.rrf_k + rank_sparse[id])
            rrf_scores[id] = rrf_score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        results = []
        for id in sorted_ids[:self.top_k]:
            results.append({
                "id": id,
                "rrf_score": rrf_scores[id],
                "metadata": metadata.get(id, None),
                "source_rank_dense": rank_dense.get(id, None),
                "source_rank_sparse": rank_sparse.get(id, None)
            })
        return results
