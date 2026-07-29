"""
BM25 reranking utility for retrieval-augmented generation (RAG).

This module provides:
- BM25Reranker: reranks a list of candidate documents given a query
- Useful as a reranking stage after initial retrieval (dense, hybrid, etc)

Usage:
    from src.reranker import BM25Reranker
    reranker = BM25Reranker(k=5)
    reranked = reranker.rerank(query, candidates)
    # candidates: list of dicts with 'text' field
    # reranked: top-k candidates with added 'bm25_score'

Relies on rank_bm25 for sparse keyword matching. Designed for quick experimentation with reranking strategies.
"""
from typing import List, Dict, Any, Optional
import numpy as np

class BM25Reranker:
    """
    Prototype BM25 reranker: takes a query and a list of candidate documents,
    scores each candidate using BM25, and returns top-k reranked candidates.
    This is useful as a reranking stage after initial retrieval (dense, hybrid, etc).
    """
    def __init__(self, k: int = 5):
        self.k = k

    def _tokenize(self, text: str) -> List[str]:
        # Simple whitespace tokenizer
        return text.lower().split()

    def rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank candidates using BM25 keyword matching.
        Args:
            query (str): input query
            candidates (List[Dict]): list of dicts with 'text' field
        Returns:
            List[Dict]: top-k reranked candidates, each with added 'bm25_score'
        """
        from rank_bm25 import BM25Okapi
        docs = [cand['text'] for cand in candidates]
        tokenized_docs = [self._tokenize(doc) for doc in docs]
        bm25 = BM25Okapi(tokenized_docs)
        query_tokens = self._tokenize(query)
        scores = bm25.get_scores(query_tokens)
        top_idx = np.argsort(scores)[::-1][:self.k]
        reranked = []
        for idx in top_idx:
            item = candidates[idx].copy()
            item['bm25_score'] = float(scores[idx])
            reranked.append(item)
        return reranked
