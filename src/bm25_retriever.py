from rank_bm25 import BM25Okapi
from typing import List, Dict, Optional, Any
import numpy as np

class BM25Retriever:
    """
    Wrapper for BM25 sparse keyword retrieval using rank_bm25.
    Supports fitting on a corpus and retrieving top-k docs for a query.
    """
    def __init__(self, corpus: List[str], ids: Optional[List[int]] = None, metadata: Optional[List[Dict[str, Any]]] = None):
        """
        Args:
            corpus: List of document strings
            ids: Optional list of integer ids for docs
            metadata: Optional list of dicts per doc
        """
        self.corpus = corpus
        self.tokenized = [self._tokenize(doc) for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized)
        self.ids = ids if ids is not None else list(range(len(corpus)))
        self.metadata = metadata if metadata is not None else [{} for _ in self.ids]

    def _tokenize(self, text: str) -> List[str]:
        # Simple whitespace tokenizer
        return text.lower().split()

    def normalize_query(self, query: str) -> str:
        """
        Normalize query text for retrieval: lowercase, strip, collapse whitespace.
        """
        import re
        query = query.lower()
        query = query.strip()
        query = re.sub(r'\s+', ' ', query)
        return query

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top-k docs for a query using BM25.
        Returns a list of dicts: {id, score, text, metadata}
        """
        query = self.normalize_query(query)
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        top_idx = np.argsort(scores)[::-1][:k]
        results = []
        for idx in top_idx:
            results.append({
                "id": self.ids[idx],
                "score": float(scores[idx]),
                "text": self.corpus[idx],
                "metadata": self.metadata[idx]
            })
        return results
