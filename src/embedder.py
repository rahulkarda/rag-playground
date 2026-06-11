import numpy as np
from typing import List, Union, Callable

class SentenceTransformerEmbedder:
    """
    Wrapper for sentence-transformers models. Provides embed(text) and embed_batch(texts).
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> np.ndarray:
        """
        Embed a single text. Returns a numpy array.
        """
        emb = self.model.encode([text], convert_to_numpy=True)[0]
        return emb

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Embed a batch of texts. Returns array of shape (len(texts), dim).
        """
        return self.model.encode(texts, convert_to_numpy=True)

# Factory for generic embed_fn

def load_embedder(model_name: str = "all-MiniLM-L6-v2") -> Callable[[str], np.ndarray]:
    """
    Returns a function text -> embedding using the chosen model.
    """
    embedder = SentenceTransformerEmbedder(model_name)
    return embedder.embed
