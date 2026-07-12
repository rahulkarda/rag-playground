import numpy as np
from src.embedder import SentenceTransformerEmbedder

class AnswerRelevanceScorer:
    """
    Computes answer relevance scores for a given question/answer/context tuple.
    Uses embedding cosine similarity between answer and context, optionally also to question.

    Usage:
        scorer = AnswerRelevanceScorer()
        score = scorer.score(question, answer, context)
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformerEmbedder(model_name)

    def score(self, question: str, answer: str, context: str) -> float:
        """
        Computes a relevance score for the answer given the context and question.
        Returns float in [0, 1]: higher means more relevant.
        """
        # Embed answer and context
        answer_emb = self.embedder.embed(answer)
        context_emb = self.embedder.embed(context)
        # Cosine similarity
        sim_ac = self._cosine_sim(answer_emb, context_emb)
        # Optionally: also question-answer similarity (less useful? can average)
        question_emb = self.embedder.embed(question)
        sim_qc = self._cosine_sim(question_emb, context_emb)
        sim_qa = self._cosine_sim(question_emb, answer_emb)
        # Simple heuristic: weighted average
        score = (sim_ac * 0.7) + (sim_qa * 0.15) + (sim_qc * 0.15)
        # Map to [0, 1]
        score = max(0.0, min(1.0, (score + 1) / 2))
        return score

    def _cosine_sim(self, a, b):
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
