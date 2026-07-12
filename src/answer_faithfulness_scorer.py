"""
Faithfulness scorer utility for evaluating answer faithfulness against context.

This module provides:
- compute_faithfulness_score: returns a score [0,1] indicating the extent to which the answer is supported/grounded in the provided context.

The method is heuristic: it checks for factual consistency by comparing n-grams and key phrases, penalizing hallucinated or unsupported claims. For demonstration, we use a simple overlap metric, but this can be replaced with a model-based approach (e.g. LLM scoring).
"""
import re
from typing import List, Dict, Any

def normalize_text(text: str) -> str:
    """
    Lowercase, strip, collapse whitespace. Simple normalization for comparison.
    """
    text = text.lower()
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def extract_key_phrases(text: str, ngram_sizes: List[int] = [2, 3]) -> List[str]:
    """
    Extract key phrases (n-grams) from text for overlap scoring.
    """
    tokens = [t for t in re.split(r'\W+', text) if t]
    phrases = set()
    for n in ngram_sizes:
        for i in range(len(tokens) - n + 1):
            phrase = ' '.join(tokens[i:i+n])
            phrases.add(phrase)
    return list(phrases)

def compute_faithfulness_score(answer: str, context: str) -> float:
    """
    Compute a faithfulness score for answer given context.
    Returns a float in [0,1]: 1.0 means fully grounded, 0.0 means unsupported/hallucinated.

    Approach:
    - Extract n-grams/key phrases from answer and context
    - Score overlap as fraction of answer phrases found in context
    - Penalize for low overlap
    """
    answer_norm = normalize_text(answer)
    context_norm = normalize_text(context)
    answer_phrases = extract_key_phrases(answer_norm)
    context_phrases = set(extract_key_phrases(context_norm))
    if not answer_phrases:
        return 1.0 if not answer.strip() else 0.0
    matched = sum(1 for phrase in answer_phrases if phrase in context_phrases)
    score = matched / len(answer_phrases)
    return round(score, 4)

# Example utility for batch scoring
def batch_faithfulness_score(answers: List[str], contexts: List[str]) -> List[float]:
    """
    Compute faithfulness scores for a batch of answers and contexts.
    """
    return [compute_faithfulness_score(a, c) for a, c in zip(answers, contexts)]

if __name__ == "__main__":
    # Demo usage
    ans = "The capital of France is Paris."
    ctx = "Paris is the capital city of France."
    score = compute_faithfulness_score(ans, ctx)
    print(f"Faithfulness score: {score}")

    ans2 = "Berlin is the capital of France."
    score2 = compute_faithfulness_score(ans2, ctx)
    print(f"Faithfulness score: {score2}")
