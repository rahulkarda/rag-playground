import argparse
import json
from src.answer_relevance_scorer import answer_relevance_score
from src.answer_faithfulness_scorer import answer_faithfulness_score


def load_questions(path):
    """
    Load questions and ground-truth contexts from a JSONL file.
    Each line: {"question": str, "answer": str, "ground_truth": str}
    """
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items

def main():
    parser = argparse.ArgumentParser(description="RAG Evaluation CLI: answer-relevance & faithfulness scoring")
    parser.add_argument('--input', '-i', required=True,
                        help='Path to JSONL file with question, answer, ground_truth')
    parser.add_argument('--output', '-o', required=True,
                        help='Path to write scored results (JSONL)')
    parser.add_argument('--faithfulness', action='store_true',
                        help='Include answer faithfulness scoring')
    args = parser.parse_args()

    items = load_questions(args.input)
    results = []
    for idx, item in enumerate(items):
        q = item.get('question', '')
        ans = item.get('answer', '')
        gt = item.get('ground_truth', '')
        rel_score = answer_relevance_score(q, ans, gt)
        out = {
            'question': q,
            'answer': ans,
            'ground_truth': gt,
            'relevance_score': rel_score
        }
        if args.faithfulness:
            faith_score = answer_faithfulness_score(ans, gt)
            out['faithfulness_score'] = faith_score
        results.append(out)
    with open(args.output, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    print(f"Scored {len(results)} examples. Output written to {args.output}")

if __name__ == '__main__':
    main()
