'''
CLI entrypoint for rag-playground evaluation harness.

Delegates to src.eval_cli.main().

Usage:
    python -m src.main --input questions.jsonl --output results.jsonl [--faithfulness]

See src/eval_cli.py for usage details.
'''
import sys
from src import eval_cli

if __name__ == "__main__":
    # Delegate to eval_cli main if called as CLI
    eval_cli.main()
