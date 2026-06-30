from typing import Iterator, Dict, Any

class StreamingAnswerGenerator:
    """
    StreamingAnswerGenerator: wrapper for LLMs that streams token-by-token output.
    
    Usage:
        gen = StreamingAnswerGenerator(provider_fn)
        for token in gen.generate_stream(prompt):
            print(token, end="")

    provider_fn must yield tokens as strings.
    """
    def __init__(self, provider_fn):
        """
        Args:
            provider_fn: function(prompt: str, **kwargs) -> Iterator[str]. Must yield tokens.
        """
        self.provider_fn = provider_fn

    def generate_stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """
        Generate answer, streaming tokens as they arrive.
        Args:
            prompt (str): input prompt
            **kwargs: additional provider arguments
        Yields:
            str: next token
        """
        for token in self.provider_fn(prompt, **kwargs):
            yield token

# Example provider_fn stub for testing
def stub_provider_fn(prompt: str, **kwargs) -> Iterator[str]:
    # Simulate streaming a response
    answer = "This is a streamed answer."
    for token in answer.split():
        yield token + " "

if __name__ == "__main__":
    gen = StreamingAnswerGenerator(stub_provider_fn)
    print("Streaming answer:")
    for token in gen.generate_stream("What is RAG?"):
        print(token, end="")
    print()
