"""
Provider-agnostic answer generator wrapper for RAG pipelines.

This module defines a minimal interface for generating answers from prompts, abstracting over LLM providers.
Supports synchronous generation via OpenAI, HuggingFace, or local function. Designed for easy extension.
"""
from typing import Any, Dict, Optional

class AnswerGenerator:
    """
    Provider-agnostic answer generator wrapper.
    Supports OpenAI API, HuggingFace Transformers, or custom local functions via unified interface.
    """
    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        local_fn: Optional[Any] = None,
        **kwargs
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.local_fn = local_fn
        self.kwargs = kwargs

    def generate(self, prompt: str, **gen_kwargs) -> Dict[str, Any]:
        """
        Generate answer from a prompt. Returns dict with 'answer' and optional metadata.
        """
        if self.provider == "openai":
            return self._openai_generate(prompt, **gen_kwargs)
        elif self.provider == "hf":
            return self._hf_generate(prompt, **gen_kwargs)
        elif self.provider == "local":
            if self.local_fn is None:
                raise ValueError("local_fn must be provided for local generation")
            answer = self.local_fn(prompt, **gen_kwargs)
            return {"answer": answer}
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _openai_generate(self, prompt: str, **gen_kwargs) -> Dict[str, Any]:
        """
        Generate with OpenAI API (requires openai package and API key).
        """
        try:
            import openai
        except ImportError:
            raise ImportError("openai package required for OpenAI generation")
        if self.api_key:
            openai.api_key = self.api_key
        model = self.model or "gpt-3.5-turbo"
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **gen_kwargs
        )
        answer = response['choices'][0]['message']['content']
        return {
            "answer": answer,
            "usage": response.get('usage', {}),
            "provider": "openai",
            "model": model
        }

    def _hf_generate(self, prompt: str, **gen_kwargs) -> Dict[str, Any]:
        """
        Generate with HuggingFace Transformers (requires transformers package).
        """
        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError("transformers package required for HuggingFace generation")
        model = self.model or "gpt2"
        gen_pipe = pipeline("text-generation", model=model)
        outputs = gen_pipe(prompt, **gen_kwargs)
        answer = outputs[0]['generated_text'] if outputs else ''
        return {
            "answer": answer,
            "provider": "hf",
            "model": model
        }
