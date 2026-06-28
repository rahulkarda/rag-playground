from typing import List, Dict, Any

def assemble_prompt(query: str, contexts: List[Dict[str, Any]], citation_prefix: str = "[" , citation_suffix: str = "]") -> str:
    """
    Assemble a prompt for generation using the query and retrieved contexts.
    Each context is assigned a citation (e.g. [1], [2]), and referenced inline after its chunk.

    Args:
        query (str): user question
        contexts (List[Dict]): list of dicts with 'text' (chunk), optional 'metadata'
        citation_prefix (str): prefix for citations (default: "[")
        citation_suffix (str): suffix for citations (default: "]")
    Returns:
        str: prompt string ready for LLM generation
    Example:
        contexts = [
            {"text": "RAG is a retrieval-augmented generation framework."},
            {"text": "Chunking improves retrieval granularity."}
        ]
        query = "What is RAG?"
        prompt = assemble_prompt(query, contexts)
    """
    assembled = ""
    for i, ctx in enumerate(contexts, start=1):
        citation = f"{citation_prefix}{i}{citation_suffix}"
        # Optionally, include metadata (source/title)
        meta = ctx.get("metadata", {})
        source_str = ""
        if meta:
            if "title" in meta:
                source_str = f"\nSource: {meta['title']}"
            elif "source" in meta:
                source_str = f"\nSource: {meta['source']}"
        assembled += f"Chunk {i}: {ctx['text']} {citation}{source_str}\n\n"
    prompt = f"Answer the following question using the provided chunks. Cite sources by number.\n\nQuestion: {query}\n\n{assembled}"
    return prompt
