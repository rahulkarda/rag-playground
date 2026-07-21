# Roadmap

Working through these in roughly this order. Items get crossed off as they land.

## Phase 1: chunking
- [x] Fixed-size chunker with configurable overlap
- [x] Recursive chunker that respects markdown/code boundaries
- [x] Semantic chunker using embedding similarity
- [x] Chunker comparison harness on a small corpus

## Phase 2: embeddings + storage
- [x] Wrapper for sentence-transformers models
- [x] Local FAISS index
- [x] Persistence to disk
- [x] Batch embed with progress

## Phase 3: retrievers
- [x] Dense retriever (FAISS)
- [x] BM25 sparse retriever
- [x] Hybrid (RRF fusion)
- [x] Reranker stage

## Phase 4: generation
- [x] Prompt assembly with citations
- [x] Answer generator wrapper (provider-agnostic)
- [x] Streaming output

## Phase 5: evaluation
- [x] Answer-relevance scorer
- [x] Faithfulness scorer
- [x] Eval CLI that takes a question file + ground-truth contexts
- [x] Notebook with results

## Phase 6: polish
- [x] CLI entrypoint
- [x] Config via YAML
- [x] Tests
- [x] Docs
