# Roadmap

Working through these in roughly this order. Items get crossed off as they land.

## Phase 1: chunking
- [x] Fixed-size chunker with configurable overlap
- [x] Recursive chunker that respects markdown/code boundaries
- [x] Semantic chunker using embedding similarity
- [x] Chunker comparison harness on a small corpus

## Phase 2: embeddings + storage
- [ ] Wrapper for sentence-transformers models
- [ ] Local FAISS index
- [ ] Persistence to disk
- [ ] Batch embed with progress

## Phase 3: retrievers
- [ ] Dense retriever (FAISS)
- [ ] BM25 sparse retriever
- [ ] Hybrid (RRF fusion)
- [ ] Reranker stage

## Phase 4: generation
- [ ] Prompt assembly with citations
- [ ] Answer generator wrapper (provider-agnostic)
- [ ] Streaming output

## Phase 5: evaluation
- [ ] Answer-relevance scorer
- [ ] Faithfulness scorer
- [ ] Eval CLI that takes a question file + ground-truth contexts
- [ ] Notebook with results

## Phase 6: polish
- [ ] CLI entrypoint
- [ ] Config via YAML
- [ ] Tests
- [ ] Docs
