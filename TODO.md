# TODO

## High Priority

- [ ] **Full QuALITY test set run** — Current 88% is on 50 questions from 20 documents. Run on the complete test set to validate the result and enable a proper leaderboard submission.
- [ ] **Finish cost tracking in `contextual.py`** — Pricing dict was added but the actual per-run cost calculation and printing was never completed (interrupted mid-run on 2026-09-26).

## Medium Priority

- [ ] **CTX + Semantic experiment** — Does contextual enrichment + semantic chunking stack or overlap with each other? If they overlap, it confirms Fixed is the right base strategy.
- [ ] **CTX + Semantic + CoT + Hybrid + Rerank** — The everything-on ceiling test.
- [ ] **Upgrade re-ranker to bge-reranker-v2-m3** — Currently using ms-marco-MiniLM (~80MB). BGE v2 (~2.3GB) is more accurate and handles 8K context. Test whether the accuracy gain justifies the size.

## Low Priority / Skip

- [ ] **Agentic retrieval** — Query decomposition, iterative search for the 5 always-failing questions. Better to test on Notion data where results matter for the product.
- [ ] **Embedding model comparison** — Test state-of-the-art embedding models. Lower ROI until base pipeline is finalized.
- [ ] **Leaderboard submission** — Formal submission to the QuALITY leaderboard at nyu-mll.github.io/quality/ (requires full test set run first).
