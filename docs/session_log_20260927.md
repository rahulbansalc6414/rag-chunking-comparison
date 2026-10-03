# Session Log — 2026-09-27

## Starting Point

- CTX + Fixed + Direct + Hybrid was our best result at **84% accuracy** (42/50)
- Re-ranker was identified as the next architectural improvement
- Tradeoff doc for re-ranker already existed but lacked model landscape data

## Activities

### 1. Re-ranker Model Research

- Searched current MTEB reranking leaderboard and published benchmarks
- Identified top models: Querit-Reranker-4B (#1 MTEB), BAAI/bge-reranker-v2-m3 (best open-source), Cohere Rerank 4 (managed API), cross-encoder/ms-marco-MiniLM-L-6-v2 (fastest local)
- Updated `tradeoffs/reranker.md` with model landscape table and sources
- Decision: start with ms-marco-MiniLM (~80MB, local, zero cost, no API key needed)

### 2. Re-ranker Implementation

- Created `reranker.py` — `Reranker` class using `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Updated `config.py` — added `rerank: bool` and `rerank_model: str` fields
- Updated `main.py` — added `--rerank` and `--rerank-model` CLI args, loads reranker once before strategy loop
- Updated `evaluator.py` — `evaluate_retrieval()` and `full_report()` accept optional reranker; when present, over-retrieves top-50 candidates and reranks to top-k

### 3. Latency Tracking Implementation

- Added `retrieval_latency_s` and `rerank_latency_s` fields to `RetrievalResult` dataclass
- Added per-query timing around retrieval and reranking in `evaluate_retrieval()`
- Created `print_latency_table()` in `main.py` — shows retrieval, rerank, LLM judge, and total per-query averages in ms
- Added latency data to `summary.json` output
- Added 4 latency columns to detail CSV: `retrieval_ms`, `rerank_ms`, `llm_ms`, `total_ms`

### 4. Re-ranker Experiment Results

User ran: `caffeinate uv run python main.py --strategy fixed --contextual --retrieval-mode hybrid --rerank`

**Result: 88% accuracy (44/50)** — up from 84% without re-ranker (+4pp)

- Re-ranker rescued 3 questions (vocabulary, thematic synthesis, negation)
- Re-ranker broke 1 question ("Which is probably the author's favorite movie?")
- Net: +3 -1 = +2 questions
- Latency: retrieval 59ms + rerank 243ms + LLM 1,505ms = 1,813ms total per query
- Eval cost unchanged at $0.36 (reranker is free, local model)

### 5. Updated Master Files

- Merged rerank results into `master_results.csv` (added 8 new columns)
- Updated `learnings/master_learnings.md`:
  - Results table: CTX+Fixed+Rerank now #1 at 88%
  - Added finding #2 about re-ranker impact
  - Renumbered remaining findings (3-6)
  - Added latency summary section
  - Updated cost summary with rerank info

### 6. Failure Analysis

- 5 questions fail across ALL configurations (truly unsolvable with current retrieval):
  1. "Which best describes the relationship between the protagonists?" — narrative synthesis
  2. "As the story reaches its climax, the antagonist is" — narrative synthesis
  3. "The characters experience many emotions..." — narrative synthesis
  4. "Who is the best actor mentioned, according to the author?" — subjective judgment
  5. "Once William received the money from Partridge, what didn't he decide to do?" — negation
- 1 question regressed with reranker ("author's favorite movie")

### 7. QuALITY Leaderboard Comparison

- Fetched the official QuALITY leaderboard from nyu-mll.github.io/quality/
- Our 88% on hard questions vs leaderboard #1 at 81.9% (Clustering + Qwen2.5 + DeepSeek)
- Human annotators: 89.1%
- Caveat: our result is on 50 sampled questions, not the full test set
- Added leaderboard comparison section to `learnings/master_learnings.md`

### 8. Resume Review

- Read user's resume PDF from Downloads
- Confirmed the RAG project fits well — reinforces existing AI narrative (AML vector search, AI copilot, GPU embeddings)
- Suggested bullet point and placement under current role or side projects

### 9. New Repo Setup (Enterprise Knowledge System)

- Discussed repo strategy: keep benchmarking repo public (portfolio), new repo for Notion product
- Cloned `rag-chunking-comparison` → `enterprise-knowledge-system`
- Removed git history (`rm -rf .git && git init`)
- Cleaned up benchmarking-specific files (learnings, tradeoffs, CSVs, QA.md)
- Kept all pipeline code (chunkers, contextual, reranker, retriever, evaluator, etc.)
- Created `ROADMAP.md` — 5 phases: Notion ingestion → adapt pipeline → evaluation → benchmark vs Notion AI → productize
- Created new `README.md` — product-focused, links back to benchmarking repo
- Updated `.gitignore` with `notion_data/`

### 10. TODO for This Repo

Created `TODO.md` with prioritized remaining work:
- High: full QuALITY test set run, finish cost tracking in contextual.py
- Medium: CTX+Semantic experiment, upgrade to bge-reranker-v2-m3
- Low: agentic retrieval, embedding model comparison, leaderboard submission

## Files Created

- `reranker.py` — cross-encoder re-ranker module
- `TODO.md` — remaining work for this repo
- `session_log_20260927.md` — this file

## Files Modified

- `config.py` — added rerank fields
- `evaluator.py` — latency tracking, reranker support, CSV latency columns
- `main.py` — rerank CLI args, latency table, reranker wiring, latency in summary.json
- `master_results.csv` — merged rerank run (8 new columns)
- `learnings/master_learnings.md` — updated results, new findings, leaderboard comparison, latency summary
- `tradeoffs/reranker.md` — added model landscape table with sources

## New Repo Created

- `/Users/rahulbansal/projects/enterprise-knowledge-system/`
- Files: ROADMAP.md, README.md, updated .gitignore, plus all pipeline code from this repo
- Not yet pushed to GitHub

## Key Numbers

| Metric | Value |
|--------|-------|
| Best accuracy | 88% (44/50) — CTX + Fixed + Direct + Hybrid + Rerank |
| Previous best | 84% (42/50) — CTX + Fixed + Direct + Hybrid |
| Improvement | +4pp from re-ranker |
| QuALITY leaderboard #1 | 81.9% (full test set) |
| Human performance | 89.1% |
| Avg query latency | 1,813ms (retrieval 59ms + rerank 243ms + LLM 1,505ms) |
| Re-ranker cost | $0 (local model, ~80MB) |
| Eval cost per run | $0.36 |
