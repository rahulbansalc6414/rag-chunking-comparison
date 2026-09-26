# Master Learnings

## Best Results So Far (as of 2026-09-26)

| Rank | Configuration                    | Accuracy        | Eval Cost | Notes                                             |
|------|----------------------------------|-----------------|-----------|---------------------------------------------------|
| 1    | Semantic + CoT + Hybrid          | **88%** (44/50) | ~$0.46*   | *Projected — CoT only ran on 18 failing questions  |
| 2    | CTX + Fixed + Direct + Hybrid    | **84%** (42/50) | $0.36     | Contextual enrichment cost: $1.95 (one-time)       |
| 3    | CTX + Fixed + CoT + Hybrid       | 80% (40/50)     | $0.59     | CoT hurt — dropped from 84% to 80%                |
| 3    | Semantic + Direct + Hybrid       | 80% (40/50)     | $1.01     | Best non-contextual, non-CoT result                |
| 5    | Fixed + Direct + Hybrid          | 76% (38/50)     | $0.23     | Baseline                                           |
| 6    | Recursive + Direct + Hybrid      | 68% (34/50)     | $0.19     | Consistently worst — loses cross-sentence context  |

## Data Sources

| Configuration                    | Source File                                    | Questions |
|----------------------------------|------------------------------------------------|-----------|
| Fixed + Direct + Hybrid          | `results/quality_20260925_150148/detail.csv`   | 50        |
| Recursive + Direct + Hybrid      | `results/quality_20260925_150148/detail.csv`   | 50        |
| Semantic + Direct + Hybrid       | `results/quality_20260925_150148/detail.csv`   | 50        |
| Semantic + CoT + Hybrid          | `results/quality_20260925_182545/detail.csv`   | 18 (failing questions only) |
| CTX + Fixed + Direct + Hybrid    | `results/quality_20260926_194337/detail.csv`   | 50        |
| CTX + Fixed + CoT + Hybrid       | `results/quality_20260926_200919/detail.csv`   | 50        |

Combined in `master_results.csv` (excludes the earlier 18-question CoT run — not comparable).

## Key Findings

### 1. Contextual retrieval is the biggest single improvement

Adding contextual enrichment to Fixed chunking jumped accuracy from 76% → 84% (+8pp). This is the largest gain from any single change we've tested. It also beats plain Semantic (80%) while costing 1/3 as much in evaluation ($0.36 vs $1.01), because Fixed chunks are smaller than Semantic chunks.

### 2. Contextual Fixed beats plain Semantic

| Metric          | Fixed (baseline) | Semantic (baseline) | CTX + Fixed        |
|-----------------|------------------|---------------------|---------------------|
| Accuracy        | 76%              | 80%                 | **84%**             |
| Eval cost       | $0.23            | $1.01               | $0.36               |
| Chunk count     | 1,095            | 326                 | 1,095               |
| Median chunk    | 500 chars        | 919 chars           | 880 chars (with prefix) |
| Chunking time   | 0.17s            | 44.6s               | 0.17s + enrichment  |

Contextual retrieval solves the "orphaned chunk" problem that Semantic chunking was designed to address, but without the cost and complexity of embedding-based boundary detection.

### 3. Chain-of-thought helps Semantic but hurts Fixed — even with contextual enrichment

CoT on the 18 hardest questions (without contextual):

| Strategy   | Direct | CoT  | Net change |
|------------|--------|------|------------|
| Fixed      | 6/18   | 5/18 | -1         |
| Recursive  | 2/18   | 2/18 | 0          |
| Semantic   | 8/18   | 12/18| **+4**     |

CoT on all 50 questions with contextual Fixed:

| Prompt Mode | Accuracy | Eval Cost |
|-------------|----------|-----------|
| Direct      | 84% (42/50) | $0.36  |
| CoT         | 80% (40/50) | $0.59  |

CoT dropped CTX+Fixed by 4 points while costing 64% more. The pattern is consistent: CoT reasoning requires large chunks (Semantic's median 919 chars) to be effective. Even contextually enriched Fixed chunks (median 880 chars) aren't enough — the context prefix adds retrievability but not the narrative continuity that CoT reasoning needs.

### 4. Hybrid retrieval consistently outperforms dense-only

Comparing across earlier runs, hybrid retrieval (BM25 + dense with RRF) improved Fixed from 72% → 76% and Recursive from 66% → 68% over dense-only.

### 5. Recursive chunking is not competitive

Recursive consistently ranks last. Its aggressive splitting at structural boundaries (sentences, paragraphs) produces many small chunks (median 391 chars) that lose the cross-sentence context needed for reasoning-heavy questions.

## Remaining Failures

5 questions fail across all tested configurations:

| Question                                                              | Failure Type           |
|-----------------------------------------------------------------------|------------------------|
| As the story reaches its climax, the antagonist is                    | Narrative synthesis    |
| The characters experience many emotions for the first time during the | Narrative synthesis    |
| Who is the best actor mentioned, according to the author?             | Subjective judgment    |
| What didn't William get accused of as a young boy?                    | Negation question      |
| Once William received the money from Partridge, what didn't he decide | Negation question      |

These are likely retrieval-bound — the right passages never make it into the LLM context. Agentic retrieval (query decomposition, iterative retrieval) is the next lever to pull.

## Untested Combinations

| Configuration                    | Why it's worth testing                                  |
|----------------------------------|---------------------------------------------------------|
| CTX + Semantic + Direct + Hybrid | Do contextual + semantic stack, or overlap?              |
| CTX + Semantic + CoT + Hybrid    | The everything-on combination — ceiling test             |

## Cost Summary

| Cost Type                      | Amount   | When incurred          |
|--------------------------------|----------|------------------------|
| Contextual enrichment (Haiku)  | $1.95    | One-time at ingestion  |
| Evaluation: Fixed (per run)    | $0.23    | Each evaluation run    |
| Evaluation: Semantic (per run) | $1.01    | Each evaluation run    |
| Evaluation: CTX+Fixed direct   | $0.36    | Each evaluation run    |
| Evaluation: CTX+Fixed CoT      | $0.59    | Each evaluation run    |

Contextual enrichment prefixes are cached to disk (`contextual_cache/`). Chroma collections are reused across runs with the same config. Only evaluation costs recur.
