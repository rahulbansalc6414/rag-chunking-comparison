# Learnings — Run 2026-09-25T15:08:21 (Direct + Hybrid)

## Run Configuration

- **Dataset**: QuALITY (hard questions only, 50 questions)
- **Retrieval**: Hybrid (BM25 + dense with RRF)
- **Prompt mode**: Direct (answer-only, no chain-of-thought)
- **LLM judge**: `anthropic/claude-sonnet-4` via OpenRouter
- **Embedding**: `all-mpnet-base-v2` (local)
- **Chunk size**: 500, overlap 50, top_k 10, num_contexts 20

## Accuracy Results

| Strategy   | Hits | Total | Hit Rate |
|------------|------|-------|----------|
| Fixed      | 38   | 50    | 76%      |
| Recursive  | 34   | 50    | 68%      |
| Semantic   | 40   | 50    | 80%      |

**Semantic chunking is the best strategy**, beating Fixed by 4 points and Recursive by 12 points.

Recursive consistently underperforms — its aggressive splitting at structural boundaries (sentences, paragraphs) produces many small chunks (median 391 chars vs Fixed's 500) that may lose cross-sentence context needed for reasoning-heavy questions.

## Cost and Latency

| Strategy   | Cost (USD) | Avg Latency (s) | Input Tokens |
|------------|------------|------------------|--------------|
| Fixed      | $0.23      | 1.60             | 74,377       |
| Recursive  | $0.19      | 1.61             | 59,333       |
| Semantic   | $1.01      | 1.80             | 333,498      |

**Semantic is 4-5x more expensive** because its chunks are much larger (median 919 chars vs 391-500), so the retrieved context passed to the LLM is substantially bigger. Latency is only slightly higher since the bottleneck is network round-trips, not token processing.

Total run cost: **$1.42** across all three strategies.

## Failure Analysis

18 out of 50 questions had at least one strategy fail. These were saved to `failing_questions.json`.

**Failure patterns:**

| Pattern                        | Count | Questions                                                     |
|--------------------------------|-------|---------------------------------------------------------------|
| All 3 failed                   | 7     | Hardest questions — no strategy could answer them              |
| Only Semantic correct          | 4     | Semantic's larger chunks captured context the others missed    |
| Fixed + Semantic correct       | 2     | Recursive split too aggressively                               |
| Fixed + Recursive correct      | 2     | Semantic retrieved wrong passages despite larger chunks        |
| Only Fixed + Semantic mixed    | 3     | Various single-strategy failures                               |

### All-3-fail questions (7)
These are the hardest cases where retrieval + LLM reasoning both break down:
1. "Which best describes the relationship between the protagonists?"
2. "As the story reaches its climax, the antagonist is"
3. "The characters experience many emotions for the first time during the..."
4. "Who is the best actor mentioned, according to the author?"
5. "What didn't William get accused of as a young boy?"
6. "Once William received the money from Partridge, what didn't he decide..."
7. "Who didn't William say strange things to?"

Notable: 3 of the 7 are **negation questions** ("didn't", "didn't"), which are known to be harder for LLMs. The others require high-level narrative reasoning (relationships, climax, subjective judgment).

### Semantic-only wins (4)
Semantic recovered 4 questions that both Fixed and Recursive missed, all from the same story (Rikud). The larger semantic chunks likely preserved narrative continuity that was lost when the text was split at fixed boundaries.

## Key Takeaways

1. **Semantic chunking is the best overall strategy** for long-form literary comprehension (QuALITY), but at ~5x the cost.
2. **Hybrid retrieval works well** — comparing to earlier dense-only runs (Sep 23), Fixed improved from 72% → 76% and Recursive from 66% → 68%.
3. **Negation questions are a systematic weak spot** across all strategies — 3 of 7 universal failures.
4. **The cost-accuracy tradeoff is steep**: Semantic gains 4 percentage points over Fixed but costs 4.4x more in LLM inference.
5. **Recursive is not competitive** for this task — its smaller chunks lose too much context for questions requiring multi-paragraph reasoning.
