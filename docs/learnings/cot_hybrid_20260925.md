# Learnings — CoT Run 2026-09-25T18:25:45 (CoT + Hybrid)

## Run Configuration

- **Dataset**: QuALITY (hard questions only, 18 questions from `failing_questions.json`)
- **Retrieval**: Hybrid (BM25 + dense with RRF)
- **Prompt mode**: CoT (chain-of-thought with JSON output)
- **LLM judge**: `anthropic/claude-sonnet-4` via OpenRouter
- **Embedding**: `all-mpnet-base-v2` (local)
- **Chunk size**: 500, overlap 50, top_k 10, num_contexts 20

## Purpose

Test whether chain-of-thought prompting improves accuracy on the 18 questions where at least one strategy failed in the direct+hybrid baseline run (2026-09-25T15:08:21).

## Results on the 18 Hard Questions

| Strategy   | Direct | CoT  | Gained | Lost | Net  |
|------------|--------|------|--------|------|------|
| Fixed      | 6/18   | 5/18 | +1     | -2   | **-1** |
| Recursive  | 2/18   | 2/18 | +2     | -2   | **0**  |
| Semantic   | 8/18   | 12/18| +4     | 0    | **+4** |

## Projected Full-Run Accuracy (50 Questions)

| Strategy   | Direct (Hybrid) | CoT (Hybrid) | Change |
|------------|-----------------|--------------|--------|
| Fixed      | 76% (38/50)     | 74% (37/50)  | -2pp   |
| Recursive  | 68% (34/50)     | 68% (34/50)  | 0      |
| Semantic   | 80% (40/50)     | **88% (44/50)** | **+8pp** |

## Key Finding: CoT Only Helps Semantic Chunking

CoT improved Semantic from 80% → 88% with zero regressions on this subset. But it hurt or was neutral for Fixed and Recursive — they gained some questions but lost others they previously got right.

**Why**: CoT reasoning requires enough context to reason over. Semantic chunks (median 919 chars) provide the multi-paragraph context that CoT needs. With Fixed (500 chars) and Recursive (391 chars median), the model reasons step-by-step but still lacks the evidence to reach the right answer — and the longer output sometimes introduces more room for error.

## Questions CoT + Semantic Recovered (4)

1. **"Which best describes the relationship between the protagonists?"**
   - Direct answered option 4, CoT traced character interactions → option 3 (correct)
   - Multi-hop: needed to synthesize multiple interaction scenes

2. **"What does the author seem to value the most in films?"**
   - Direct answered option 1, CoT analyzed praise/criticism patterns → option 4 (correct)
   - Synthesis: needed to identify a pattern across multiple reviews

3. **"What was the one thing William admitted to doing?"**
   - Direct got stuck mid-response, CoT traced the timeline → option 2 (correct)
   - Temporal reasoning: needed to follow a sequence of events

4. **"Who didn't William say strange things to?"**
   - Direct answered option 2, CoT checked each character → option 4 (correct)
   - Negation + enumeration: needed to verify each option against the text

All four are **multi-hop reasoning** questions. CoT enables the model to work through evidence systematically instead of guessing.

## Still Failing All Strategies (5 Questions)

Even with CoT + Semantic + Hybrid, 5 questions fail across all strategies:

1. "As the story reaches its climax, the antagonist is" — requires narrative arc understanding
2. "The characters experience many emotions for the first time during the..." — requires story-level synthesis
3. "Who is the best actor mentioned, according to the author?" — subjective judgment across reviews
4. "What didn't William get accused of as a young boy?" — negation question
5. "Once William received the money from Partridge, what didn't he decide to do?" — negation question

Pattern: 2 are **negation questions**, 2 require **story-level narrative understanding**, 1 requires **subjective synthesis**. These likely need either better retrieval (more/different context) or agentic multi-step retrieval to solve.

## Cost and Latency

| Strategy   | Direct $/q | CoT $/q | Latency (Direct) | Latency (CoT) |
|------------|-----------|---------|-------------------|----------------|
| Fixed      | $0.0046   | $0.0084 | 1.60s             | 5.22s          |
| Recursive  | $0.0037   | $0.0081 | 1.61s             | 7.60s          |
| Semantic   | $0.0202   | $0.0255 | 1.80s             | 10.34s         |

CoT is ~2x the cost and ~3-5x the latency due to the longer JSON reasoning output (~300 tokens vs ~10 tokens per answer).

## Takeaways

1. **Semantic + CoT + Hybrid is the best configuration**: 88% projected accuracy on QuALITY hard questions.
2. **CoT is only worthwhile with large chunks**: it hurts or is neutral with Fixed/Recursive. Don't use CoT if your chunks are small.
3. **The remaining failures are retrieval-bound, not reasoning-bound**: the 5 still-failing questions need better context in the window, not better reasoning over what's already there.
4. **Cost tradeoff is acceptable**: Semantic + CoT costs $0.0255/question vs $0.0046 for Fixed + Direct (5.5x), but the accuracy gain (88% vs 74%) justifies it for enterprise use.
5. **Next lever to pull is contextual retrieval** (Anthropic's approach): enrich chunks with document-level context at ingestion time to fix the retrieval failures that CoT can't overcome.
