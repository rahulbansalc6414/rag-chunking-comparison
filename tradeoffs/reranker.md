# Tradeoff Analysis: Introducing a Re-ranker

## What It Does

A re-ranker sits between retrieval and the LLM. Instead of passing the top-K results from hybrid search directly to the LLM, you over-retrieve (top-50) and then use a cross-encoder model to re-score each candidate against the original query. The cross-encoder sees the full (query, chunk) pair together, giving it much deeper understanding of relevance than embedding similarity alone.

## Accuracy

**Positive**:
- Cross-encoders are significantly more accurate than bi-encoder similarity because they attend to the query and document jointly, not independently.
- Anthropic's contextual retrieval research showed re-ranking contributed to a 67% total reduction in retrieval failures (from 5.7% to 1.9%) when combined with contextual embeddings + BM25.
- Particularly helps with queries where the right chunk uses different vocabulary than the question (semantic gap).
- Filters out false positives — chunks that are topically similar but don't actually answer the question.

**Negative**:
- A re-ranker can only re-order what was retrieved. If the correct chunk isn't in the top-50 candidates, re-ranking can't help. It improves precision, not recall.
- Re-ranker quality varies. A weak re-ranker can demote relevant results.

## Latency

**Positive**:
- Re-ranking 50 chunks is fast — typically 50-200ms for a cross-encoder, depending on chunk size.
- Net latency may actually decrease if you feed fewer, better chunks to the LLM (shorter context = faster LLM response).

**Negative**:
- Adds a synchronous step in the query-time critical path. Every query pays this cost.
- If using an API-based re-ranker (Cohere, Jina), network round-trip is added.
- With very large chunks (Semantic chunking, median 919 chars), scoring 50 candidates takes longer than with small chunks.

**Estimate**: +100-300ms per query for an API-based re-ranker, +20-50ms for a local model.

## Cost

**Positive**:
- Can reduce LLM cost by sending fewer, more relevant chunks (top-5 instead of top-10), shrinking the context window.
- Local cross-encoder models (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) are free to run.

**Negative**:
- API-based re-rankers charge per query. Cohere Rerank: ~$2/1000 queries. At scale (100K queries/month), this is $200/month.
- Self-hosted cross-encoders need GPU for acceptable latency. CPU inference is ~5-10x slower.
- Over-retrieval (top-50 instead of top-10) increases vector DB query cost slightly.

**Estimate**: $0.001-0.002 per query (API) or near-zero (self-hosted with GPU).

## Complexity

**Positive**:
- Simple to integrate — it's a filter step between retrieval and LLM, no changes to indexing or chunking.
- Can be added/removed without re-indexing.
- Well-understood technique with mature libraries and APIs.

**Negative**:
- New dependency in the query path. If the re-ranker service is down, you need a fallback (skip re-ranking and use raw retrieval scores).
- Adds a tuning parameter: how many candidates to over-retrieve (top-N before re-ranking). Too few = no benefit. Too many = slow.
- Need to evaluate whether to re-rank against the original query or a reformulated version.

## Security and Privacy

**Positive**:
- Self-hosted cross-encoders keep all data local. No external API calls needed.

**Negative**:
- API-based re-rankers (Cohere, Jina) receive your chunks and queries. For enterprise customers with sensitive data, this is a data residency and privacy concern.
- Terms of service vary — some providers may use data for model training unless opted out.
- Adds another vendor to your security review and SOC 2 scope.

**Mitigation**: Use a self-hosted cross-encoder for enterprise deployments. Reserve API re-rankers for non-sensitive or internal use.

## Scalability

**Positive**:
- Re-ranker load scales linearly with query volume, not document volume. Adding more documents to the index doesn't affect re-ranking latency.
- Cross-encoders are small models — easy to scale horizontally.

**Negative**:
- At very high QPS (1000+), a single re-ranker instance becomes a bottleneck. Need load balancing.
- GPU memory limits how many concurrent re-ranking requests you can handle.

## Vendor Lock-in

- **Low risk if self-hosted**: Cross-encoder models are open-source (HuggingFace). Switching models is a config change.
- **Medium risk if API-based**: Cohere and Jina have different APIs. Switching requires code changes, but the interface is simple (list of texts in, scores out).

## Recommendation

Add a re-ranker as the **lowest-risk, highest-ROI improvement** to the current pipeline:
- Start with Cohere Rerank API for fast validation.
- Move to a self-hosted cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2` or similar) before enterprise deployment.
- Over-retrieve top-50 with hybrid search, re-rank to top-5 for the LLM.
- Add a bypass fallback: if re-ranker is unavailable, fall back to raw hybrid scores.
