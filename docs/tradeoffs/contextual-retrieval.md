# Tradeoff Analysis: Introducing Contextual Retrieval

## What It Does

At ingestion time, each chunk is sent to an LLM along with the full source document. The LLM generates a short context prefix (50-100 tokens) that situates the chunk within the broader document — adding entity names, section titles, temporal references, and other disambiguating information. This prefix is prepended to the chunk before embedding and BM25 indexing.

Example: A chunk containing *"Revenue grew by 3% over the previous quarter"* gets prepended with *"This is from Acme Corp's Q3 2026 earnings report, discussing financial performance following Q2's $1.2B revenue."*

## Accuracy

**Positive**:
- Anthropic's research showed 35% fewer retrieval failures with contextual embeddings alone, 49% with contextual BM25 added.
- Directly fixes the "orphaned chunk" problem — chunks that are meaningful in context but ambiguous in isolation.
- BM25 benefits enormously: the prefix injects entity names and keywords that wouldn't otherwise appear in the chunk, making lexical matching far more effective.
- Particularly valuable for enterprise documents where the same terms appear across many pages (e.g., "the project" could be any project — the prefix disambiguates).

**Negative**:
- Context quality depends on the LLM. A poor or hallucinated prefix can actively mislead retrieval — pushing the chunk toward wrong queries.
- The LLM sees only the document, not the full corpus. It can't know what disambiguation the retrieval system will need across all possible queries.
- Uniform prefix length (50-100 tokens) may under-serve chunks that need more context or waste tokens on self-explanatory chunks.

## Latency

**Positive**:
- Zero impact on query-time latency. All enrichment happens at ingestion, not at query time.
- Better retrieval quality means fewer irrelevant chunks in the context, which can speed up LLM response time.

**Negative**:
- Ingestion time increases substantially. Every chunk requires an LLM call. For 1,000 chunks, that's 1,000 API calls (though batching and prompt caching reduce wall-clock time).
- Re-indexing on document updates is slower — when a page changes, all its chunks need fresh context prefixes.
- Initial index build for a large Notion workspace (10K+ pages) could take hours.

**Estimate**: With prompt caching, ~2-5 seconds per document (not per chunk, since the full document context is cached across its chunks). A 10K-page workspace: ~6-14 hours for initial ingestion.

## Cost

**Positive**:
- Prompt caching makes this dramatically cheaper than naive per-chunk LLM calls. Anthropic reports ~$1.02 per million document tokens.
- One-time cost at ingestion. Amortized over all queries, it's negligible.
- Can reduce downstream LLM costs by improving retrieval precision (fewer tokens in the answer context).

**Negative**:
- Without prompt caching, cost explodes. A 500-token chunk from a 5,000-token document means sending 5,500 tokens per chunk. For a document with 10 chunks, that's 55,000 input tokens just for context generation — vs 5,000 with caching.
- Re-indexing costs recur. If documents change frequently (e.g., a living wiki), you're paying for context regeneration on every update.
- Model choice matters: using Claude Sonnet vs Haiku for prefix generation can be a 5-10x cost difference.

**Estimate**:
- With prompt caching: ~$1/million document tokens (initial), recurring cost proportional to churn rate.
- Without prompt caching: ~$10-15/million document tokens — use prompt caching.
- For a typical Notion workspace (5M tokens): ~$5 initial ingestion cost with caching.

## Complexity

**Positive**:
- Conceptually simple — it's a preprocessing step, not a change to the retrieval algorithm.
- The enriched chunks are just longer strings. No changes needed to the vector store, BM25 index, or query pipeline.
- Easy to A/B test: index the same documents with and without contextual prefixes and compare retrieval quality.

**Negative**:
- Adds an LLM dependency to the ingestion pipeline. Ingestion now requires API access, rate limit handling, retry logic, and error handling for the context generation step.
- Prompt engineering for the context generator matters. A bad prompt produces useless prefixes. Needs iteration and evaluation.
- Debugging retrieval issues becomes harder: is the problem the chunk content, the prefix, the embedding, or the query? One more layer to inspect.
- Need to store both the original chunk and the prefix separately — the prefix shouldn't be shown to users in the final answer, only used for retrieval.
- Document-level context window limits: very long documents may need to be summarized before being sent as context for each chunk.

## Security and Privacy

**Positive**:
- Context generation can use the same LLM provider you already use for answering, so no new vendor is introduced.
- Prefixes are derived from the document itself — no external data is mixed in.

**Negative**:
- **Full documents are sent to the LLM at ingestion time.** This is the biggest security concern. Every document in the knowledge base passes through an external API. For enterprises with sensitive data (financial, legal, HR, medical), this may violate data handling policies.
- With prompt caching, the full document persists in the provider's cache for minutes. The caching TTL and data retention policies matter.
- A compromised or hallucinated prefix could leak information from one part of a document into a chunk from another part — creating unexpected cross-references in retrieval results.
- The prefix becomes part of the indexed data. If the LLM hallucinates a wrong entity name into a prefix, that chunk may surface for queries about that entity — a subtle data integrity issue.

**Mitigation**:
- Use a self-hosted LLM for context generation in high-security deployments (e.g., local Llama/Mistral).
- Validate prefixes: reject or flag any prefix that introduces entity names not present in the source document.
- Implement prefix auditing: store and review generated prefixes, especially for sensitive document categories.
- Ensure the data processing agreement with your LLM provider covers ingestion-time data, not just query-time data.

## Storage

**Positive**:
- Prefix overhead is small: 50-100 tokens (~200-400 chars) per chunk. For 10,000 chunks, that's ~2-4MB of additional text.

**Negative**:
- Embeddings of longer chunks (original + prefix) are the same dimensionality, so vector storage doesn't increase.
- BM25 index grows slightly due to the additional prefix tokens.
- Need to store the prefix separately from the original chunk if you don't want to show it to users.

**Estimate**: <5% increase in total storage. Negligible.

## Freshness and Maintenance

**Positive**:
- Prefix generation is deterministic for a given document state. Same document produces the same prefixes (with temperature=0).

**Negative**:
- **When a document changes, all its chunks need re-contextualization**, not just the changed chunks. The prefix for chunk 5 might reference content in paragraph 2 — if paragraph 2 changes, chunk 5's prefix is stale.
- For rapidly changing documents (daily-edited wiki pages), this creates a re-indexing burden.
- Stale prefixes are subtly harmful: they don't cause errors, they just quietly degrade retrieval quality. Hard to detect without ongoing evaluation.

**Mitigation**:
- Track document-level edit timestamps. Re-contextualize all chunks when any part of the document changes.
- For high-churn documents, consider lighter-weight prefixes (page title + section header only, no LLM call) that can be regenerated instantly.
- Set up periodic retrieval quality checks on a sample of queries to detect prefix staleness.

## Vendor Lock-in

- **Low**: The prefix is just text prepended to a chunk. Switching LLM providers for prefix generation requires only changing the API call. The indexed data works with any embedding model and any vector store.
- **Caveat**: If you tune the prefix prompt heavily for one model's style, switching models may produce different-quality prefixes. Re-indexing is needed.

## Model Choice for Prefix Generation

- **Haiku/small models**: Cheaper, faster, sufficient for most factual prefix generation. Recommended for production.
- **Sonnet/large models**: Better at nuanced context (e.g., understanding narrative structure in literary texts). Overkill for most enterprise documents.
- **Self-hosted (Llama, Mistral)**: Best for security-sensitive deployments. Quality is lower but may be acceptable for straightforward enterprise docs.

## Recommendation

Contextual retrieval is a **high-impact, moderate-complexity improvement** that attacks the root cause of most retrieval failures:

- Implement with prompt caching from day one — the cost difference is 10x.
- Use Haiku-class models for prefix generation in production (cheaper, fast enough).
- Store prefixes separately from original chunks — use for retrieval, strip before showing to users.
- Plan for re-indexing: build the pipeline to handle document updates, not just initial ingestion.
- For enterprise customers with strict data policies, offer a self-hosted prefix generation option.
- Validate prefixes against source documents to catch hallucinated entity names.
