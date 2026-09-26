# Questions & Answers

## How to get the Anthropic API key?

Go to [console.anthropic.com](https://console.anthropic.com) → sign up or log in → Settings → API Keys → Create Key. You'll need to add credits (minimum $5). The contextual enrichment for the current dataset should cost well under $0.10 with Haiku + prompt caching.

## What is the difference between parent-child chunking and contextual retrieval?

They solve the same problem (chunks losing context) from opposite directions:

- **Contextual retrieval**: Enrich the chunk *at index time* with a text prefix that adds context. The chunk itself gets bigger and more searchable. Retrieval and LLM both see the enriched chunk.

- **Parent-child chunking**: Keep two levels — small chunks for retrieval precision, but when a small chunk matches, expand to the *parent* (full section or page) before sending to the LLM. The index stays lean, but the LLM gets richer context.

They're complementary. You could do both — contextual-enriched small chunks for better retrieval, then expand to the parent for the LLM answer. But contextual retrieval is the more impactful one to test first since it improves retrieval itself, which is the main bottleneck.

## Why didn't we use OpenRouter for contextual retrieval?

OpenRouter supports prompt caching, but through the OpenAI SDK it's awkward — you'd need `extra_body` hacks for `cache_control`, and verifying cache hits (checking `cache_read_input_tokens` vs `cache_creation_input_tokens`) requires parsing raw response fields that the OpenAI SDK doesn't expose cleanly.

The Anthropic SDK has native `cache_control` support and returns cache usage directly on `response.usage`. Since prompt caching is the main cost-saving mechanism here (90% savings), clean verification matters — you want to confirm it's actually working, not just hope.

The rest of the pipeline (evaluation, embeddings) stays on OpenRouter. Only the contextual enrichment step uses the Anthropic SDK directly.

## What is `reconstruct_chunks()` — we already have the collection, right?

The Chroma collection stores chunk *texts* and *metadata*, but the rest of the pipeline works with `Chunk` objects (a Python dataclass with `text`, `index`, `strategy`, `source_id`). When we reuse a collection (skip re-chunking), we need to build those `Chunk` objects from what's in Chroma so the hybrid retriever can build its BM25 index — BM25 operates on the raw chunk list, not the Chroma collection.

Without `reconstruct_chunks()`, reusing a collection would only work for dense retrieval, not hybrid.

## Does embedding model quality make a difference in result quality?

Yes, significantly. The project currently uses `all-mpnet-base-v2` — a solid general-purpose model, but not state-of-the-art. Top embedding models right now (Voyage 3 Large, Gemini text-embedding, Cohere embed-v4) typically improve retrieval recall by 10-20% over mpnet on benchmarks like MTEB.

However — there's a tradeoff: API-based embedding models add cost per query *and* per indexing run, and latency for every retrieval. Local mpnet is free and fast. Switching to an API model means every experiment costs more.

Recommendation: test contextual retrieval with the current embedding model first. If the results are strong, that's a win without changing the embedding. If retrieval is still the bottleneck after contextual enrichment, *then* upgrade the embedding model — it'll be a clean comparison to isolate the effect.
