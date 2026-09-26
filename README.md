# RAG Chunking Strategy Comparison

A Python CLI tool that compares RAG chunking strategies using real QA datasets, ChromaDB, and Sentence Transformers. Supports contextual retrieval, chain-of-thought prompting, hybrid retrieval (BM25 + dense), and LLM-as-judge evaluation.

## Chunking Strategies

- **Fixed** — Splits every N characters regardless of content. The baseline.
- **Recursive** — Tries paragraph breaks first, then sentences, then words. Keeps semantic units together when possible.
- **Semantic** — Embeds every sentence and places breakpoints where topic shifts. The smartest strategy but slowest.

## Key Features

- **Contextual retrieval** — Enriches each chunk with document-level context via LLM before embedding (based on [Anthropic's research](https://www.anthropic.com/engineering/contextual-retrieval))
- **Prompt modes** — Direct (answer-only) or chain-of-thought (step-by-step reasoning with JSON output)
- **Hybrid retrieval** — BM25 + dense embeddings with Reciprocal Rank Fusion
- **Collection caching** — Chroma collections encode their config in the name, so re-runs reuse existing embeddings
- **Contextual prefix caching** — Generated prefixes are saved to disk, so re-runs skip LLM calls entirely

## Datasets

- **SQuAD** — Short Wikipedia paragraphs with extractive QA pairs.
- **QuALITY** — Long-form articles (5,000–28,000 chars) with multiple-choice questions requiring reasoning across the full document.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/YOUR_USERNAME/rag-chunking-comparison.git
cd rag-chunking-comparison
uv sync
```

### API Keys

```bash
# Required for LLM judge evaluation
export OPENROUTER_API_KEY="sk-or-..."

# Required only for --contextual (contextual retrieval enrichment)
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Common Commands

### Basic runs

```bash
# Run all strategies with defaults (QuALITY, hard questions, dense retrieval, LLM judge)
uv run python main.py

# Run a single strategy
uv run python main.py --strategy semantic

# Quick test with fewer queries
uv run python main.py --strategy fixed --max-queries 10
```

### Contextual retrieval

```bash
# Step 1: Generate contextual chunks (pay LLM cost once, results cached)
caffeinate uv run python main.py --strategy semantic --contextual --chunk-only

# Step 2: Evaluate (reuses cached collection, no re-chunking or API calls)
caffeinate uv run python main.py --strategy semantic --contextual --prompt-mode cot --retrieval-mode hybrid

# Step 3: Compare with non-contextual baseline
caffeinate uv run python main.py --strategy semantic --prompt-mode cot --retrieval-mode hybrid
```

### Retrieval modes

```bash
# Dense only (embedding similarity)
uv run python main.py --retrieval-mode dense

# Hybrid (BM25 + dense with Reciprocal Rank Fusion)
uv run python main.py --retrieval-mode hybrid
```

### Prompt modes

```bash
# Direct — answer only (faster, cheaper)
uv run python main.py --prompt-mode direct

# Chain-of-thought — step-by-step reasoning in JSON (slower, more accurate for Semantic)
uv run python main.py --prompt-mode cot
```

### Other useful commands

```bash
# Run on specific questions from a file
uv run python main.py --strategy semantic --questions-file failing_questions.json

# SQuAD dataset instead of QuALITY
uv run python main.py --dataset squad

# Include easy questions too
uv run python main.py --no-difficult-only

# Use a different LLM judge
uv run python main.py --llm-judge google/gemini-2.5-flash

# Disable LLM judge (embedding-only evaluation)
uv run python main.py --llm-judge ""

# OpenRouter embeddings instead of local
uv run python main.py --embedding-provider openrouter \
  --embedding-model openai/text-embedding-3-small
```

## Options

All defaults come from `config.py` — edit that file to change your baseline.

| Flag | Default | Description |
|------|---------|-------------|
| `--strategy` | `all` | Strategy to run: `fixed`, `recursive`, `semantic`, or `all` |
| `--chunk-only` | off | Stop after chunking + indexing (no evaluation) |
| `--contextual` | off | Enrich chunks with document-level context via LLM |
| `--contextual-model` | `claude-haiku-4-5-20251001` | Anthropic model for contextual enrichment |
| `--prompt-mode` | `direct` | `direct` (answer-only) or `cot` (chain-of-thought) |
| `--retrieval-mode` | `dense` | `dense` or `hybrid` (BM25 + dense with RRF) |
| `--dataset` | `quality` | Dataset: `squad` or `quality` |
| `--chunk-size` | `500` | Chunk size in characters |
| `--chunk-overlap` | `50` | Overlap between chunks in characters |
| `--top-k` | `10` | Chunks to retrieve per query |
| `--num-contexts` | `20` | Documents to sample from the dataset |
| `--max-queries` | `50` | Max queries to evaluate (0 = all) |
| `--difficult-only` | on | Only hard questions (QuALITY) |
| `--questions-file` | | JSON file with specific questions to evaluate |
| `--llm-judge` | `anthropic/claude-sonnet-4` | OpenRouter model for evaluation |
| `--embedding-model` | `all-mpnet-base-v2` | Embedding model name |
| `--embedding-provider` | `local` | `local` or `openrouter` |
| `--seed` | `42` | Random seed for reproducible sampling |

## Output

Each evaluation run creates a folder under `results/`:

- **summary.json** — Config + metrics for all strategies
- **detail.csv** — Per-question results with each strategy's answer and match status

Terminal output includes chunk statistics, hit rate table, LLM cost breakdown, and sample chunks.

## Results (QuALITY Dataset, Hard Questions, 50 queries)

| Rank | Configuration                 | Accuracy        | Eval Cost |
|------|-------------------------------|-----------------|-----------|
| 1    | CTX + Fixed + Direct + Hybrid | **84%** (42/50) | $0.36     |
| 2    | Semantic + Direct + Hybrid    | 80% (40/50)     | $1.01     |
| 3    | Fixed + Direct + Hybrid       | 76% (38/50)     | $0.23     |
| 4    | Recursive + Direct + Hybrid   | 68% (34/50)     | $0.19     |

Key findings:
- **Contextual retrieval is the biggest single improvement**: Fixed + contextual (84%) beats plain Semantic (80%) at 1/3 the cost
- **Hybrid retrieval outperforms dense-only** across all strategies
- **Chain-of-thought helps large chunks only**: CoT improves Semantic but hurts Fixed, even with contextual enrichment
- **Recursive chunking is not competitive** for reasoning-heavy questions

See `learnings/master_learnings.md` for detailed analysis and `master_results.csv` for per-question results.

## Project Structure

```
├── config.py          # Defaults and prompt templates
├── data_loader.py     # Dataset loaders (SQuAD, QuALITY)
├── chunkers.py        # Chunking strategies (Fixed, Recursive, Semantic)
├── contextual.py      # Contextual retrieval enrichment with prompt caching
├── embeddings.py      # Embedding function factory (local / OpenRouter)
├── vector_store.py    # ChromaDB wrapper with collection caching
├── retriever.py       # Dense and Hybrid (BM25 + RRF) retrievers
├── evaluator.py       # Metrics, LLM judge, CSV export
├── main.py            # CLI entry point
├── contextual_cache/  # Cached contextual prefixes (auto-generated)
├── chroma_data/       # Persisted ChromaDB collections
├── results/           # Timestamped output from each run
├── tradeoffs/         # Tradeoff analysis docs for architectural decisions
├── PLAN.md            # Roadmap for enterprise knowledge system
└── QA.md              # Technical Q&A
```

## License

MIT
