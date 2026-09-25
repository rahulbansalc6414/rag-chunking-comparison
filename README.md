# RAG Chunking Strategy Comparison

A Python CLI tool that compares three RAG chunking strategies side by side using real QA datasets, ChromaDB, and Sentence Transformers. Optionally uses an LLM-as-judge for full end-to-end RAG evaluation.

## Chunking Strategies

- **Fixed** — Splits every N characters regardless of content. The baseline.
- **Recursive** — Tries paragraph breaks first, then sentences, then words. Keeps semantic units together when possible.
- **Semantic** — Embeds every sentence and places breakpoints where topic shifts. The smartest strategy but slowest.

## Datasets

- **SQuAD** — Short Wikipedia paragraphs with extractive QA pairs. Evaluation checks if the answer string appears in retrieved chunks.
- **QuALITY** — Long-form articles (5,000–28,000 chars) with multiple-choice questions requiring reasoning across the full document.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/YOUR_USERNAME/rag-chunking-comparison.git
cd rag-chunking-comparison
uv sync
```

## Usage

All defaults are set in `config.py`. You only need CLI flags to override them.

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

### Common commands

```bash
# Run with all defaults (QuALITY, hard questions, dense retrieval, LLM judge)
uv run python main.py

# Hybrid retrieval (BM25 + dense embeddings with Reciprocal Rank Fusion)
uv run python main.py --retrieval-mode hybrid

# Compare dense vs hybrid side by side (run both, compare results/)
uv run python main.py --retrieval-mode dense
uv run python main.py --retrieval-mode hybrid

# SQuAD dataset instead of QuALITY
uv run python main.py --dataset squad

# Include easy questions too
uv run python main.py --no-difficult-only

# Quick test with fewer documents
uv run python main.py --num-contexts 5 --max-queries 10

# Use a different LLM judge
uv run python main.py --llm-judge google/gemini-2.5-flash

# Disable LLM judge (embedding-only evaluation)
uv run python main.py --llm-judge ""

# OpenRouter embeddings instead of local
uv run python main.py --embedding-provider openrouter \
  --embedding-model openai/text-embedding-3-small
```

### Options

All defaults come from `config.py` — edit that file to change your baseline.

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `quality` | Dataset to use (`squad`, `quality`) |
| `--chunk-size` | `500` | Chunk size in characters (Fixed and Recursive) |
| `--chunk-overlap` | `50` | Overlap between chunks in characters |
| `--top-k` | `10` | Number of chunks to retrieve per query |
| `--num-contexts` | `20` | Number of documents to sample from the dataset |
| `--max-queries` | `50` | Max queries to evaluate per strategy (0 = all) |
| `--difficult-only` | `true` | Only use hard questions (QuALITY only) |
| `--no-difficult-only` | | Use all questions, not just hard ones |
| `--embedding-model` | `all-mpnet-base-v2` | Embedding model name |
| `--embedding-provider` | `local` | `local` (Sentence Transformers) or `openrouter` |
| `--retrieval-mode` | `dense` | `dense` (embedding-only) or `hybrid` (BM25 + dense with RRF) |
| `--openrouter-api-key` | | OpenRouter API key (or set `OPENROUTER_API_KEY` env var) |
| `--llm-judge` | `anthropic/claude-sonnet-4` | OpenRouter model for LLM-as-judge (empty string to disable) |
| `--seed` | `42` | Random seed for reproducible sampling |

## Output

Each run creates its own folder under `results/`, named `<dataset>_<timestamp>/`:

```
results/
├── quality_20260922_165442/
│   ├── summary.json            # Config + metrics for all strategies
│   ├── fixed_detail.csv        # Per-query results (when --llm-judge is set)
│   ├── recursive_detail.csv
│   └── semantic_detail.csv
├── squad_20260923_091500/
│   └── summary.json
└── ...
```

- **summary.json** — full config and metrics for all strategies
- **Detail CSVs** (when `--llm-judge` is set) — question, expected answer, LLM answer, and match column per strategy

Terminal output includes:
- Chunk statistics table (count, avg size, std dev, timing)
- Hit rate table (the headline metric)
- Sample chunks from each strategy for visual comparison

## Project Structure

```
├── config.py          # Tunable parameters (single source of truth for defaults)
├── data_loader.py     # Dataset loaders (SQuAD, QuALITY)
├── chunkers.py        # LangChain splitter wrappers
├── embeddings.py      # Embedding function factory (local / OpenRouter)
├── vector_store.py    # ChromaDB wrapper
├── retriever.py       # Retrieval abstraction (Dense, Hybrid with BM25 + RRF)
├── evaluator.py       # Metrics, LLM judge, detailed CSV export
├── main.py            # CLI entry point
└── results/           # Timestamped output from each run
```

## License

MIT
