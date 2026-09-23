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

### Basic runs

```bash
# SQuAD with default settings
uv run python main.py

# QuALITY with a better embedding model
uv run python main.py --dataset quality --embedding-model all-mpnet-base-v2

# QuALITY with fewer documents for a quick test
uv run python main.py --dataset quality --embedding-model all-mpnet-base-v2 --num-contexts 20
```

### LLM-as-judge (full RAG pipeline)

Uses an LLM via OpenRouter to answer questions using the retrieved chunks, then compares to ground truth. Produces a detailed CSV per strategy with question, expected answer, LLM answer, and match status.

```bash
export OPENROUTER_API_KEY="sk-or-..."

# QuALITY with LLM judge
uv run python main.py --dataset quality --embedding-model all-mpnet-base-v2 \
  --num-contexts 20 --llm-judge anthropic/claude-sonnet-4

# SQuAD with LLM judge
uv run python main.py --dataset squad --llm-judge anthropic/claude-sonnet-4

# Use OpenRouter embeddings instead of local
uv run python main.py --dataset quality --embedding-provider openrouter \
  --embedding-model openai/text-embedding-3-small --llm-judge anthropic/claude-sonnet-4
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `squad` | Dataset to use (`squad`, `quality`) |
| `--chunk-size` | `500` | Chunk size in characters (Fixed and Recursive) |
| `--chunk-overlap` | `50` | Overlap between chunks in characters |
| `--top-k` | `3` | Number of chunks to retrieve per query |
| `--num-contexts` | `100` | Number of documents to sample from the dataset |
| `--max-queries` | `50` | Max queries to evaluate per strategy (0 = all) |
| `--embedding-model` | `all-MiniLM-L6-v2` | Embedding model name |
| `--embedding-provider` | `local` | `local` (Sentence Transformers) or `openrouter` |
| `--openrouter-api-key` | | OpenRouter API key (or set `OPENROUTER_API_KEY` env var) |
| `--llm-judge` | | OpenRouter model for LLM-as-judge (e.g. `anthropic/claude-sonnet-4`) |
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
├── config.py          # Tunable parameters
├── data_loader.py     # Dataset loaders (SQuAD, QuALITY)
├── chunkers.py        # LangChain splitter wrappers
├── embeddings.py      # Embedding function factory (local / OpenRouter)
├── vector_store.py    # ChromaDB wrapper
├── evaluator.py       # Metrics, LLM judge, detailed CSV export
├── main.py            # CLI entry point
└── results/           # Timestamped output from each run
```

## License

MIT
