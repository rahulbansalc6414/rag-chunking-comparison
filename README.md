# RAG Chunking Strategy Comparison

A Python CLI tool that compares three RAG chunking strategies side by side using real QA datasets, ChromaDB, and Sentence Transformers.

## Chunking Strategies

- **Fixed** — Splits every N characters regardless of content. The baseline.
- **Recursive** — Tries paragraph breaks first, then sentences, then words. Keeps semantic units together when possible.
- **Semantic** — Embeds every sentence and places breakpoints where topic shifts. The smartest strategy but slowest.

## Datasets

- **SQuAD** — Short Wikipedia paragraphs with extractive QA pairs. Evaluation checks if the answer string appears in retrieved chunks.
- **QuALITY** — Long-form articles (5,000–28,000 chars) with multiple-choice questions. Evaluation checks if retrieved chunks are semantically closest to the correct option.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/YOUR_USERNAME/rag-chunking-comparison.git
cd rag-chunking-comparison
uv sync
```

## Usage

```bash
# Run with SQuAD (default)
uv run python main.py

# Run with QuALITY (long documents)
uv run python main.py --dataset quality

# Customize parameters
uv run python main.py --dataset quality --num-contexts 50 --chunk-size 1000 --top-k 5
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `squad` | Dataset to use (`squad`, `quality`) |
| `--chunk-size` | `500` | Chunk size in characters (Fixed and Recursive) |
| `--chunk-overlap` | `50` | Overlap between chunks in characters |
| `--top-k` | `3` | Number of chunks to retrieve per query |
| `--num-contexts` | `100` | Number of documents to sample from the dataset |
| `--embedding-model` | `all-MiniLM-L6-v2` | Sentence Transformer model for embeddings |
| `--seed` | `42` | Random seed for reproducible sampling |

## Output

Each run produces:

1. **Stats table** — chunk count, avg size, std dev, timing per strategy
2. **Hit rate table** — the headline metric, percentage per strategy
3. **Sample chunks** — first 2–3 chunks from each strategy for visual comparison
4. **Bar chart** — saved as `comparison_chart.png`
5. **JSON results** — saved under `results/` with full config and metrics

## Project Structure

```
├── config.py          # Tunable parameters
├── data_loader.py     # Dataset loaders (SQuAD, QuALITY)
├── chunkers.py        # LangChain splitter wrappers
├── vector_store.py    # ChromaDB wrapper
├── evaluator.py       # Metrics (chunk stats, retrieval relevance, hit rate)
├── main.py            # CLI entry point
└── results/           # JSON output from each run
```

## License

MIT
