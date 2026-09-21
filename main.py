from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from sentence_transformers import SentenceTransformer

from config import Config
from data_loader import load_dataset_sample, DATASET_REGISTRY
from chunkers import get_all_strategies, chunk_documents
from vector_store import VectorStore
from evaluator import full_report, StrategyReport


console = Console()


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="RAG Chunking Strategy Comparison")
    parser.add_argument(
        "--dataset", type=str, default="squad",
        help=f"Dataset to use ({', '.join(sorted(DATASET_REGISTRY.keys()))})",
    )
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--num-contexts", type=int, default=100)
    parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    return Config(
        dataset=args.dataset,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        num_contexts=args.num_contexts,
        embedding_model_name=args.embedding_model,
        random_seed=args.seed,
    )


def print_stats_table(reports: list[StrategyReport]) -> None:
    table = Table(title="Chunk Statistics", show_lines=True)
    table.add_column("Metric", style="bold")
    for r in reports:
        table.add_column(r.strategy, justify="right")

    rows = [
        ("Count", lambda r: str(r.chunk_stats.count)),
        ("Avg Size (chars)", lambda r: f"{r.chunk_stats.avg_size:.1f}"),
        ("Median Size", lambda r: f"{r.chunk_stats.median_size:.1f}"),
        ("Min Size", lambda r: str(r.chunk_stats.min_size)),
        ("Max Size", lambda r: str(r.chunk_stats.max_size)),
        ("Std Dev", lambda r: f"{r.chunk_stats.std_size:.1f}"),
        ("Chunking Time (s)", lambda r: f"{r.chunk_stats.chunking_time:.2f}"),
    ]
    for label, fn in rows:
        table.add_row(label, *[fn(r) for r in reports])
    console.print(table)


def print_hit_rate_table(reports: list[StrategyReport], dataset: str) -> None:
    is_mc = dataset == "quality"
    if is_mc:
        title = "Answer Accuracy (retrieved chunks help pick the correct option)"
    else:
        title = "Answer Hit Rate (top-k retrieval contains the answer)"

    table = Table(title=title, show_lines=True)
    table.add_column("Strategy", style="bold")
    table.add_column("Hit Rate (%)", justify="right")
    table.add_column("Mean Distance", justify="right")
    table.add_column("Total Queries", justify="right")

    for r in reports:
        table.add_row(
            r.strategy,
            f"{r.hit_rate:.1f}%",
            f"{r.mean_relevance:.4f}",
            str(len(r.retrieval_results)),
        )
    console.print(table)


def print_sample_chunks(chunks_by_strategy: dict[str, list]) -> None:
    first_source = None
    for chunks in chunks_by_strategy.values():
        if chunks:
            first_source = chunks[0].source_id
            break
    if not first_source:
        return

    console.print(Panel("[bold]Sample Chunks (first source document)[/bold]"))
    for strategy_name, chunks in chunks_by_strategy.items():
        source_chunks = [c for c in chunks if c.source_id == first_source][:3]
        console.print(f"\n[bold cyan]{strategy_name}[/bold cyan] — {len(source_chunks)} sample chunks:")
        for c in source_chunks:
            snippet = c.text[:200] + ("..." if len(c.text) > 200 else "")
            console.print(f"  [dim]Chunk {c.index}[/dim] ({len(c.text)} chars): {snippet}")


def save_results(config: Config, reports: list[StrategyReport]) -> Path:
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{config.dataset}_{timestamp}.json"
    filepath = results_dir / filename

    result = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "dataset": config.dataset,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "top_k": config.top_k,
            "num_contexts": config.num_contexts,
            "embedding_model": config.embedding_model_name,
            "semantic_breakpoint_type": config.semantic_breakpoint_type,
            "random_seed": config.random_seed,
        },
        "strategies": {},
    }

    for r in reports:
        result["strategies"][r.strategy] = {
            "chunk_stats": {
                "count": r.chunk_stats.count,
                "avg_size": round(r.chunk_stats.avg_size, 1),
                "median_size": round(r.chunk_stats.median_size, 1),
                "min_size": r.chunk_stats.min_size,
                "max_size": r.chunk_stats.max_size,
                "std_size": round(r.chunk_stats.std_size, 1),
                "chunking_time_s": round(r.chunk_stats.chunking_time, 3),
            },
            "hit_rate_pct": round(r.hit_rate, 2),
            "mean_distance": round(r.mean_relevance, 4),
            "total_queries": len(r.retrieval_results),
            "hits": sum(1 for rr in r.retrieval_results if rr.hit),
        }

    filepath.write_text(json.dumps(result, indent=2))
    return filepath


def save_chart(reports: list[StrategyReport], dataset: str, path: str = "comparison_chart.png") -> None:
    strategies = [r.strategy for r in reports]
    counts = [r.chunk_stats.count for r in reports]
    avg_sizes = [r.chunk_stats.avg_size for r in reports]
    hit_rates = [r.hit_rate for r in reports]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    axes[0].bar(strategies, counts, color=colors)
    axes[0].set_title("Chunk Count")
    axes[0].set_ylabel("Number of Chunks")

    axes[1].bar(strategies, avg_sizes, color=colors)
    axes[1].set_title("Avg Chunk Size (chars)")
    axes[1].set_ylabel("Characters")

    axes[2].bar(strategies, hit_rates, color=colors)
    axes[2].set_title("Answer Hit Rate")
    axes[2].set_ylabel("Hit Rate (%)")
    axes[2].set_ylim(0, 100)

    plt.suptitle(f"RAG Chunking Strategy Comparison ({dataset})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    console.print(f"\nChart saved to [bold]{path}[/bold]")


def main():
    config = parse_args()
    console.print(Panel(f"[bold]RAG Chunking Strategy Comparison — {config.dataset}[/bold]", style="blue"))

    console.print(f"\n[bold]Loading {config.dataset} dataset...[/bold]")
    documents, qa_pairs = load_dataset_sample(config)
    console.print(f"Loaded {len(documents)} documents, {len(qa_pairs)} QA pairs")

    is_mc = any(qa.is_multiple_choice for qa in qa_pairs)
    mc_model = None
    if is_mc:
        console.print("[bold]Multiple-choice dataset detected — loading evaluation model...[/bold]")
        mc_model = SentenceTransformer(config.embedding_model_name)

    store = VectorStore(config)
    strategies = get_all_strategies(config)

    reports: list[StrategyReport] = []
    chunks_by_strategy: dict[str, list] = {}

    for name, chunker in strategies.items():
        console.print(f"\n[bold yellow]Processing: {name}[/bold yellow]")

        console.print("  Chunking documents...")
        chunks, elapsed = chunk_documents(chunker, documents)
        chunks_by_strategy[name] = chunks
        console.print(f"  {len(chunks)} chunks in {elapsed:.2f}s")

        console.print("  Creating collection and inserting...")
        collection = store.create_collection(name)
        store.add_chunks(collection, chunks)

        console.print("  Evaluating retrieval...")
        report = full_report(
            name, chunks, elapsed, store, collection, qa_pairs, config.top_k, mc_model
        )
        reports.append(report)
        console.print(f"  Hit rate: {report.hit_rate:.1f}%")

    console.print("\n")
    print_stats_table(reports)
    console.print()
    print_hit_rate_table(reports, config.dataset)
    console.print()
    print_sample_chunks(chunks_by_strategy)
    save_chart(reports, config.dataset)

    results_path = save_results(config, reports)
    console.print(f"Results saved to [bold]{results_path}[/bold]")


if __name__ == "__main__":
    main()
