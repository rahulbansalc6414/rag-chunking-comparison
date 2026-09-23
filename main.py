from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from config import Config
from data_loader import load_dataset_sample, DATASET_REGISTRY
from chunkers import get_all_strategies, chunk_documents
from vector_store import VectorStore
from evaluator import (
    full_report, get_embedding_helper, LLMJudge,
    save_detailed_results, StrategyReport,
)


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
    parser.add_argument("--max-queries", type=int, default=50, help="Max queries to run (0 = all)")
    parser.add_argument("--difficult-only", action="store_true", help="Only use hard questions (QuALITY only)")
    parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument(
        "--embedding-provider", type=str, default="local", choices=["local", "openrouter"],
        help="Use 'local' for Sentence Transformers or 'openrouter' for API-based embeddings",
    )
    parser.add_argument("--openrouter-api-key", type=str, default="")
    parser.add_argument(
        "--llm-judge", type=str, default="",
        help="OpenRouter model for LLM-as-judge evaluation (e.g. anthropic/claude-sonnet-4)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    return Config(
        dataset=args.dataset,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        num_contexts=args.num_contexts,
        max_queries=args.max_queries,
        difficult_only=args.difficult_only,
        embedding_model_name=args.embedding_model,
        embedding_provider=args.embedding_provider,
        openrouter_api_key=args.openrouter_api_key,
        llm_judge=args.llm_judge,
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


def print_hit_rate_table(reports: list[StrategyReport], config: Config) -> None:
    if config.llm_judge:
        title = f"LLM Judge Accuracy ({config.llm_judge})"
    elif config.dataset == "quality":
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


def save_results(config: Config, reports: list[StrategyReport], run_dir: Path) -> Path:
    filename = "summary.json"
    filepath = run_dir / filename

    result = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "dataset": config.dataset,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "top_k": config.top_k,
            "num_contexts": config.num_contexts,
            "max_queries": config.max_queries,
            "embedding_model": config.embedding_model_name,
            "embedding_provider": config.embedding_provider,
            "llm_judge": config.llm_judge,
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


def main():
    config = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("results") / f"{config.dataset}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    console.print(Panel(f"[bold]RAG Chunking Strategy Comparison — {config.dataset}[/bold]", style="blue"))

    console.print(f"\n[bold]Loading {config.dataset} dataset...[/bold]")
    documents, qa_pairs = load_dataset_sample(config)
    console.print(f"Loaded {len(documents)} documents, {len(qa_pairs)} QA pairs")

    if config.difficult_only:
        qa_pairs = [qa for qa in qa_pairs if qa.difficult]
        console.print(f"Filtered to {len(qa_pairs)} hard questions (--difficult-only)")

    if config.max_queries > 0 and len(qa_pairs) > config.max_queries:
        qa_pairs = qa_pairs[:config.max_queries]
        console.print(f"Capped to {config.max_queries} queries (--max-queries)")

    llm_judge = None
    if config.llm_judge:
        api_key = config.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            console.print("[bold red]Error: --llm-judge requires OPENROUTER_API_KEY[/bold red]")
            return
        llm_judge = LLMJudge(config.llm_judge, api_key)
        console.print(f"[bold]LLM judge enabled: {config.llm_judge}[/bold]")

    is_mc = any(qa.is_multiple_choice for qa in qa_pairs)
    embedding_helper = None
    if is_mc and not llm_judge:
        console.print("[bold]Multiple-choice dataset detected — loading evaluation model...[/bold]")
        embedding_helper = get_embedding_helper(config)

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
            name, chunks, elapsed, store, collection, qa_pairs, config.top_k,
            embedding_helper, llm_judge,
        )
        reports.append(report)
        console.print(f"  Hit rate: {report.hit_rate:.1f}%")

        if llm_judge:
            detail_path = save_detailed_results(name, report.retrieval_results, run_dir)
            console.print(f"  Detail CSV: [bold]{detail_path}[/bold]")

    console.print("\n")
    print_stats_table(reports)
    console.print()
    print_hit_rate_table(reports, config)
    console.print()
    print_sample_chunks(chunks_by_strategy)

    results_path = save_results(config, reports, run_dir)
    console.print(f"\nResults saved to [bold]{run_dir}/[/bold]")


if __name__ == "__main__":
    main()
