from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from chunkers import Chunk
from data_loader import QAPair
from vector_store import VectorStore


@dataclass
class ChunkStats:
    count: int
    avg_size: float
    median_size: float
    min_size: int
    max_size: int
    std_size: float
    chunking_time: float


@dataclass
class RetrievalResult:
    question: str
    answer: str
    top_chunks: list[str]
    distances: list[float]
    hit: bool


@dataclass
class StrategyReport:
    strategy: str
    chunk_stats: ChunkStats
    mean_relevance: float
    hit_rate: float
    retrieval_results: list[RetrievalResult]


def compute_chunk_stats(chunks: list[Chunk], elapsed: float) -> ChunkStats:
    sizes = [len(c.text) for c in chunks]
    arr = np.array(sizes)
    return ChunkStats(
        count=len(chunks),
        avg_size=float(arr.mean()),
        median_size=float(np.median(arr)),
        min_size=int(arr.min()),
        max_size=int(arr.max()),
        std_size=float(arr.std()),
        chunking_time=elapsed,
    )


def _check_extractive_hit(answer: str, docs: list[str]) -> bool:
    return any(answer.lower() in doc.lower() for doc in docs)


def _check_mc_hit(
    qa: QAPair,
    docs: list[str],
    model: SentenceTransformer,
) -> bool:
    context = " ".join(docs)
    texts = [f"{qa.question} {opt}" for opt in qa.options]
    texts.append(f"{qa.question} {context}")

    embeddings = model.encode(texts)
    option_embs = embeddings[:-1]
    context_emb = embeddings[-1]

    sims = np.dot(option_embs, context_emb) / (
        np.linalg.norm(option_embs, axis=1) * np.linalg.norm(context_emb) + 1e-10
    )
    return int(np.argmax(sims)) == qa.gold_label


def evaluate_retrieval(
    store: VectorStore,
    collection,
    qa_pairs: list[QAPair],
    top_k: int,
    mc_model: SentenceTransformer | None = None,
) -> list[RetrievalResult]:
    results = []
    for qa in qa_pairs:
        res = store.query(collection, qa.question, top_k)
        docs = res["documents"][0]
        dists = res["distances"][0]

        if qa.is_multiple_choice:
            hit = _check_mc_hit(qa, docs, mc_model)
        else:
            hit = _check_extractive_hit(qa.answer, docs)

        results.append(
            RetrievalResult(
                question=qa.question,
                answer=qa.answer,
                top_chunks=docs,
                distances=dists,
                hit=hit,
            )
        )
    return results


def full_report(
    strategy_name: str,
    chunks: list[Chunk],
    elapsed: float,
    store: VectorStore,
    collection,
    qa_pairs: list[QAPair],
    top_k: int,
    mc_model: SentenceTransformer | None = None,
) -> StrategyReport:
    stats = compute_chunk_stats(chunks, elapsed)
    retrieval = evaluate_retrieval(store, collection, qa_pairs, top_k, mc_model)

    hits = sum(1 for r in retrieval if r.hit)
    hit_rate = hits / len(retrieval) * 100 if retrieval else 0.0

    all_dists = [d for r in retrieval for d in r.distances]
    mean_relevance = float(np.mean(all_dists)) if all_dists else 0.0

    return StrategyReport(
        strategy=strategy_name,
        chunk_stats=stats,
        mean_relevance=mean_relevance,
        hit_rate=hit_rate,
        retrieval_results=retrieval,
    )
