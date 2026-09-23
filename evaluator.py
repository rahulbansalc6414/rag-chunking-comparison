from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from openai import OpenAI

from chunkers import Chunk
from config import Config
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
    llm_answer: str = ""
    correct_option: int | None = None
    difficult: bool = False


@dataclass
class StrategyReport:
    strategy: str
    chunk_stats: ChunkStats
    mean_relevance: float
    hit_rate: float
    retrieval_results: list[RetrievalResult]


class EmbeddingHelper:
    def encode(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class LocalEmbeddingHelper(EmbeddingHelper):
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts)


class OpenRouterEmbeddingHelper(EmbeddingHelper):
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        response = self.client.embeddings.create(model=self.model_name, input=texts)
        return np.array([item.embedding for item in response.data])


def get_embedding_helper(config: Config) -> EmbeddingHelper:
    if config.is_api_embedding:
        api_key = config.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY", "")
        return OpenRouterEmbeddingHelper(config.embedding_model_name, api_key)
    return LocalEmbeddingHelper(config.embedding_model_name)


class LLMJudge:
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    def answer_question(self, question: str, context: str, options: list[str] | None = None) -> str:
        if options:
            options_text = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(options))
            prompt = (
                f"Based on the following context, answer the multiple-choice question. "
                f"Reply with ONLY the number of the correct option (1, 2, 3, or 4).\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\n"
                f"Options:\n{options_text}\n\n"
                f"Answer (number only):"
            )
        else:
            prompt = (
                f"Based on the following context, answer the question in as few words as possible. "
                f"Reply with ONLY the answer, nothing else.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\n"
                f"Answer:"
            )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0,
        )
        return response.choices[0].message.content.strip()


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
    helper: EmbeddingHelper,
) -> bool:
    context = " ".join(docs)
    texts = [f"{qa.question} {opt}" for opt in qa.options]
    texts.append(f"{qa.question} {context}")

    embeddings = helper.encode(texts)
    option_embs = embeddings[:-1]
    context_emb = embeddings[-1]

    sims = np.dot(option_embs, context_emb) / (
        np.linalg.norm(option_embs, axis=1) * np.linalg.norm(context_emb) + 1e-10
    )
    return int(np.argmax(sims)) == qa.gold_label


def _check_llm_hit(
    qa: QAPair,
    docs: list[str],
    judge: LLMJudge,
) -> tuple[bool, str]:
    context = "\n\n".join(docs)
    if qa.is_multiple_choice:
        llm_answer = judge.answer_question(qa.question, context, qa.options)
        try:
            chosen = int(llm_answer.strip().rstrip(".")) - 1
            hit = chosen == qa.gold_label
        except ValueError:
            hit = False
        return hit, llm_answer
    else:
        llm_answer = judge.answer_question(qa.question, context)
        hit = qa.answer.lower() in llm_answer.lower() or llm_answer.lower() in qa.answer.lower()
        return hit, llm_answer


def evaluate_retrieval(
    store: VectorStore,
    collection,
    qa_pairs: list[QAPair],
    top_k: int,
    embedding_helper: EmbeddingHelper | None = None,
    llm_judge: LLMJudge | None = None,
) -> list[RetrievalResult]:
    results = []
    for qa in qa_pairs:
        res = store.query(collection, qa.question, top_k)
        docs = res["documents"][0]
        dists = res["distances"][0]

        llm_answer = ""
        if llm_judge:
            hit, llm_answer = _check_llm_hit(qa, docs, llm_judge)
        elif qa.is_multiple_choice:
            hit = _check_mc_hit(qa, docs, embedding_helper)
        else:
            hit = _check_extractive_hit(qa.answer, docs)

        results.append(
            RetrievalResult(
                question=qa.question,
                answer=qa.answer,
                top_chunks=docs,
                distances=dists,
                hit=hit,
                llm_answer=llm_answer,
                correct_option=qa.gold_label + 1 if qa.gold_label is not None else None,
                difficult=qa.difficult,
            )
        )
    return results


def save_detailed_results(
    strategy_name: str,
    results: list[RetrievalResult],
    run_dir: Path,
) -> Path:
    filepath = run_dir / f"{strategy_name.lower()}_detail.csv"
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "question", "expected_answer", "correct_option",
            "llm_answer", "match", "difficulty", "top_chunk_distance",
        ])
        for r in results:
            writer.writerow([
                r.question,
                r.answer,
                r.correct_option or "",
                r.llm_answer,
                "YES" if r.hit else "NO",
                "hard" if r.difficult else "easy",
                f"{r.distances[0]:.4f}" if r.distances else "",
            ])
    return filepath


def full_report(
    strategy_name: str,
    chunks: list[Chunk],
    elapsed: float,
    store: VectorStore,
    collection,
    qa_pairs: list[QAPair],
    top_k: int,
    embedding_helper: EmbeddingHelper | None = None,
    llm_judge: LLMJudge | None = None,
) -> StrategyReport:
    stats = compute_chunk_stats(chunks, elapsed)
    retrieval = evaluate_retrieval(store, collection, qa_pairs, top_k, embedding_helper, llm_judge)

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
