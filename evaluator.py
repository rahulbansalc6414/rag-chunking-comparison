from __future__ import annotations

import csv
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from openai import OpenAI

from chunkers import Chunk
from config import Config, PROMPT_TEMPLATES
from data_loader import QAPair
from retriever import Retriever
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
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    cost: float = 0.0


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
    llm_usage: LLMUsage | None = None


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


def _fetch_openrouter_pricing(model: str, api_key: str) -> tuple[float, float]:
    """Fetch per-token pricing (input, output) from OpenRouter's models API."""
    import httpx
    resp = httpx.get("https://openrouter.ai/api/v1/models", timeout=15)
    resp.raise_for_status()
    for m in resp.json().get("data", []):
        if m.get("id", "").startswith(model) or model.startswith(m.get("id", "")):
            pricing = m.get("pricing", {})
            return float(pricing.get("prompt", 0)), float(pricing.get("completion", 0))
    return 0.0, 0.0


class LLMJudge:
    def __init__(self, model: str, api_key: str, prompt_mode: str = "direct"):
        self.model = model
        self.prompt_mode = prompt_mode
        self.templates = PROMPT_TEMPLATES[prompt_mode]
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.input_price, self.output_price = _fetch_openrouter_pricing(model, api_key)

    def answer_question(self, question: str, context: str, options: list[str] | None = None) -> tuple[str, LLMUsage]:
        if options:
            options_text = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(options))
            prompt = self.templates["mc"].format(
                context=context, question=question, options=options_text,
            )
        else:
            prompt = self.templates["open"].format(
                context=context, question=question,
            )

        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.templates["max_tokens"],
            "temperature": 0,
        }
        if self.templates.get("json_mode"):
            kwargs["response_format"] = {"type": "json_object"}

        start = time.perf_counter()
        raw = self.client.chat.completions.with_raw_response.create(**kwargs)
        latency = time.perf_counter() - start

        response = raw.parse()
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        cost = input_tokens * self.input_price + output_tokens * self.output_price
        content = response.choices[0].message.content.strip()

        if self.templates.get("json_mode"):
            try:
                parsed = json.loads(content)
                answer = str(parsed.get("answer", ""))
                reasoning = parsed.get("reasoning", "")
                content = f"{reasoning}\n\n{answer}" if reasoning else answer
            except json.JSONDecodeError:
                pass

        return content, LLMUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_s=latency,
            cost=cost,
        )


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
) -> tuple[bool, str, LLMUsage]:
    context = "\n\n".join(docs)
    if qa.is_multiple_choice:
        llm_answer, usage = judge.answer_question(qa.question, context, qa.options)
        numbers = re.findall(r"\b([1-4])\b", llm_answer)
        try:
            chosen = int(numbers[-1]) - 1 if numbers else int(llm_answer.strip().rstrip(".")) - 1
            hit = chosen == qa.gold_label
        except (ValueError, IndexError):
            hit = False
        return hit, llm_answer, usage
    else:
        llm_answer, usage = judge.answer_question(qa.question, context)
        last_line = llm_answer.strip().rsplit("\n", 1)[-1].strip()
        hit = (
            qa.answer.lower() in last_line.lower()
            or last_line.lower() in qa.answer.lower()
            or qa.answer.lower() in llm_answer.lower()
        )
        return hit, llm_answer, usage


def evaluate_retrieval(
    retriever: Retriever,
    qa_pairs: list[QAPair],
    top_k: int,
    embedding_helper: EmbeddingHelper | None = None,
    llm_judge: LLMJudge | None = None,
) -> list[RetrievalResult]:
    results = []
    for qa in qa_pairs:
        res = retriever.retrieve(qa.question, top_k)
        docs = res.documents
        dists = res.distances

        llm_answer = ""
        llm_usage = None
        if llm_judge:
            hit, llm_answer, llm_usage = _check_llm_hit(qa, docs, llm_judge)
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
                llm_usage=llm_usage,
            )
        )
    return results


def save_detailed_results(
    reports: list[StrategyReport],
    run_dir: Path,
) -> Path:
    filepath = run_dir / "detail.csv"
    strategies = [r.strategy for r in reports]
    results_by_strategy = {r.strategy: r.retrieval_results for r in reports}
    num_questions = len(reports[0].retrieval_results)

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["question", "expected_answer", "correct_option", "difficulty"]
        for s in strategies:
            header += [f"{s}_answer", f"{s}_match", f"{s}_dist", f"{s}_cost"]
        writer.writerow(header)

        for i in range(num_questions):
            ref = results_by_strategy[strategies[0]][i]
            row = [
                ref.question,
                ref.answer,
                ref.correct_option or "",
                "hard" if ref.difficult else "easy",
            ]
            for s in strategies:
                r = results_by_strategy[s][i]
                u = r.llm_usage
                row += [
                    r.llm_answer,
                    "YES" if r.hit else "NO",
                    f"{r.distances[0]:.4f}" if r.distances else "",
                    f"{u.cost:.6f}" if u else "",
                ]
            writer.writerow(row)
    return filepath


def full_report(
    strategy_name: str,
    chunks: list[Chunk],
    elapsed: float,
    retriever: Retriever,
    qa_pairs: list[QAPair],
    top_k: int,
    embedding_helper: EmbeddingHelper | None = None,
    llm_judge: LLMJudge | None = None,
) -> StrategyReport:
    stats = compute_chunk_stats(chunks, elapsed)
    retrieval = evaluate_retrieval(retriever, qa_pairs, top_k, embedding_helper, llm_judge)

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
