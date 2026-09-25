from __future__ import annotations

import re
from dataclasses import dataclass

import chromadb
from rank_bm25 import BM25Okapi

from chunkers import Chunk
from vector_store import VectorStore


@dataclass
class RetrievalOutput:
    documents: list[str]
    distances: list[float]


class Retriever:
    def build_index(self, store: VectorStore, collection: chromadb.Collection, chunks: list[Chunk]) -> None:
        raise NotImplementedError

    def retrieve(self, query: str, top_k: int) -> RetrievalOutput:
        raise NotImplementedError


class DenseRetriever(Retriever):
    def build_index(self, store: VectorStore, collection: chromadb.Collection, chunks: list[Chunk]) -> None:
        self.store = store
        self.collection = collection

    def retrieve(self, query: str, top_k: int) -> RetrievalOutput:
        res = self.store.query(self.collection, query, top_k)
        return RetrievalOutput(
            documents=res["documents"][0],
            distances=res["distances"][0],
        )


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class HybridRetriever(Retriever):
    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k

    def build_index(self, store: VectorStore, collection: chromadb.Collection, chunks: list[Chunk]) -> None:
        self.store = store
        self.collection = collection
        self.chunk_texts = [c.text for c in chunks]
        self.chunk_ids = [f"{c.source_id}_{c.index}" for c in chunks]
        tokenized = [_tokenize(t) for t in self.chunk_texts]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_k: int) -> RetrievalOutput:
        fetch_k = min(top_k * 3, len(self.chunk_texts))

        dense_res = self.store.query(self.collection, query, fetch_k)
        dense_docs = dense_res["documents"][0]
        dense_dists = dense_res["distances"][0]

        query_tokens = _tokenize(query)
        bm25_scores = self.bm25.get_scores(query_tokens)
        bm25_top_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:fetch_k]

        dense_rank: dict[str, tuple[int, float]] = {}
        for rank, (doc, dist) in enumerate(zip(dense_docs, dense_dists)):
            dense_rank[doc] = (rank, dist)

        bm25_rank: dict[str, tuple[int, float]] = {}
        for rank, idx in enumerate(bm25_top_idx):
            bm25_rank[self.chunk_texts[idx]] = (rank, bm25_scores[idx])

        all_docs = set(dense_rank.keys()) | set(bm25_rank.keys())

        scored: list[tuple[str, float, float]] = []
        for doc in all_docs:
            rrf_score = 0.0
            dist = 1.0
            if doc in dense_rank:
                r, d = dense_rank[doc]
                rrf_score += 1.0 / (self.rrf_k + r + 1)
                dist = d
            if doc in bm25_rank:
                r, _ = bm25_rank[doc]
                rrf_score += 1.0 / (self.rrf_k + r + 1)
                if doc not in dense_rank:
                    dist = 1.0
            scored.append((doc, rrf_score, dist))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]

        return RetrievalOutput(
            documents=[d for d, _, _ in top],
            distances=[dist for _, _, dist in top],
        )
