from __future__ import annotations

from sentence_transformers import CrossEncoder

from retriever import RetrievalOutput


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, retrieval: RetrievalOutput, top_k: int) -> RetrievalOutput:
        if not retrieval.documents:
            return retrieval

        pairs = [(query, doc) for doc in retrieval.documents]
        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(retrieval.documents, retrieval.distances, scores),
            key=lambda x: x[2],
            reverse=True,
        )[:top_k]

        return RetrievalOutput(
            documents=[doc for doc, _, _ in ranked],
            distances=[dist for _, dist, _ in ranked],
        )
