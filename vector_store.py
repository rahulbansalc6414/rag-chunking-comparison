from __future__ import annotations

import re
import shutil

import chromadb

from chunkers import Chunk
from config import Config
from embeddings import get_chroma_embedding_function


_STRATEGY_ABBREV = {"fixed": "fix", "recursive": "rec", "semantic": "sem"}


def _sanitize_name(s: str, max_len: int = 15) -> str:
    return re.sub(r"[^a-z0-9_]", "_", s.lower())[:max_len].strip("_")


class VectorStore:
    def __init__(self, config: Config):
        self.config = config
        self.client = chromadb.PersistentClient(path=str(config.chroma_persist_dir))
        self.ef = get_chroma_embedding_function(config)

    def _collection_name(self, strategy_name: str, contextual: bool) -> str:
        strat = _STRATEGY_ABBREV.get(strategy_name.lower(), strategy_name.lower()[:3])
        embed = _sanitize_name(self.config.embedding_model_name)
        ctx = "ctx" if contextual else "noctx"
        name = f"{strat}_c{self.config.chunk_size}_o{self.config.chunk_overlap}_n{self.config.num_contexts}_{embed}_{ctx}"
        return name[:63]

    def get_or_create_collection(
        self, strategy_name: str, contextual: bool, expected_count: int | None = None,
    ) -> tuple[chromadb.Collection, bool]:
        name = self._collection_name(strategy_name, contextual)
        try:
            collection = self.client.get_collection(name=name, embedding_function=self.ef)
            count = collection.count()
            if count > 0 and (expected_count is None or count == expected_count):
                return collection, True
            self.client.delete_collection(name)
        except Exception:
            pass
        collection = self.client.create_collection(name=name, embedding_function=self.ef)
        return collection, False

    def create_collection(self, strategy_name: str) -> chromadb.Collection:
        name = f"{strategy_name.lower()}_chunks"
        try:
            self.client.delete_collection(name)
        except Exception:
            pass
        return self.client.create_collection(name=name, embedding_function=self.ef)

    def add_chunks(self, collection: chromadb.Collection, chunks: list[Chunk], batch_size: int = 500) -> None:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            collection.add(
                ids=[f"{c.source_id}_{c.index}" for c in batch],
                documents=[c.text for c in batch],
                metadatas=[
                    {"strategy": c.strategy, "source_id": c.source_id, "chunk_index": c.index}
                    for c in batch
                ],
            )

    def reconstruct_chunks(self, collection: chromadb.Collection, strategy_name: str) -> list[Chunk]:
        result = collection.get(include=["documents", "metadatas"])
        chunks = []
        for doc, meta in zip(result["documents"], result["metadatas"]):
            chunks.append(Chunk(
                text=doc,
                index=meta["chunk_index"],
                strategy=strategy_name,
                source_id=meta["source_id"],
            ))
        chunks.sort(key=lambda c: (c.source_id, c.index))
        return chunks

    def query(self, collection: chromadb.Collection, query_text: str, top_k: int) -> dict:
        return collection.query(query_texts=[query_text], n_results=top_k)

    def reset(self) -> None:
        if self.config.chroma_persist_dir.exists():
            shutil.rmtree(self.config.chroma_persist_dir)
        self.client = chromadb.PersistentClient(path=str(self.config.chroma_persist_dir))
