from __future__ import annotations

import shutil

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from chunkers import Chunk
from config import Config


class VectorStore:
    def __init__(self, config: Config):
        self.config = config
        self.client = chromadb.PersistentClient(path=str(config.chroma_persist_dir))
        self.ef = SentenceTransformerEmbeddingFunction(model_name=config.embedding_model_name)

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

    def query(self, collection: chromadb.Collection, query_text: str, top_k: int) -> dict:
        return collection.query(query_texts=[query_text], n_results=top_k)

    def reset(self) -> None:
        if self.config.chroma_persist_dir.exists():
            shutil.rmtree(self.config.chroma_persist_dir)
        self.client = chromadb.PersistentClient(path=str(self.config.chroma_persist_dir))
