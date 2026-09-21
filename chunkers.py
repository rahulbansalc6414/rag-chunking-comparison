from __future__ import annotations

import time
from dataclasses import dataclass

from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

from config import Config


@dataclass
class Chunk:
    text: str
    index: int
    strategy: str
    source_id: str


class ChunkerBase:
    name: str

    def chunk(self, text: str, source_id: str) -> list[Chunk]:
        raise NotImplementedError


class FixedChunker(ChunkerBase):
    name = "Fixed"

    def __init__(self, config: Config):
        self.splitter = CharacterTextSplitter(
            separator="",
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

    def chunk(self, text: str, source_id: str) -> list[Chunk]:
        texts = self.splitter.split_text(text)
        return [Chunk(text=t, index=i, strategy=self.name, source_id=source_id) for i, t in enumerate(texts)]


class RecursiveChunker(ChunkerBase):
    name = "Recursive"

    def __init__(self, config: Config):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk(self, text: str, source_id: str) -> list[Chunk]:
        texts = self.splitter.split_text(text)
        return [Chunk(text=t, index=i, strategy=self.name, source_id=source_id) for i, t in enumerate(texts)]


class SemanticChunkerWrapper(ChunkerBase):
    name = "Semantic"

    def __init__(self, config: Config):
        embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model_name)
        self.splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=config.semantic_breakpoint_type,
        )

    def chunk(self, text: str, source_id: str) -> list[Chunk]:
        texts = self.splitter.split_text(text)
        return [Chunk(text=t, index=i, strategy=self.name, source_id=source_id) for i, t in enumerate(texts)]


def get_all_strategies(config: Config) -> dict[str, ChunkerBase]:
    return {
        "Fixed": FixedChunker(config),
        "Recursive": RecursiveChunker(config),
        "Semantic": SemanticChunkerWrapper(config),
    }


def chunk_documents(chunker: ChunkerBase, documents: list[dict]) -> tuple[list[Chunk], float]:
    start = time.perf_counter()
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunker.chunk(doc["text"], doc["id"]))
    elapsed = time.perf_counter() - start
    return all_chunks, elapsed
