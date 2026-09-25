from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    dataset: str = "quality"
    chunk_size: int = 500
    chunk_overlap: int = 50
    semantic_breakpoint_type: str = "percentile"
    embedding_model_name: str = "all-mpnet-base-v2"
    embedding_provider: str = "local"
    openrouter_api_key: str = ""
    top_k: int = 10
    num_contexts: int = 20
    max_queries: int = 50
    difficult_only: bool = True
    retrieval_mode: str = "dense"
    llm_judge: str = "anthropic/claude-sonnet-4"
    chroma_persist_dir: Path = field(default_factory=lambda: Path("./chroma_data"))
    random_seed: int = 42

    @property
    def is_api_embedding(self) -> bool:
        return self.embedding_provider == "openrouter"
