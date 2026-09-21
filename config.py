from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    dataset: str = "squad"
    chunk_size: int = 500
    chunk_overlap: int = 50
    semantic_breakpoint_type: str = "percentile"
    embedding_model_name: str = "all-MiniLM-L6-v2"
    top_k: int = 3
    num_contexts: int = 100
    chroma_persist_dir: Path = field(default_factory=lambda: Path("./chroma_data"))
    random_seed: int = 42
