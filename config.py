from dataclasses import dataclass, field
from pathlib import Path


PROMPT_TEMPLATES = {
    "direct": {
        "mc": (
            "Based on the following context, answer the multiple-choice question. "
            "Reply with ONLY the number of the correct option (1, 2, 3, or 4).\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Options:\n{options}\n\n"
            "Answer (number only):"
        ),
        "open": (
            "Based on the following context, answer the question in as few words as possible. "
            "Reply with ONLY the answer, nothing else.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        ),
        "max_tokens": 100,
    },
    "cot": {
        "mc": (
            "Based on the following context, answer the multiple-choice question. "
            "Think through the evidence step by step.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Options:\n{options}\n\n"
            'Respond with JSON: {{"reasoning": "your step-by-step reasoning", "answer": <option number>}}'
        ),
        "open": (
            "Based on the following context, answer the question. "
            "Think through the evidence step by step.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            'Respond with JSON: {{"reasoning": "your step-by-step reasoning", "answer": "your answer in as few words as possible"}}'
        ),
        "max_tokens": 500,
        "json_mode": True,
    },
}


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
    prompt_mode: str = "direct"
    questions_file: str = ""
    strategy: str = "all"
    chunk_only: bool = False
    contextual: bool = False
    contextual_model: str = "claude-haiku-4-5-20251001"
    llm_judge: str = "anthropic/claude-sonnet-4"
    chroma_persist_dir: Path = field(default_factory=lambda: Path("./chroma_data"))
    random_seed: int = 42

    @property
    def is_api_embedding(self) -> bool:
        return self.embedding_provider == "openrouter"
