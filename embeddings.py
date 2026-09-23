from __future__ import annotations

import os

from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from openai import OpenAI

from config import Config


class OpenRouterEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    def __call__(self, input: Documents) -> Embeddings:
        batch_size = 100
        all_embeddings = []
        for i in range(0, len(input), batch_size):
            batch = input[i : i + batch_size]
            response = self.client.embeddings.create(model=self.model_name, input=batch)
            all_embeddings.extend([item.embedding for item in response.data])
        return all_embeddings


def get_chroma_embedding_function(config: Config) -> EmbeddingFunction:
    if config.is_api_embedding:
        api_key = config.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise ValueError("OpenRouter API key required. Pass --openrouter-api-key or set OPENROUTER_API_KEY env var.")
        return OpenRouterEmbeddingFunction(model_name=config.embedding_model_name, api_key=api_key)

    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    return SentenceTransformerEmbeddingFunction(model_name=config.embedding_model_name)
