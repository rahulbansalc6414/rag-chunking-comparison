from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from itertools import groupby
from pathlib import Path

import anthropic
from rich.console import Console

from chunkers import Chunk
from config import Config

console = Console()

# Pricing per million tokens (input, output) for Anthropic models
_ANTHROPIC_PRICING: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-sonnet-4-6-20250725": (3.00, 15.00),
    "claude-opus-4-20250515": (15.00, 75.00),
}


def _get_pricing(model: str) -> tuple[float, float]:
    if model in _ANTHROPIC_PRICING:
        return _ANTHROPIC_PRICING[model]
    for key, prices in _ANTHROPIC_PRICING.items():
        if key.startswith(model.split("-")[0:3][0]):
            return prices
    return (1.00, 5.00)


CONTEXT_PROMPT = (
    "Here is the chunk we want to situate within the whole document\n"
    "<chunk>\n{chunk_text}\n</chunk>\n"
    "Please give a short succinct context to situate this chunk within the overall "
    "document for the purposes of improving search retrieval of the chunk. "
    "Answer only with the succinct context and nothing else."
)


@dataclass
class CachedPrefix:
    chunk_index: int
    prefix: str
    model: str
    timestamp: str


class ContextualEnricher:
    def __init__(self, model: str, cache_dir: Path = Path("contextual_cache")):
        self.model = model
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self.client = anthropic.Anthropic()

    def _cache_path(self, strategy: str, config: Config, document_id: str) -> Path:
        model_short = self.model.replace("/", "_").replace(".", "_")[:20]
        filename = f"{strategy.lower()}_c{config.chunk_size}_o{config.chunk_overlap}_{model_short}_{document_id}.json"
        return self.cache_dir / filename

    def _load_cache(self, path: Path) -> dict[int, CachedPrefix]:
        if not path.exists():
            return {}
        with open(path) as f:
            data = json.load(f)
        return {
            entry["chunk_index"]: CachedPrefix(**entry)
            for entry in data
        }

    def _save_cache(self, path: Path, prefixes: dict[int, CachedPrefix]) -> None:
        data = [asdict(p) for p in sorted(prefixes.values(), key=lambda p: p.chunk_index)]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _generate_prefix(self, doc_text: str, chunk_text: str) -> tuple[str, dict]:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=150,
            system=[{
                "type": "text",
                "text": f"<document>\n{doc_text}\n</document>",
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{
                "role": "user",
                "content": CONTEXT_PROMPT.format(chunk_text=chunk_text),
            }],
        )
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cache_creation_input_tokens": getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
            "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", 0) or 0,
        }
        return response.content[0].text.strip(), usage

    def enrich_chunks(self, chunks: list[Chunk], documents: list[dict], config: Config) -> list[Chunk]:
        doc_texts = {doc["id"]: doc["text"] for doc in documents}
        strategy = chunks[0].strategy if chunks else "unknown"

        sorted_chunks = sorted(chunks, key=lambda c: c.source_id)
        grouped = {sid: list(grp) for sid, grp in groupby(sorted_chunks, key=lambda c: c.source_id)}

        total_usage = {"input_tokens": 0, "output_tokens": 0, "cache_write": 0, "cache_read": 0}
        total_api_calls = 0
        total_cache_hits = 0
        all_prefixes: dict[str, dict[int, CachedPrefix]] = {}

        for doc_id, doc_chunks in grouped.items():
            cache_path = self._cache_path(strategy, config, doc_id)
            cached = self._load_cache(cache_path)

            uncached_chunks = [c for c in doc_chunks if c.index not in cached]
            if not uncached_chunks:
                console.print(f"  [dim]Doc {doc_id}: {len(doc_chunks)} chunks loaded from cache[/dim]")
                total_cache_hits += len(doc_chunks)
                all_prefixes[doc_id] = cached
                continue

            doc_text = doc_texts.get(doc_id, "")
            doc_char_len = len(doc_text)
            if doc_char_len < 12000:
                console.print(f"  [yellow]Doc {doc_id}: ~{doc_char_len} chars — may be under 4096-token cache minimum[/yellow]")

            console.print(f"  Doc {doc_id}: generating {len(uncached_chunks)} prefixes ({len(doc_chunks) - len(uncached_chunks)} cached)...")

            doc_usage = {"input": 0, "output": 0, "cache_write": 0, "cache_read": 0}
            for i, chunk in enumerate(uncached_chunks):
                prefix, usage = self._generate_prefix(doc_text, chunk.text)
                cached[chunk.index] = CachedPrefix(
                    chunk_index=chunk.index,
                    prefix=prefix,
                    model=self.model,
                    timestamp=datetime.now().isoformat(),
                )
                doc_usage["input"] += usage["input_tokens"]
                doc_usage["output"] += usage["output_tokens"]
                doc_usage["cache_write"] += usage["cache_creation_input_tokens"]
                doc_usage["cache_read"] += usage["cache_read_input_tokens"]
                total_api_calls += 1

            cache_pct = doc_usage["cache_read"] / max(doc_usage["cache_read"] + doc_usage["cache_write"] + doc_usage["input"], 1) * 100
            console.print(f"    Cache hit ratio: {cache_pct:.0f}% | Tokens: {doc_usage['input']}in + {doc_usage['output']}out")

            total_usage["input_tokens"] += doc_usage["input"]
            total_usage["output_tokens"] += doc_usage["output"]
            total_usage["cache_write"] += doc_usage["cache_write"]
            total_usage["cache_read"] += doc_usage["cache_read"]

            self._save_cache(cache_path, cached)
            all_prefixes[doc_id] = cached

        if total_api_calls > 0:
            console.print(f"\n  [bold]Contextual enrichment summary:[/bold]")
            console.print(f"    API calls: {total_api_calls} | File cache hits: {total_cache_hits}")
            console.print(f"    Prompt cache writes: {total_usage['cache_write']} tokens")
            console.print(f"    Prompt cache reads: {total_usage['cache_read']} tokens")
            console.print(f"    Uncached input: {total_usage['input_tokens']} tokens")
            console.print(f"    Output: {total_usage['output_tokens']} tokens")
        else:
            console.print(f"  [bold green]All {total_cache_hits} prefixes loaded from file cache — no API calls needed[/bold green]")

        enriched = []
        for chunk in chunks:
            prefix_entry = all_prefixes.get(chunk.source_id, {}).get(chunk.index)
            if prefix_entry:
                enriched_text = f"{prefix_entry.prefix}\n\n{chunk.text}"
            else:
                enriched_text = chunk.text
            enriched.append(Chunk(
                text=enriched_text,
                index=chunk.index,
                strategy=chunk.strategy,
                source_id=chunk.source_id,
            ))
        return enriched
