from __future__ import annotations

import random
from dataclasses import dataclass, field

from datasets import load_dataset

from config import Config

DATASET_REGISTRY: dict[str, type[DatasetLoader]] = {}


@dataclass
class QAPair:
    question: str
    answer: str
    context_id: str
    options: list[str] = field(default_factory=list)
    gold_label: int | None = None
    difficult: bool = False

    @property
    def is_multiple_choice(self) -> bool:
        return len(self.options) > 0 and self.gold_label is not None


class DatasetLoader:
    name: str

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "name"):
            DATASET_REGISTRY[cls.name] = cls

    def load(self, config: Config) -> tuple[list[dict], list[QAPair]]:
        raise NotImplementedError


class SQuADLoader(DatasetLoader):
    name = "squad"

    def load(self, config: Config) -> tuple[list[dict], list[QAPair]]:
        ds = load_dataset("rajpurkar/squad", split="validation")

        contexts: dict[str, str] = {}
        qa_by_context: dict[str, list[QAPair]] = {}

        for row in ds:
            ctx = row["context"]
            ctx_id = row["id"]

            if ctx not in contexts.values():
                contexts[ctx_id] = ctx
                qa_by_context[ctx_id] = []

            matched_id = next(k for k, v in contexts.items() if v == ctx)
            qa_by_context[matched_id].append(
                QAPair(
                    question=row["question"],
                    answer=row["answers"]["text"][0],
                    context_id=matched_id,
                )
            )

        all_ctx_ids = list(contexts.keys())
        random.seed(config.random_seed)
        sampled_ids = random.sample(all_ctx_ids, min(config.num_contexts, len(all_ctx_ids)))

        documents = [{"id": cid, "text": contexts[cid]} for cid in sampled_ids]
        qa_pairs = [qa for cid in sampled_ids for qa in qa_by_context[cid]]

        return documents, qa_pairs


class QuALITYLoader(DatasetLoader):
    name = "quality"

    def load(self, config: Config) -> tuple[list[dict], list[QAPair]]:
        ds = load_dataset("tasksource/QuALITY", split="train")

        articles: dict[str, str] = {}
        qa_by_article: dict[str, list[QAPair]] = {}

        for row in ds:
            aid = str(row["article_id"])
            if aid not in articles:
                articles[aid] = row["article"]
                qa_by_article[aid] = []

            gold_idx = row["gold_label"] - 1
            qa_by_article[aid].append(
                QAPair(
                    question=row["question"],
                    answer=row["options"][gold_idx],
                    context_id=aid,
                    options=row["options"],
                    gold_label=gold_idx,
                    difficult=bool(row.get("difficult", 0)),
                )
            )

        all_ids = list(articles.keys())
        random.seed(config.random_seed)
        sampled_ids = random.sample(all_ids, min(config.num_contexts, len(all_ids)))

        documents = [{"id": aid, "text": articles[aid]} for aid in sampled_ids]
        qa_pairs = [qa for aid in sampled_ids for qa in qa_by_article[aid]]

        return documents, qa_pairs


def load_dataset_sample(config: Config) -> tuple[list[dict], list[QAPair]]:
    loader_cls = DATASET_REGISTRY.get(config.dataset)
    if loader_cls is None:
        available = ", ".join(sorted(DATASET_REGISTRY.keys()))
        raise ValueError(f"Unknown dataset '{config.dataset}'. Available: {available}")
    return loader_cls().load(config)
