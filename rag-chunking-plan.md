# RAG Chunking Strategy Comparison — Project Plan

## Goal

Build a Python CLI project that compares three chunking strategies (Fixed, Recursive, Semantic) side by side using real data from the SQuAD dataset, ChromaDB as the vector store, and Sentence Transformers for embeddings. The comparison should show which strategy produces the best retrieval quality for a RAG pipeline.

## Environment

- Python managed with **uv** (already installed)
- Use `uv init`, `uv add`, and `uv run` throughout — no pip, no venv manually
- Project setup:
  ```bash
  uv init rag-chunking-comparison
  cd rag-chunking-comparison
  uv add langchain langchain-text-splitters langchain-experimental \
       datasets sentence-transformers chromadb numpy matplotlib rich
  ```

---

## Data Source: SQuAD (Stanford Question Answering Dataset)

Load SQuAD v1.1 using HuggingFace's `datasets` library:

```python
from datasets import load_dataset
ds = load_dataset("squad")
```

Each record in SQuAD has this structure:

```json
{
  "id": "56be8553...",
  "title": "Beyoncé",
  "context": "A Wikipedia paragraph...",
  "question": "A human-written question about the paragraph",
  "answers": {
    "text": ["exact answer span"],
    "answer_start": [269]
  }
}
```

- **context** = a Wikipedia paragraph (the document to chunk and ingest)
- **question** = a natural language question (the test query)
- **answers.text** = the exact text span that answers the question (ground truth)

Multiple questions can share the same context. For this project, sample 100–200 unique context paragraphs and all their associated question-answer pairs. Group by `title` or `context` to avoid duplicates.

---

## Chunking Strategies (using LangChain)

Use LangChain's built-in text splitters. Wrap each one in a thin adapter so they all expose the same interface to the rest of the codebase.

### 1. Fixed-Size Chunking

```python
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator="",           # no separator = blind character split
    chunk_size=500,
    chunk_overlap=50,
)
```

Cuts every N characters regardless of content. The baseline.

### 2. Recursive Chunking

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],
)
```

Tries paragraph breaks first, then sentences, then words. Keeps semantic units together when possible.

### 3. Semantic Chunking

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile",
)
```

Embeds every sentence and places breakpoints where topic shifts. The smartest strategy but slowest.

---

## Embeddings

Use Sentence Transformers with the `all-MiniLM-L6-v2` model.

- Free, local, no API key needed
- ~80MB download, 384-dimensional vectors
- Fast enough for this scale
- ChromaDB has built-in integration — pass a `SentenceTransformerEmbeddingFunction` when creating a collection, and ChromaDB handles embedding automatically on insert and query

Design the embedding model as a config option so it can be swapped later (e.g., to OpenAI's `text-embedding-3-small` via ChromaDB's `OpenAIEmbeddingFunction`).

---

## Vector Store: ChromaDB

Use ChromaDB in embedded mode (no server, just `uv add chromadb`).

- Create a **separate collection per chunking strategy** (e.g., `fixed_chunks`, `recursive_chunks`, `semantic_chunks`)
- Each collection uses the same embedding function
- Store chunk text as documents, with metadata like `source_context_id`, `strategy`, `chunk_index`
- Query each collection with the same test questions and retrieve `top_k` results (default k=3)

---

## Evaluation Metrics

Three tiers of comparison:

### Tier 1: Chunk Statistics
- Total number of chunks produced
- Average, median, min, max chunk size (in characters)
- Standard deviation of chunk sizes (measures uniformity)
- Chunking time

### Tier 2: Retrieval Relevance
- For each test query, retrieve top-k chunks from each strategy's ChromaDB collection
- Report the cosine similarity scores
- Compute mean relevance score per strategy across all queries

### Tier 3: Answer Hit Rate (the most important metric)
- Since SQuAD provides the ground truth answer text, check whether any of the top-k retrieved chunks actually **contain** the answer string
- Report hit rate as a percentage: "out of N queries, what % of the time did the top-k chunks contain the correct answer?"
- This directly measures which chunking strategy would give an LLM the right context to answer correctly

---

## Project Structure

```
rag-chunking-comparison/
├── pyproject.toml         # Managed by uv — dependencies and project metadata
├── config.py              # All tunable parameters in one place
├── data_loader.py         # Load and sample from SQuAD dataset
├── chunkers.py            # LangChain splitter wrappers with a shared interface
├── vector_store.py        # ChromaDB wrapper — create collections, insert, query
├── evaluator.py           # Compute stats, retrieval relevance, answer hit rate
├── main.py                # CLI entry point — orchestrates the full pipeline
└── README.md              # Setup instructions and usage
```

### config.py
Central place for all knobs:
- `CHUNK_SIZE`, `OVERLAP` (for Fixed and Recursive)
- `SEMANTIC_BREAKPOINT_TYPE` (default `"percentile"`)
- `EMBEDDING_MODEL_NAME` (default `"all-MiniLM-L6-v2"`)
- `TOP_K` (default 3)
- `NUM_CONTEXTS` (how many SQuAD passages to sample, default 100)
- `CHROMA_PERSIST_DIR` (where ChromaDB stores data)

### data_loader.py
- Load SQuAD via HuggingFace `datasets`
- Extract unique context paragraphs (deduplicate by context text)
- Sample `NUM_CONTEXTS` paragraphs
- Collect all question-answer pairs associated with those paragraphs
- Return two things: a list of documents (contexts) and a list of test QA pairs

### chunkers.py
- Thin wrappers around the three LangChain splitters
- Each wrapper takes a text and a `source_id`, runs the LangChain splitter, and returns a list of `Chunk` objects (dataclass with `text`, `index`, `strategy`, `source_id`, `metadata`)
- `source_id` links each chunk back to its original SQuAD context — needed for answer hit rate evaluation
- A `get_all_strategies(config)` function that returns a dict of `{"Fixed": splitter, "Recursive": splitter, "Semantic": splitter}` ready to use

### vector_store.py
- Wraps ChromaDB
- `create_collection(strategy_name)` — creates a collection with the Sentence Transformer embedding function
- `add_chunks(collection, chunks)` — inserts chunk texts and metadata
- `query(collection, query_text, top_k)` — returns top-k results with distances and metadata
- `reset()` — clears all collections for a fresh run

### evaluator.py
- `compute_chunk_stats(chunks)` → chunk count, size distribution, timing
- `evaluate_retrieval(collection, qa_pairs, top_k)` → per-query relevance scores
- `compute_hit_rate(collection, qa_pairs, top_k)` → percentage of queries where top-k contains the answer
- `full_report(strategy_name, chunks, collection, qa_pairs)` → combines all three tiers

### main.py
Pipeline flow:
1. Load and sample SQuAD data
2. For each strategy:
   a. Chunk all documents
   b. Create a ChromaDB collection
   c. Insert all chunks
   d. Run all test queries
   e. Generate the evaluation report
3. Print a side-by-side comparison table (use `rich` library for nice terminal output)
4. Save a comparison chart as PNG (use `matplotlib`)
5. Print sample chunks from each strategy so the user can eyeball the differences

Run with: `uv run python main.py`

---

## Output

The program should produce:

1. **Stats table** — side-by-side chunk count, avg size, std dev, timing per strategy
2. **Retrieval table** — per-query top-k results with similarity scores for each strategy
3. **Answer hit rate table** — the headline metric, percentage per strategy
4. **Bar chart PNG** — visual comparison of chunk count, avg size, and answer hit rate
5. **Sample chunks** — first 2-3 chunks from each strategy printed side by side so you can see how each one splits the same text

---