# RAG Evaluation Pipeline — Zoomcamp FAQ Search

## Pipeline Architecture

```
documents.json (source)
    │
    ▼
data_cleaning/          ← download, parse, clean, dedup, ingest
    │
    ├── ES (faqs_complete)   ← 1150 docs with question_vector
    └── Qdrant (faqs)        ← 1150 docs, all-MiniLM-L6-v2 embeddings
    │
    ▼
eval/
├── judges/             ← LLM-as-judge: context sufficiency
├── benchmarks/         ← retrieval evaluation: BM25, vector, hybrid, Qdrant
├── generation/         ← test query generation (3 prompt strategies)
├── analysis/           ← dashboard, A/B tests, visualizations
└── shared.py           ← llm_call, run_sequential (rate-limit pacing)
    │
    ▼
gen/                    ← answer generation (planned)
```

---

## What We've Built

### 1. Data Cleaning Pipeline (`data_cleaning/`)

| Step | File | Description |
|------|------|-------------|
| Download | `download.py` | Pulls FAQ markdown from DataTalksClub/faq (4 courses) |
| Parse | `parse.py` | Extracts `id`, `question`, `answer`, `course`, `section` from frontmatter |
| Clean | (in `parse.py`) | Removes `<{IMAGE:...}>`, markdown headers, HTML, Jinja2 macros |
| Extract Metadata | `extract_metadata.py` | Saves image references to `metadata/images.json` for future use |
| Deduplicate | `dedup.py` | 95% similarity threshold on normalized text — removed 3 duplicates |
| Ingest (ES) | `ingest.py` | ES `faqs_complete` index with `question_vector` (384-dim dense_vector) |
| Ingest (Qdrant) | `ingest_qdrant.py` | Qdrant `faqs` collection with cosine similarity |
| Train/Test Split | `train_test_split.py` | 80/20 split by course (preserved for future holdout use) |

**Output:** 1150 clean, deduplicated documents in both ES and Qdrant.

### 2. Retrieval Methods

| Method | Backend | Description |
|--------|---------|-------------|
| BM25 | Elasticsearch | Lexical search with configurable question/text boosting |
| Vector | Elasticsearch | Cosine similarity on `question_vector` (all-MiniLM-L6-v2) |
| Hybrid | Elasticsearch | RRF fusion of BM25 + vector |
| Qdrant Vector | Qdrant | Cosine similarity on same embeddings |

**Configurations tested:** `bm25_default`, `bm25_balanced`, `bm25_high_question`, `bm25_high_text`, `vector_default`, `hybrid_default`, `hybrid_balanced`, `qdrant_default`

### 3. LLM-as-Judge Context Sufficiency (`eval/judges/`)

Evaluates whether retrieved contexts contain enough information to answer a question.

- **Model:** NVIDIA Llama 3.1 70B
- **Prompt:** Context delimiters (`<context_i>`), few-shot examples, reasoning before JSON
- **Output:** `judge_any_yes`, `judge_verdicts`, `judged_rank`, `reasoning`
- **Results:** Saved incrementally to `experiments/judge/<subset>_progress.csv`

### 4. Test Query Generation (`eval/generation/`)

Three prompt strategies to stress-test retrieval:

| Strategy | Style | Temperature |
|----------|-------|-------------|
| `grounded_analyst` | Technical, precise, uses domain terms | 0.7 |
| `creative_student` | Natural frustration, symptom descriptions | 0.7 |
| `chaos_monkey` | Wrong angles, abstract, unexpected connections | 0.7 |

- Shows 3 Q&A pairs per LLM call
- Model sees both question AND answer to generate realistic rephrased queries
- Prompts stored in `eval/generation/prompts.json`

### 5. Evaluation Framework (`eval/benchmarks/` + `eval/analysis/`)

**Metrics:**
- Recall@K (1, 3, 5, 10)
- MRR (Mean Reciprocal Rank)
- Precision@K
- Latency percentiles (P50, P95, P99)
- A/B testing between configs

**Running evaluations:**
```bash
# Full benchmark
uv run python eval/benchmarks/run_full_benchmark.py

# Single config
uv run python eval/benchmarks/run_full_benchmark.py --config qdrant_default

# Qdrant-specific eval suite
uv run python eval/benchmarks/qdrant_evaluation.py
```

### 6. Rate-Limit Handling (`eval/judges/shared.py`)

- **`run_sequential`**: Fixed gap between calls, 60s wait on 429/504 errors
- **Consistent pacing**: 3-second gap by default
- **Granular output**: Shows status (✓/✗), elapsed time, and wait periods

---

## Current State of the Repo

```
eval/
├── generation/
│   ├── generate_test_queries.py   # Batch query generation (3 strategies × 3 docs)
│   ├── test_gen.py                # Single-doc test with full transparency
│   └── prompts.json               # 3 prompt templates
├── judges/
│   ├── run_judge.py               # Context sufficiency evaluation
│   ├── calibrate_judge.py         # Model calibration (can it say NO?)
│   ├── classify_failures.py       # Failure analysis
│   └── shared.py                  # llm_call, run_sequential, parse_verdicts
├── benchmarks/
│   ├── benchmark_runner.py        # ES benchmark runner
│   ├── qdrant_benchmark_runner.py # Qdrant benchmark runner
│   ├── qdrant_evaluation.py       # Qdrant eval suite (5 tests)
│   ├── run_full_benchmark.py      # Run all configs
│   ├── compare_retrievers.py      # Side-by-side comparison
│   └── test_holdout.py            # Holdout test with train/test split
├── analysis/
│   ├── dashboard.py               # Main dashboard with conclusions
│   ├── visualizer.py              # matplotlib/seaborn plots
│   └── ab_test.py, stats.py       # Statistical analysis
└── shared.py                      # Shared utilities (deprecated, use judges/shared.py)
```

---

## Key Decisions Made

1. **No query truncation** — full questions sent to judge (max_tokens=2000)
2. **Answers shown during generation** — model uses answer context to generate better rephrased queries
3. **Qdrant as primary vector store** — cleaner API, same embeddings
4. **60s wait on rate limits** — simpler than adaptive gap adjustment
5. **Dedup by question+answer similarity (95%)** — keeps cross-course variations
6. **Images extracted to metadata** — removed from answer text for cleaner retrieval

---

## Next Steps

| Step | Status |
|------|--------|
| Generate full test query set (60+ docs) | Ready to run |
| Run all retrievers against generated queries | Pending |
| Compare hit rates across retrievers | Pending |
| Answer generation pipeline (`gen/`) | Planned |
| Answer quality judging | Planned |