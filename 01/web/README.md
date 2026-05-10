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
├── analysis/           ← dashboard, A/B tests, plots, significance tests
└── (shared.py)         ← llm_call, run_sequential (rate-limit pacing)
    │
    ▼
gen/                    ← CAG: Cache-Augmented Generation (planned)
```

---

## Data Cleaning (`data_cleaning/`)

| Step | File | Description |
|------|------|-------------|
| Download | `download.py` | Pulls FAQ markdown from DataTalksClub/faq (4 courses) |
| Parse | `parse.py` | Extracts `id`, `question`, `answer`, `course`, `section` from markdown |
| Clean | (in `parse.py`) | Removes `<{IMAGE:...}>`, markdown headers, HTML, Jinja2 macros |
| Extract Metadata | `extract_metadata.py` | Saves image references to `metadata/images.json` |
| Deduplicate | `dedup.py` | 95% similarity threshold on normalized text — removed 3 duplicates |
| Ingest (ES) | `ingest.py` | ES `faqs_complete` index with `question_vector` (384-dim dense_vector) |
| Ingest (Qdrant) | `ingest_qdrant.py` | Qdrant `faqs` collection with cosine similarity |
| Train/Test Split | `train_test_split.py` | 80/20 split by course |

**Output:** 1150 clean, deduplicated documents in both ES and Qdrant.

---

## Retrieval Methods

| Method | Backend | Description |
|--------|---------|-------------|
| BM25 | Elasticsearch | Lexical search with configurable question/text boosting |
| Vector | Elasticsearch | Cosine similarity on `question_vector` (all-MiniLM-L6-v2) |
| Hybrid | Elasticsearch | RRF or linear weighted fusion of BM25 + vector |
| Qdrant Vector | Qdrant | Cosine similarity on same embeddings |
| Cross-encoder Rerank | ES + CrossEncoder | Retrieve 20, rerank to 10 with cross-encoder/ms-marco-MiniLM-L-6-v2 |

### Configurations Tested

| Config | Type | Details |
|--------|------|---------|
| `bm25_default` | BM25 | question^20, answer^1 |
| `bm25_balanced` | BM25 | question^5, answer^5 |
| `bm25_high_text` | BM25 | question^1, answer^10 |
| `vector_cosine` | Vector | Cosine similarity |
| `hybrid_rrf` | Hybrid | Reciprocal Rank Fusion (k=60) |
| `hybrid_50_50` | Hybrid | Linear 50% BM25 + 50% vector |
| `hybrid_70_30_vec` | Hybrid | Linear 30% BM25 + 70% vector |
| `hybrid_30_70_vec` | Hybrid | Linear 70% BM25 + 30% vector |
| `qdrant_cosine` | Qdrant | Cosine distance |
| `vector_reranked` | Rerank | Vector + cross-encoder rerank |
| `bm25_reranked` | Rerank | BM25 + cross-encoder rerank |

---

## Test Query Generation (`eval/generation/`)

Three LLM prompt strategies generate rephrased queries from existing FAQ entries:

| Strategy | Style | Temperature |
|----------|-------|-------------|
| `grounded_analyst` | Technical, precise, uses domain terms | 0.7 |
| `creative_student` | Natural frustration, symptom descriptions | 0.7 |
| `chaos_monkey` | Wrong angles, abstract, unexpected connections | 0.7 |

- Model: NVIDIA Llama 3.1 70B
- Batch size: 5 FAQ entries per LLM call
- Output: `experiments/eval_queries.json` — 390 queries across 50 documents

---

## Evaluation Results

### Retrieval Performance (390 Generated Queries)

| Retriever | Hit | R@1 | R@5 | MRR | NDCG | Fail | P50ms | P95ms |
|-----------|-----|-----|-----|-----|------|------|-------|-------|
| bm25_default | 82.82% | 53.85% | 75.13% | 0.6357 | 0.6822 | 67 | 5.4 | 6.5 |
| bm25_balanced | 90.26% | 66.92% | 87.44% | 0.7607 | 0.7960 | 38 | 5.3 | 6.3 |
| bm25_high_text | 81.03% | 55.38% | 76.67% | 0.6441 | 0.6846 | 74 | 5.4 | 6.4 |
| vector_cosine | 93.59% | 66.67% | 87.69% | 0.7574 | 0.8006 | 25 | 19.3 | 24.2 |
| qdrant_cosine | 93.59% | 66.67% | 87.69% | 0.7574 | 0.8006 | 25 | 20.3 | 24.1 |
| hybrid_rrf | 91.79% | 59.74% | 85.38% | 0.7048 | 0.7564 | 32 | 30.3 | 35.2 |
| hybrid_50_50 | 92.05% | 60.51% | 86.92% | 0.7221 | 0.7710 | 31 | 30.4 | 35.2 |
| **hybrid_70_30_vec** | **93.85%** | 66.67% | **90.26%** | **0.7609** | **0.8042** | **24** | 30.6 | 35.6 |
| hybrid_30_70_vec | 90.26% | 53.85% | 85.90% | 0.6750 | 0.7313 | 38 | 30.1 | 35.0 |
| vector_reranked | 91.79% | 61.54% | 84.36% | 0.7156 | 0.7644 | 32 | 580.8 | 700.2 |
| bm25_reranked | 84.87% | 59.74% | 80.26% | 0.6859 | 0.7255 | 59 | 578.3 | 688.8 |

### Per-Strategy Breakdown (best config: hybrid_70_30_vec)

| Strategy | Hit | R@5 | MRR |
|----------|-----|-----|-----|
| grounded_analyst | 97.33% | 96.00% | 0.8516 |
| creative_student | 99.05% | 95.24% | 0.7660 |
| chaos_monkey | 85.93% | 80.00% | 0.6562 |

### Per-Course Breakdown (best config: hybrid_70_30_vec)

| Course | R@5 |
|--------|------|
| de-zoomcamp | 91.67% |
| llm-zoomcamp | 98.67% |
| ml-zoomcamp | 83.33% |
| mlops-zoomcamp | 92.16% |

---

## Key Findings

1. **Best overall: `hybrid_70_30_vec`** — 90.26% R@5, 0.8042 NDCG, 24 failures
2. **BM25 balanced is the fastest** — 5.3ms P50 vs 30.6ms for hybrid (5.8x faster), with only 2.82% lower R@5
3. **Cross-encoder reranking HURTS** — loses 3-4% R@5 and costs 30x more latency
4. **chaos_monkey queries are hardest** — 80% R@5 vs 96% for grounded/creative
5. **ml-zoomcamp is the hardest course** — 83% R@5 vs 99% for llm-zoomcamp
6. **Qdrant and ES vector perform identically** — same embeddings, same results
7. **Zero cross-course contamination** — course filters are working perfectly

---

## Running Evaluations

```bash
# Full benchmark (self-retrieval)
uv run python eval/benchmarks/run_full_benchmark.py

# Single config
uv run python eval/benchmarks/run_full_benchmark.py --config bm25_default

# Generated query evaluation (all configs)
uv run python eval/benchmarks/test_variations.py

# Dashboard with plots (notebook)
from eval.analysis.dashboard import show_dashboard
show_dashboard()

# A/B comparison
uv run python eval/analysis/variations_ab_test.py --a bm25_balanced --b hybrid_70_30_vec

# Save plots to files
uv run python eval/analysis/visualizer.py
```

---

## Project Structure

```
.
├── configs/                    # Search configs, ES settings, API keys
├── data_cleaning/              # Data pipeline (download → ingest)
├── eval/
│   ├── judges/                 # Context sufficiency evaluation
│   ├── benchmarks/             # Retrieval performance tests
│   ├── generation/             # Test query generation
│   └── analysis/               # Dashboard, A/B tests, plots
├── experiments/
│   ├── results/                # Benchmark output JSON files
│   ├── judge/                  # Judge progress CSVs
│   ├── plots/                  # Saved visualization PNGs
│   └── eval_queries.json       # Generated test queries
├── gen/                        # CAG pipeline (planned)
├── src/                        # Core application code
│   └── retrieval/              # BM25, vector, hybrid, Qdrant retrievers
└── docker-compose*.yaml        # ES, Qdrant, and app containers
```

# RAG Evaluation Pipeline — Zoomcamp FAQ Search

## Final Results (bge-base 768d, open search, no course filter)

| Config | R@1 | R@5 | P50ms |
|--------|-----|-----|-------|
| BM25 (q^5+a^5) | 58.3% | 79.0% | 5.0ms |
| ES kNN (bge-base 768d) | 60.2% | 81.9% | 6.7ms |
| Qdrant (bge-base 768d) | 60.7% | 82.4% | 4.8ms |
| **Hybrid: BM25 + Qdrant** | **61.9%** | **86.9%** | **9.8ms** |
| Hybrid: BM25 + ES kNN | 61.7% | 86.7% | 11.7ms |

**Winner: BM25 (q^5+a^5) + Qdrant (bge-base 768d) with RRF fusion — 86.9% R@5**

### Model Comparison (Qdrant only)

| Model | Dims | R@5 | Enc(ms) |
|-------|------|-----|---------|
| bge-small-en-v1.5 | 384 | 80.0% | 41ms |
| e5-small-v2 | 384 | 78.3% | 37ms |
| **bge-base-en-v1.5** | **768** | **82.4%** | 122ms |
| e5-base-v2 | 768 | 81.7% | 124ms |

### Per-Strategy (best config)

| Strategy | R@5 | Notes |
|----------|-----|-------|
| grounded_analyst | 96% | Technical, precise queries |
| creative_student | 95% | Natural frustration, symptom-based |
| chaos_monkey | 80% | Wrong angles, high temperature |

### Per-Course (best config)

| Course | R@5 | Docs |
|--------|-----|------|
| llm-zoomcamp | 100% | 79 |
| de-zoomcamp | 89% | 393 |
| mlops-zoomcamp | 85% | 245 |
| ml-zoomcamp | 82% | 433 |

---

## Key Findings

1. **86.9% R@5 open search** — no course filter, real-world performance
2. **BM25 + vector hybrid** adds 7.9% over BM25 alone
3. **bge-base 768d** beats bge-small 384d by 2.4% but encodes 3x slower
4. **E5 models underperformed** BGE on full retrieval despite higher single-query similarity
5. **Three-way hybrid is WORSE** than two-way — more sources add noise
6. **Cross-encoder reranking hurts** — loses 3-4% R@5 and costs 30x more latency
7. **chaos_monkey queries are the bottleneck** — 80% vs 96% for other strategies
8. **ml-zoomcamp is the hardest course** — 82% R@5 with 433 docs
9. **BM25 searching answer fields** closes the vocabulary gap — question^5 + answer^5 is optimal

---

## Next Steps

| Step | Status |
|------|--------|
| Data cleaning & ingestion | ✅ Complete |
| Test query generation (3 strategies) | ✅ Complete |
| BM25, vector, hybrid, Qdrant benchmarks | ✅ Complete |
| Embedding model comparison (4 models) | ✅ Complete |
| Cross-encoder reranking evaluation | ✅ Complete |
| Per-strategy / per-course breakdown | ✅ Complete |
| Failure deep-dive analysis | ✅ Complete |
| A/B testing with significance | ✅ Complete |
| Dashboard with inline plots | ✅ Complete |
| CAG pipeline (32 answers generated) | 📋 In progress |
| Scale CAG to all 1140 FAQs | 📋 Planned |




## Course Routing Strategy

The FAQ bot is deployed across course-specific Slack channels. When a question comes from `#de-zoomcamp`, we automatically apply a course filter — this is not keyword guessing, it's known context from the channel.

**This gives us 92.4% R@5 at 9.9ms** — up from 86.9% open search.

| Scenario | R@5 | How |
|----------|-----|-----|
| Open search (unknown course) | 86.9% | No filter, cross-course retrieval |
| **Slack channel (known course)** | **92.4%** | Course filter from channel context |
| Keyword-based smart routing | 85.2% | Not recommended — keywords cause wrong routes |

**Bottom line:** Don't guess the course. If you know it (Slack channel), filter. If you don't, search open.

---

## Final Results (bge-base 768d, BM25 + Qdrant RRF)

| Config | R@1 | R@5 | P50ms | Use Case |
|--------|-----|-----|-------|----------|
| Open search | 61.9% | 86.9% | 9.8ms | Unknown context |
| **Course filter** | — | **92.4%** | **9.9ms** | **Slack channels** |

### Model Comparison (Qdrant only)

| Model | Dims | R@5 | Enc(ms) |
|-------|------|-----|---------|
| bge-small-en-v1.5 | 384 | 80.0% | 41ms |
| e5-small-v2 | 384 | 78.3% | 37ms |
| **bge-base-en-v1.5** | **768** | **82.4%** | 122ms |
| e5-base-v2 | 768 | 81.7% | 124ms |

### Per-Strategy (open search, best config)

| Strategy | R@5 | Notes |
|----------|-----|-------|
| grounded_analyst | 98.7% | Technical, precise queries |
| creative_student | 92.5% | Natural frustration, symptom-based |
| chaos_monkey | 71.3% | Wrong angles, high temperature — acceptable loss |

### Per-Course (open search, best config)

| Course | R@5 | Docs |
|--------|-----|------|
| llm-zoomcamp | 100% | 79 |
| de-zoomcamp | 89% | 392 |
| mlops-zoomcamp | 85% | 243 |
| ml-zoomcamp | 82% | 426 |

### Signal Quality (keyword routing — not recommended)

| Course | Correct | Missed | Wrong |
|--------|---------|--------|-------|
| de-zoomcamp | 70 | 62 | 0 |
| llm-zoomcamp | 12 | 63 | 6 |
| ml-zoomcamp | 32 | 111 | 4 |
| mlops-zoomcamp | 28 | 31 | 1 |

Keyword-based routing has 11 wrong routes that hurt more than the 142 correct routes help. Don't use it.

---

## Key Findings

1. **92.4% R@5 with course filter** — deploy to Slack channels for free 5.5% gain
2. **86.9% R@5 open search** — real-world performance without context
3. **bge-base 768d** beats bge-small 384d by 2.4% but encodes 3x slower
4. **BM25 + Qdrant RRF** is the optimal config — three-way hybrid adds noise
5. **Cross-encoder reranking hurts** — loses 3-4% R@5, 30x slower
6. **17 of 20 tools appear in 3+ courses** — open search is the right default
7. **chaos_monkey is the bottleneck** — 71% vs 99% for grounded queries
8. **Keyword routing can't work** — shared vocabulary across courses means wrong routes are inevitable

---

## Next Steps

| Step | Status |
|------|--------|
| Data cleaning & ingestion (1140 docs) | ✅ Complete |
| Test query generation (420 queries, 3 strategies) | ✅ Complete |
| BM25, vector, hybrid benchmarking (11 configs) | ✅ Complete |
| Embedding model comparison (4 models) | ✅ Complete |
| Cross-encoder reranking | ✅ Complete |
| Smart routing evaluation (keyword + semantic) | ✅ Complete |
| Per-strategy / per-course breakdown | ✅ Complete |
| Failure deep-dive analysis | ✅ Complete |
| Dashboard with inline plots | ✅ Complete |
| CAG pipeline (32/1140 answers) | 📋 In progress |
| Deploy to Slack with course filter | 📋 Planned |