## 📊 Search Evaluation: The Baseline (BM25)

### The Question
**"How effective is lexical search at retrieving FAQ answers across multiple distinct courses?"**

### Dataset
- **948 FAQ documents** from 3 courses (Data Engineering, Machine Learning, MLOps)
- **Evaluation Set:** 30 deterministic samples (Top 10 questions from each course)
- **Stable Identity:** Document ID = `hash(course + cleaned_question + text_snippet[:50])`

### What We Fixed

| Problem | Solution |
|---------|----------|
| IDs mismatched between eval set and Elasticsearch | Fixed `run_stats.py` to clean questions before generating IDs |
| Results saved to wrong directory | Fixed path logic in `stats.py` and `visualizer.py` |
| One problematic ML document polluted global search | Deleted `"Is it going to be live? When?"` from index |
| Data Engineering recall at 0.00 | Root cause was ID mismatch, not search quality |

### Final Results (Filtered Search with course_context)

| Metric | Result | Interpretation |
| :--- | :--- | :--- |
| **Recall@1** | **100%** | Exact document found at Rank #1 for all queries |
| **Recall@5** | **100%** | Target document always within top 5 |
| **Avg Latency** | **~2ms** | Extremely high performance |
| **Cross-Course Rate** | 0% | Filtered search stays within course |

### Global Search Results (No course_context)

| Config | Recall@5 | Cross-Course Rate |
| :--- | :--- | :--- |
| baseline_bm25 | 100% | 0% |
| global_cross_fields | 96.67% | 15% |
| most_fields | 100% | 0% |

### A/B Test: baseline_bm25 vs global_cross_fields

| Result | Count |
| :--- | :--- |
| Config B (global_cross_fields) wins | 5 |
| Config A (baseline_bm25) wins | 2 |
| Ties | 2 |

**Key Insight:** `global_cross_fields` produces higher scores but occasionally returns answers from different courses. When judging by answer quality alone (not course origin), it outperforms the filtered baseline.

### Key Findings

1. **Lexical search is highly effective** when the search space is narrowed by course context
2. **ID stability is critical** for deterministic evaluation - any change in question cleaning breaks traceability
3. **One poorly written document** can pollute global search results across multiple queries
4. **Course origin doesn't determine correctness** - overlapping content means answers from other courses can be perfectly valid

### Next Steps

1. **Implement Vector Search (Embeddings)** to improve global search without metadata filters
2. **Add LLM-as-Judge** for large-scale A/B testing based on answer quality
3. **Create harder eval set** with paraphrased and cross-course queries