## 📊 Search Evaluation: The Baseline (BM25)

### The Question
**"How effective is global keyword search at distinguishing between similar FAQ entries across multiple distinct courses?"**

### Relevant Facts (The Baseline)
*   **Dataset:** 948 FAQ documents from 3 courses (Data Engineering, Machine Learning, MLOps).
*   **Evaluation Set:** 30 deterministic samples (Top 10 questions from each course).
*   **Stable Identity:** Every document is assigned a unique `hash(course + question + text_snippet)`. This ensures that even identical questions in different courses are treated as unique entities.
*   **Global Search Space:** Search is performed across the entire index (`N=948`) without "hints" or course filters.

### Current Performance (Top-K Sweep)

| Metric | Score | Observation |
| :--- | :--- | :--- |
| **Recall@1** | **33.33%** | 2 out of 3 queries fail to find the exact document at Rank #1. |
| **Recall@5** | **33.33%** | The plateau suggests the "correct" doc isn't just ranked low; it's being out-competed. |

### Key Findings & Diagnostic Insights
Our `eval_diagnostic.md` revealed three specific reasons for the low baseline:

*   **Cross-Course Collision:** Many questions (e.g., *"How do I join Slack?"*) exist in multiple courses. Without a course filter, BM25 returns the most "keyword-heavy" version. Even if the answer text is identical, the **ID mismatch** results in an evaluation failure.
*   **The "Word Counter" Trap:** BM25 rewards term frequency. Short, specific FAQ titles are often outranked by longer documents that repeat the same keywords more often in the body text.
*   **Semantic Blindness:** Queries like *"I don't know math"* sometimes return general course descriptions instead of the specific "Math Prerequisites" FAQ because they share common stop-words.

### Conclusion for Next Steps
The **33.33% Recall** is our "Semantic Gap." This justifies the move toward **Vector Search (Embeddings)** or **Hybrid Search**, which should theoretically resolve these collisions by understanding the context and "vibe" of the query rather than just counting word occurrences.


### 🕵️ Diagnostic Deep-Dive: Why 33%?
The diagnostic report (`eval_diagnostic.md`) shows the 33% score is misleading due to:
*   **Semantic Duplicates:** ES often finds the "correct" answer text, but from a different record ID (e.g., 2024 version vs 2025 version).
*   **Dirty Ground Truth:** Some "Expected" documents are just FAQ headers/guidelines, while ES finds the actual answer in a different record.
*   **Redundancy:** Identical answers exist across courses; without a filter, BM25 picks the one with the highest term frequency.
