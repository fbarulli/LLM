### 1. Cutoff Threshold (Score-Based)
Filters out documents that fall below a strict similarity score before sending them to the LLM.

* **Specific Use Case**: Thinning out low-quality "filler" matches for highly specific, factual lookup queries.
* **Latency Profile**: Minimal impact. The score filtering happens instantly within the database or code.
* **Budget Profile**: Highly positive. Actively shrinks prompt context by removing useless documents, lowering input token costs.
* **Cost vs. Accuracy Optimization**: High cost savings with almost zero accuracy loss. Only "noise" is discarded.
* **Compatible Caching**: **Result Caching** (caching the filtered Top-K IDs associated with normalized query strings).

---

### 2. Maximal Marginal Relevance (MMR)
Pulls a large candidate pool and runs matrix math to pick items that are highly relevant but have low similarity to each other.

* **Specific Use Case**: Broad topic summaries where you need to prevent the LLM from reading 5 overlapping, duplicate documents.
* **Latency Profile**: Adds minor compute overhead (tens of milliseconds) to calculate document-to-document diversity scores in Python.
* **Budget Profile**: Highly positive. It replaces 5 repetitive documents with 5 unique ones, making every paid token count.
* **Cost vs. Accuracy Optimization**: Sacrifices pure relevance ranking to achieve high-breadth accuracy. Excellent for maximizing context window utility.
* **Compatible Caching**: **Embedding Caching** (storing document vectors to avoid re-running dot products during the MMR loop).

---

### 3. Two-Stage Retrieval (Reranking)
Pulls a wide net of candidates (e.g., K=50) via fast search, then runs a dense Cross-Encoder model to bubble the absolute top 3-5 to the prompt.

* **Specific Use Case**: High-stakes accuracy pipelines (like legal or medical data) where finding the perfect needle in a haystack is mandatory.
* **Latency Profile**: Heavy impact. Running a secondary cross-encoder model can add 100ms to 500ms of execution time per request.
* **Budget Profile**: Negative up-front (running the GPU rerank model), but positive on the API side by trimming the final prompt to the absolute best 3 docs.
* **Cost vs. Accuracy Optimization**: Prioritizes extreme accuracy over cost. You are paying for a secondary model to ensure the prompt is flawless.
* **Compatible Caching**: **Semantic Caching** (caching the final re-ranked results for semantically similar user questions).

---

### 4. Agentic / Adaptive Retrieval
An LLM analyzes the user's prompt before searching to decide exactly how many documents to fetch or which tools to trigger.

* **Specific Use Case**: Complex, multi-hop user questions that require searching multiple databases or breaking questions into parts.
* **Latency Profile**: Severe impact. Requires at least one full LLM generation loop ("thinking") before it even begins the database search.
* **Budget Profile**: Highly negative. You are paying for multiple LLM calls (classification, execution, synthesis) to answer a single prompt.
* **Cost vs. Accuracy Optimization**: The peak of accuracy and reasoning capabilities, but financially unviable for high-traffic, low-cost user bots.
* **Compatible Caching**: **Final Answer Caching** (caching the full, heavily-processed answer to exact or highly similar complex questions).
