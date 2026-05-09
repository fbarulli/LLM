"""
eval/benchmarks/test_variations.py
===================================
Tests retrieval variations against generated queries.
Includes per-strategy breakdown, per-course breakdown, and failure analysis.

Run:    uv run python eval/benchmarks/test_variations.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json, time, numpy as np
from datetime import datetime
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from qdrant_client import QdrantClient

MODEL_NAME = 'all-MiniLM-L6-v2'
ES_INDEX = 'faqs_complete'
QDRANT_COLLECTION = 'faqs'
K_VALUES = [1, 3, 5, 10]

def load_test_queries(path='experiments/eval_queries.json'):
    if not os.path.exists(path):
        print(f"No {path}")
        return []
    with open(path) as f:
        data = json.load(f)
    queries = []
    for doc in data['queries']:
        for strategy, variations in doc['prompt_results'].items():
            for query in variations:
                queries.append({
                    'query': query, 'expected_id': doc['expected_id'],
                    'original': doc['original_question'], 'course': doc['course'],
                    'strategy': strategy,
                })
    return queries

def evaluate(name, search_fn, test_queries, k=10, use_course_filter=True):
    results, latencies = [], []
    for tq in test_queries:
        course = tq['course'] if use_course_filter else None
        t0 = time.time()
        hits = search_fn(tq['query'], k, course)
        elapsed = (time.time() - t0) * 1000
        latencies.append(elapsed)
        
        hit_ids = [h.get('id', h.get('es_id', '')) for h in hits]
        hit_courses = [h.get('course', '') for h in hits]
        
        rank = next((pos for pos, hid in enumerate(hit_ids, 1) if hid == tq['expected_id']), None)
        wrong_course = sum(1 for c in hit_courses[:5] if c != tq['course'] and c)
        
        results.append({
            'query': tq['query'], 'expected_id': tq['expected_id'],
            'strategy': tq['strategy'], 'course': tq['course'],
            'rank': rank, 'found': rank is not None,
            'top3_ids': hit_ids[:3], 'cross_course': wrong_course,
            'latency_ms': round(elapsed, 1),
        })
    return results, latencies

def compute_metrics(results, latencies, total):
    found = sum(1 for r in results if r['found'])
    recall_at = {}
    for k in K_VALUES:
        hits = sum(1 for r in results if r['found'] and r['rank'] and r['rank'] <= k)
        recall_at[k] = hits / total if total else 0
    mrr = sum(1.0 / r['rank'] for r in results if r['found'] and r['rank']) / total if total else 0
    cross = sum(r['cross_course'] for r in results) / (total * 5) if total else 0
    return {
        'total': total, 'found': found, 'hit_rate': found/total if total else 0,
        'recall_at_k': recall_at, 'mrr': round(mrr, 4), 'cross_course_rate': round(cross, 3),
        'p50_latency': round(float(np.percentile(latencies, 50)), 1),
        'p95_latency': round(float(np.percentile(latencies, 95)), 1),
    }

def slice_by(results, latencies, key):
    groups = defaultdict(list)
    groups_lat = defaultdict(list)
    for r, l in zip(results, latencies):
        groups[r[key]].append(r)
        groups_lat[r[key]].append(l)
    return {k: compute_metrics(groups[k], groups_lat[k], len(groups[k])) for k in groups}

def failures(results):
    return [
        {'query': r['query'][:80], 'strategy': r['strategy'], 'course': r['course'],
         'expected_id': r['expected_id'], 'returned_ids': r['top3_ids']}
        for r in results if not r['found']
    ]

# ── Search functions ─────────────────────────────────────────────────────────
def make_bm25(es, boost_q, boost_t):
    def search(query, size, course=None):
        body = {'size': size, 'query': {'multi_match': {'query': query,
            'fields': [f'question^{boost_q}', f'answer^{boost_t}'], 'type': 'best_fields'}}}
        if course:
            body['query'] = {'bool': {'must': body['query'], 'filter': {'term': {'course': course}}}}
        result = es.search(index=ES_INDEX, body=body)
        return [h['_source'] for h in result['hits']['hits']]
    return search

def make_vector(es, model):
    def search(query, size, course=None):
        vec = model.encode(query).tolist()
        body = {'size': size, 'query': {'script_score': {'query': {'match_all': {}},
            'script': {'source': "cosineSimilarity(params.query_vector, 'question_vector') + 1.0",
                       'params': {'query_vector': vec}}}}}
        if course:
            body['query']['script_score']['query'] = {'bool': {'filter': {'term': {'course': course}}}}
        result = es.search(index=ES_INDEX, body=body)
        return [h['_source'] for h in result['hits']['hits']]
    return search

def make_qdrant(client, model):
    def search(query, size, course=None):
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        vec = model.encode(query).tolist()
        qfilter = Filter(must=[FieldCondition(key='course', match=MatchValue(value=course))]) if course else None
        results = client.query_points(collection_name=QDRANT_COLLECTION, query=vec, limit=size, query_filter=qfilter, with_payload=True)
        return [h.payload for h in results.points]
    return search

def make_hybrid(es, model, method, bm25_w=0.5, vec_w=0.5, rrf_k=60):
    """Hybrid search combining BM25 and vector results."""
    bm25_fn = make_bm25(es, 20, 1)
    vec_fn = make_vector(es, model)
    
    def search(query, size, course=None):
        bm25_hits = bm25_fn(query, size * 2, course)
        vec_hits = vec_fn(query, size * 2, course)
        
        if not bm25_hits and not vec_hits:
            return []
        
        scores = {}
        if method == "rrf":
            for rank, hit in enumerate(bm25_hits):
                hid = hit.get("id", "")
                if hid:
                    scores[hid] = scores.get(hid, 0) + 1.0 / (rrf_k + rank + 1)
            for rank, hit in enumerate(vec_hits):
                hid = hit.get("id", "")
                if hid:
                    scores[hid] = scores.get(hid, 0) + 1.0 / (rrf_k + rank + 1)
        else:
            for rank, hit in enumerate(bm25_hits):
                hid = hit.get("id", "")
                if hid:
                    scores[hid] = scores.get(hid, 0) + bm25_w * (1.0 / (rank + 1))
            for rank, hit in enumerate(vec_hits):
                hid = hit.get("id", "")
                if hid:
                    scores[hid] = scores.get(hid, 0) + vec_w * (1.0 / (rank + 1))
        
        sorted_ids = sorted(scores, key=scores.get, reverse=True)[:size]
        all_hits = {h["id"]: h for h in bm25_hits + vec_hits if h.get("id")}
        return [all_hits[id] for id in sorted_ids if id in all_hits]
    return search

def make_rerank(base_search, cross_encoder, model, rerank_k=20):
    """Re-rank candidates using a cross-encoder."""
    def search(query, size, course=None):
        candidates = base_search(query, rerank_k, course)
        if len(candidates) <= size:
            return candidates[:size]
        pairs = [[query, c.get("answer", c.get("text", ""))[:500]] for c in candidates]
        scores = cross_encoder.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [c for c, _ in ranked[:size]]
    return search

def main():
    test_queries = load_test_queries()
    if not test_queries:
        print("No test queries. Run: uv run python eval/generation/generate_test_queries.py")
        return
    
    total = len(test_queries)
    print(f"Test queries: {total}")
    print(f"Strategies: {set(q['strategy'] for q in test_queries)}")
    print(f"Courses: {set(q['course'] for q in test_queries)}\n")
    
    model = SentenceTransformer(MODEL_NAME)
    es = Elasticsearch('http://localhost:9200')
    client = QdrantClient('localhost', port=6333)
    
    # Try loading cross-encoder for reranking
    has_reranker = False
    cross_encoder = None
    try:
        from sentence_transformers import CrossEncoder
        print(f"  Loading {RERANK_MODEL}...")
        cross_encoder = CrossEncoder(RERANK_MODEL)
        has_reranker = True
        print(f"  {RERANK_MODEL} loaded")
    except Exception as e:
        print(f"  Cross-encoder not available: {e}")
    
    configs = [
        ('bm25_default', make_bm25(es, 20, 1)),
        ('bm25_balanced', make_bm25(es, 5, 5)),
        ('bm25_high_text', make_bm25(es, 1, 10)),
        ('vector_cosine', make_vector(es, model)),
        ('qdrant_cosine', make_qdrant(client, model)),
        ('hybrid_rrf', make_hybrid(es, model, 'rrf')),
        ('hybrid_50_50', make_hybrid(es, model, 'linear', 0.5, 0.5)),
        ('hybrid_70_30_vec', make_hybrid(es, model, 'linear', 0.3, 0.7)),
        ('hybrid_30_70_vec', make_hybrid(es, model, 'linear', 0.7, 0.3)),
    ]
    
    # Add cross-encoder reranker if available
    if has_reranker:
        configs.append(('vector_reranked', make_rerank(make_vector(es, model), cross_encoder, model)))
        configs.append(('bm25_reranked', make_rerank(make_bm25(es, 20, 1), cross_encoder, model)))
    
    all_metrics = []
    for name, fn in configs:
        print(f"Testing {name}...")
        results, lats = evaluate(name, fn, test_queries)
        metrics = compute_metrics(results, lats, total)
        metrics['name'] = name
        metrics['per_strategy'] = slice_by(results, lats, 'strategy')
        metrics['per_course'] = slice_by(results, lats, 'course')
        metrics['failures'] = failures(results)[:10]
        all_metrics.append(metrics)
        print(f"  hit={metrics['hit_rate']:.2%} recall@5={metrics['recall_at_k'][5]:.2%} MRR={metrics['mrr']}")
    
    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"{'Retriever':<25} {'Hit':>8} {'R@5':>8} {'MRR':>8} {'P95ms':>8}")
    print(f"{'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for m in all_metrics:
        print(f"{m['name']:<25} {m['hit_rate']:>7.2%} {m['recall_at_k'][5]:>7.2%} {m['mrr']:>8.4f} {m['p95_latency']:>7.1f}")
    
    # ── Per-strategy ──────────────────────────────────────────────────────────
    if all_metrics:
        best = max(all_metrics, key=lambda m: m['recall_at_k'][5])
        strategies = list(best['per_strategy'].keys())
        print(f"\n{'='*70}")
        print(f"PER-STRATEGY BREAKDOWN ({best['name']})")
        print(f"{'Strategy':<25} {'Hit':>8} {'R@5':>8} {'MRR':>8}")
        print(f"{'-'*25} {'-'*8} {'-'*8} {'-'*8}")
        for s in strategies:
            m = best['per_strategy'][s]
            print(f"{s:<25} {m['hit_rate']:>7.2%} {m['recall_at_k'][5]:>7.2%} {m['mrr']:>8.4f}")
    
    # ── Save ──────────────────────────────────────────────────────────────────
    output = {'metadata': {'total_queries': total, 'timestamp': datetime.now().isoformat()}, 'configs': all_metrics}
    path = f'experiments/results/variations_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    os.makedirs('experiments/results', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {path}")

if __name__ == '__main__':
    main()
