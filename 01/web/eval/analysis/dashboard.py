import sys
import os
import json
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')

import numpy as np
import pandas as pd
from eval.visualizer import RAGVisualizer
from eval.run_full_benchmark import run_single_benchmark

def _load_configs():
    configs_path = '/home/admin/LLM/LLM/01/web/configs/search_configs.json'
    with open(configs_path, 'r') as f:
        return list(json.load(f).keys())

CONFIGS = _load_configs()
RESULTS_DIR = '/home/admin/LLM/LLM/01/web/experiments/results'
EXPECTED_QUERIES = 1154

def _check_and_run_missing():
    missing = []
    for config in CONFIGS:
        result_file = os.path.join(RESULTS_DIR, f'{config}.json')
        if not os.path.exists(result_file):
            missing.append(config)
        else:
            with open(result_file, 'r') as f:
                data = json.load(f)
                if data['metadata']['total_queries'] != EXPECTED_QUERIES:
                    missing.append(config)
    
    if missing:
        print(f"Running missing configs: {missing}")
        for config in missing:
            run_single_benchmark(config, batch_size=50)
        print("Complete.\n")

def _run_ab_test(config_a, config_b, k=5):
    a_file = os.path.join(RESULTS_DIR, f'{config_a}.json')
    b_file = os.path.join(RESULTS_DIR, f'{config_b}.json')
    
    if not os.path.exists(a_file) or not os.path.exists(b_file):
        return None
    
    with open(a_file, 'r') as f:
        a_data = json.load(f)
    with open(b_file, 'r') as f:
        b_data = json.load(f)
    
    a_results = {r['query']: r for r in a_data['results'] if r['k'] == k}
    b_results = {r['query']: r for r in b_data['results'] if r['k'] == k}
    
    common_queries = list(set(a_results.keys()) & set(b_results.keys()))
    
    a_wins = 0
    b_wins = 0
    ties = 0
    
    for query in common_queries:
        a_success = a_results[query]['success']
        b_success = b_results[query]['success']
        
        if a_success and not b_success:
            a_wins += 1
        elif b_success and not a_success:
            b_wins += 1
        else:
            ties += 1
    
    return {
        'config_a': config_a,
        'config_b': config_b,
        'a_wins': a_wins,
        'b_wins': b_wins,
        'ties': ties,
        'total': len(common_queries),
        'a_win_rate': a_wins / len(common_queries),
        'b_win_rate': b_wins / len(common_queries)
    }

def _generate_conclusions(df, latency_df, recall_pivot):
    conclusions = []
    
    best_recall_config = recall_pivot[5].idxmax()
    best_recall_value = recall_pivot[5].max()
    conclusions.append(f"Best recall@5: {best_recall_config} with {best_recall_value:.2%}")
    
    min_k_for_perfect = None
    for config in recall_pivot.index:
        for k in [1, 3, 5, 10]:
            if recall_pivot.loc[config, k] >= 0.999:
                if min_k_for_perfect is None or k < min_k_for_perfect:
                    min_k_for_perfect = k
                break
    if min_k_for_perfect:
        conclusions.append(f"Perfect recall achieved at K={min_k_for_perfect}")
    
    bm25_configs = [c for c in recall_pivot.index if 'bm25' in c.lower()]
    hybrid_configs = [c for c in recall_pivot.index if 'hybrid' in c.lower()]
    vector_configs = [c for c in recall_pivot.index if 'vector' in c.lower()]
    
    if bm25_configs and hybrid_configs:
        bm25_recall = recall_pivot.loc[bm25_configs[0], 5]
        hybrid_recall = recall_pivot.loc[hybrid_configs[0], 5]
        if abs(bm25_recall - hybrid_recall) < 0.01:
            conclusions.append("BM25 and Hybrid have equivalent recall (difference < 1%)")
    
    if bm25_configs and vector_configs:
        bm25_recall = recall_pivot.loc[bm25_configs[0], 5]
        vector_recall = recall_pivot.loc[vector_configs[0], 5]
        # Also compute recall@1 gap
        recall1_bm25 = recall_pivot.loc[bm25_configs[0], 1]
        recall1_vector = recall_pivot.loc[vector_configs[0], 1]
        conclusions.append(f"BM25 outperforms Vector by {(bm25_recall - vector_recall):.2%} recall@5")
        conclusions.append(f"At recall@1, BM25 leads by {(recall1_bm25 - recall1_vector):.2%}")
    
    latency_comparison = []
    for config in recall_pivot.index:
        config_latency = latency_df[latency_df['run_label'] == config]['p95'].values
        if len(config_latency) > 0:
            latency_comparison.append((config, config_latency[0]))
    
    if latency_comparison:
        fastest = min(latency_comparison, key=lambda x: x[1])
        slowest = max(latency_comparison, key=lambda x: x[1])
        conclusions.append(f"Fastest: {fastest[0]} ({fastest[1]:.1f}ms P95)")
        conclusions.append(f"Slowest: {slowest[0]} ({slowest[1]:.1f}ms P95)")
        if slowest[1] / fastest[1] > 3:
            conclusions.append(f"Speed difference: {slowest[1]/fastest[1]:.1f}x slower for {slowest[0]}")
    
    return conclusions

def show_dashboard():
    _check_and_run_missing()
    
    viz = RAGVisualizer()
    registry = viz.get_experiment_registry()
    filenames = registry['filename'].tolist()
    df = viz.load_selected_experiments(filenames)
    
    print("\n=== RECALL@K SUMMARY ===")
    recall_summary = viz.compute_recall_at_k(df)
    recall_pivot = recall_summary.pivot(index='run_label', columns='k', values='success')
    print(recall_pivot.round(4))
    
    print("\n=== MRR COMPARISON ===")
    mrr_df = viz.compute_mrr(df)
    print(mrr_df.round(4))
    
    print("\n=== PRECISION@K ===")
    precision_df = viz.compute_precision_at_k(df)
    precision_pivot = precision_df.pivot(index='run_label', columns='k', values='precision')
    print(precision_pivot.round(4))
    
    print("\n=== LATENCY P95 (ms) ===")
    latency_df = viz.compute_latency_percentiles(df)
    latency_pivot = latency_df.pivot(index='run_label', columns='k', values='p95')
    print(latency_pivot.round(2))
    
    print("\n=== A/B TEST: BM25 Default vs Vector Default (win rates) ===")
    ab_result = _run_ab_test('bm25_default', 'vector_default')
    if ab_result:
        print(f"BM25 Default wins: {ab_result['a_wins']}")
        print(f"Vector Default wins: {ab_result['b_wins']}")
        print(f"Ties: {ab_result['ties']}")
        print(f"BM25 Win Rate: {ab_result['a_win_rate']:.2%}")
        print(f"Vector Win Rate: {ab_result['b_win_rate']:.2%}")
    
    print("\n=== A/B TEST: Hybrid Default vs Vector Default (win rates) ===")
    ab_result = _run_ab_test('hybrid_default', 'vector_default')
    if ab_result:
        print(f"Hybrid Default wins: {ab_result['a_wins']}")
        print(f"Vector Default wins: {ab_result['b_wins']}")
        print(f"Ties: {ab_result['ties']}")
    
    print("\n=== CONCLUSIONS ===")
    conclusions = _generate_conclusions(df, latency_df, recall_pivot)
    for i, conclusion in enumerate(conclusions, 1):
        print(f"{i}. {conclusion}")
    
    viz.plot_leaderboard(df)
    viz.plot_mrr_comparison(mrr_df)
    viz.plot_precision_at_k(precision_df)
    viz.plot_latency_percentiles(latency_df)

if __name__ == "__main__":
    show_dashboard()