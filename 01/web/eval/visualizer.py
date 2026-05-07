# /home/admin/LLM/LLM/01/web/eval/visualizer.py

import os
import json
import glob
import logging
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger("visualizer")


class RAGVisualizer:
    def __init__(self, results_dir: str = None):
        self.web_root = '/home/admin/LLM/LLM/01/web'
        if results_dir is None:
            self.results_dir = os.path.join(self.web_root, "experiments", "results")
        else:
            self.results_dir = results_dir

    def get_experiment_registry(self) -> pd.DataFrame:
        search_pattern = os.path.join(self.results_dir, "*.json")
        files = glob.glob(search_pattern)
        
        if not files:
            print(f"WARNING: No JSON files found in: {self.results_dir}")
            return pd.DataFrame(columns=["filename", "experiment_name", "created_at", "path"])

        registry = []
        for f in files:
            stats = os.stat(f)
            try:
                with open(f, 'r') as j:
                    data = json.load(j)
                    meta = data.get('metadata', {})
                
                registry.append({
                    "filename": os.path.basename(f),
                    "experiment_name": meta.get('name', 'unknown'),
                    "created_at": pd.to_datetime(stats.st_mtime, unit='s'),
                    "path": f
                })
            except Exception as e:
                logger.warning(f"Could not read {f}: {e}")
            
        df = pd.DataFrame(registry)
        return df.sort_values("created_at", ascending=False)

    def load_selected_experiments(self, filenames: List[str]) -> pd.DataFrame:
        all_dfs = []
        for fname in filenames:
            path = os.path.join(self.results_dir, fname)
            try:
                with open(path, 'r') as f:
                    payload = json.load(f)
                    temp_df = pd.DataFrame(payload['results'])
                    temp_df['run_label'] = payload['metadata']['name']
                    all_dfs.append(temp_df)
            except Exception as e:
                logger.error(f"Failed to load {path}: {e}")
                logger.error(traceback.format_exc())
        
        if all_dfs:
            return pd.concat(all_dfs, ignore_index=True)
        return pd.DataFrame()

    def compute_recall_at_k(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby(['run_label', 'k'])['success'].mean().reset_index()

    def compute_mrr(self, df: pd.DataFrame) -> pd.DataFrame:
        def mrr_for_group(group):
            max_k = group['k'].max()
            reciprocal_ranks = []
            for _, row in group.iterrows():
                if row['success']:
                    reciprocal_ranks.append(1.0 / row['k'])
                else:
                    reciprocal_ranks.append(0.0)
            return pd.Series({
                'mrr': max(reciprocal_ranks) if reciprocal_ranks else 0.0
            })
        
        return df.groupby(['run_label']).apply(mrr_for_group).reset_index()

    def compute_precision_at_k(self, df: pd.DataFrame) -> pd.DataFrame:
        df['precision'] = df['success'].astype(float)
        return df.groupby(['run_label', 'k'])['precision'].mean().reset_index()

    def compute_ndcg(self, df: pd.DataFrame, max_k: int = 10) -> pd.DataFrame:
        def dcg(relevances, k):
            relevances = relevances[:k]
            return sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevances))
        
        def idcg(k):
            return sum(1.0 / np.log2(idx + 2) for idx in range(k))
        
        results = []
        for (run_label, query), group in df.groupby(['run_label', 'query']):
            sorted_group = group.sort_values('k')
            relevances = sorted_group['success'].astype(float).tolist()
            for k in range(1, max_k + 1):
                if k <= len(relevances):
                    dcg_k = dcg(relevances, k)
                    idcg_k = idcg(k)
                    ndcg = dcg_k / idcg_k if idcg_k > 0 else 0.0
                else:
                    ndcg = 0.0
                results.append({
                    'run_label': run_label,
                    'query': query,
                    'k': k,
                    'ndcg': ndcg
                })
        
        result_df = pd.DataFrame(results)
        return result_df.groupby(['run_label', 'k'])['ndcg'].mean().reset_index()

    def compute_latency_percentiles(self, df: pd.DataFrame) -> pd.DataFrame:
        percentiles = df.groupby(['run_label', 'k'])['latency_ms'].agg([
            ('p50', lambda x: np.percentile(x, 50)),
            ('p95', lambda x: np.percentile(x, 95)),
            ('p99', lambda x: np.percentile(x, 99))
        ]).reset_index()
        return percentiles

    def plot_leaderboard(self, df: pd.DataFrame):
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='k', y='success', hue='run_label', marker='o')
        plt.title("Search Leaderboard: Recall@K")
        plt.ylabel("Recall")
        plt.xlabel("K")
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    def plot_mrr_comparison(self, mrr_df: pd.DataFrame):
        plt.figure(figsize=(10, 6))
        sns.barplot(data=mrr_df, x='run_label', y='mrr')
        plt.title("MRR (Mean Reciprocal Rank) by Experiment")
        plt.ylabel("MRR")
        plt.xlabel("Experiment")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def plot_precision_at_k(self, precision_df: pd.DataFrame):
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=precision_df, x='k', y='precision', hue='run_label', marker='o')
        plt.title("Precision@K by Experiment")
        plt.ylabel("Precision")
        plt.xlabel("K")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_ndcg_comparison(self, ndcg_df: pd.DataFrame):
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=ndcg_df, x='k', y='ndcg', hue='run_label', marker='o')
        plt.title("NDCG@K by Experiment")
        plt.ylabel("NDCG")
        plt.xlabel("K")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_latency_percentiles(self, latency_df: pd.DataFrame):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        pivot_p50 = latency_df.pivot(index='k', columns='run_label', values='p50')
        pivot_p50.plot(kind='bar', ax=axes[0])
        axes[0].set_title("P50 Latency by Experiment")
        axes[0].set_ylabel("Latency (ms)")
        axes[0].set_xlabel("K")
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        pivot_p95 = latency_df.pivot(index='k', columns='run_label', values='p95')
        pivot_p95.plot(kind='bar', ax=axes[1])
        axes[1].set_title("P95 Latency by Experiment")
        axes[1].set_ylabel("Latency (ms)")
        axes[1].set_xlabel("K")
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()

    def get_summary_table(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby(['run_label', 'k'])['success'].mean().unstack()


def main():
    visualizer = RAGVisualizer()
    registry = visualizer.get_experiment_registry()
    
    if registry.empty:
        print("No experiment results found in experiments/results/")
        return
    
    print("Available experiments:")
    print(registry[['experiment_name', 'filename']].to_string())
    
    filenames = registry['filename'].tolist()
    df = visualizer.load_selected_experiments(filenames)
    
    if df.empty:
        print("No data loaded")
        return
    
    visualizer.plot_leaderboard(df)
    
    mrr_df = visualizer.compute_mrr(df)
    visualizer.plot_mrr_comparison(mrr_df)
    
    summary = visualizer.get_summary_table(df)
    print("\n=== Summary Table (Recall@K) ===")
    print(summary.round(4))


if __name__ == "__main__":
    main()
