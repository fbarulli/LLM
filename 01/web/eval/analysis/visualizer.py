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
        """Recall@K = (number of queries where relevant doc in top-k) / total queries"""
        return df.groupby(['run_label', 'k'])['success'].mean().reset_index()

    def compute_mrr(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        MRR = mean reciprocal rank of the first correct answer.
        Requires that each row has a 'rank' field (1-indexed rank of the correct answer, 0 if not found).
        If 'rank' not present, we approximate using 'k' and 'success'? Not possible.
        We'll use the 'found_id' and 'expected_id' to compute rank (but we don't have rank in the DF).
        A workaround: assume that the correct answer is always at rank k if success? That's wrong.
        For now, we use the original method (which is flawed) but print a warning.
        To properly compute MRR, we need to store the rank of the correct answer per query.
        """
        # Check if we have a 'rank' column
        if 'rank' not in df.columns:
            logger.warning("MRR: 'rank' column missing. Computing dummy MRR (use only for relative comparison).")
            # Fallback: use the reciprocal of the smallest k where success=True, else 0
            def mrr_per_query(group):
                successes = group[group['success']]
                if successes.empty:
                    return 0.0
                best_k = successes['k'].min()
                return 1.0 / best_k
            mrr_df = df.groupby(['run_label', 'query']).apply(mrr_per_query).reset_index(name='mrr')
            return mrr_df.groupby('run_label')['mrr'].mean().reset_index()
        else:
            # Proper MRR using rank (1-indexed)
            df['reciprocal'] = df['rank'].apply(lambda x: 1.0 / x if x > 0 else 0.0)
            return df.groupby('run_label')['reciprocal'].mean().reset_index(name='mrr')

    def compute_precision_at_k(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Precision@K = (number of relevant docs in top-k) / k.
        Since only one relevant document exists, precision@k = 1/k if success, else 0.
        """
        df['precision'] = df.apply(lambda row: 1.0 / row['k'] if row['success'] else 0.0, axis=1)
        return df.groupby(['run_label', 'k'])['precision'].mean().reset_index()

    def compute_ndcg(self, df: pd.DataFrame, max_k: int = 10) -> pd.DataFrame:
        """NDCG@K for single relevant document."""
        def dcg(relevances, k):
            relevances = relevances[:k]
            return sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevances))
        
        def idcg(k):
            return 1.0 / np.log2(2)  # since only one relevant at position 1
        
        results = []
        for (run_label, query), group in df.groupby(['run_label', 'query']):
            sorted_group = group.sort_values('k')
            # We need rank information. Without rank, we approximate using success at each k
            # Better: use the actual rank if available.
            # For now, if we have 'rank' column, use it.
            if 'rank' in df.columns:
                rank_val = group['rank'].iloc[0]
                for k in range(1, max_k+1):
                    rel = 1 if rank_val <= k else 0
                    dcg_k = rel / np.log2(rank_val + 1) if rel else 0.0
                    idcg_k = 1.0 / np.log2(1+1)  # always 1
                    ndcg = dcg_k / idcg_k if idcg_k > 0 else 0.0
                    results.append({'run_label': run_label, 'query': query, 'k': k, 'ndcg': ndcg})
            else:
                # fallback: use success at each k
                for k in range(1, max_k+1):
                    # Check if any success for k' <= k
                    success = any(sorted_group[sorted_group['k'] <= k]['success'])
                    dcg_k = 1.0 / np.log2(1+1) if success else 0.0
                    idcg_k = 1.0 / np.log2(1+1)
                    ndcg = dcg_k / idcg_k if idcg_k > 0 else 0.0
                    results.append({'run_label': run_label, 'query': query, 'k': k, 'ndcg': ndcg})
        
        result_df = pd.DataFrame(results)
        return result_df.groupby(['run_label', 'k'])['ndcg'].mean().reset_index()

    def compute_latency_percentiles(self, df: pd.DataFrame) -> pd.DataFrame:
        # Note: latency is the same for all k because we searched once with top_k=10.
        # We group by run_label and k, but latency is identical; the percentiles will be the same.
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
        axes[0].set_title("P50 Latency by Experiment (search size = max_k)")
        axes[0].set_ylabel("Latency (ms)")
        axes[0].set_xlabel("K")
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        pivot_p95 = latency_df.pivot(index='k', columns='run_label', values='p95')
        pivot_p95.plot(kind='bar', ax=axes[1])
        axes[1].set_title("P95 Latency by Experiment (search size = max_k)")
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