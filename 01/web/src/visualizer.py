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
        self.web_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.results_dir = results_dir if results_dir else os.path.join(self.web_root, "experiments", "results")

    def get_experiment_registry(self) -> pd.DataFrame:
        search_pattern = os.path.join(self.results_dir, "*.json")
        files = glob.glob(search_pattern)
        
        if not files:
            print(f"⚠️ [WARNING] No JSON files found in: {self.results_dir}")
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
                print(f"Could not read {f}: {e}")
            
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
        return pd.concat(all_dfs, ignore_index=True)

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

    def compute_hit_rate(self, df: pd.DataFrame) -> pd.DataFrame:
        df['hit'] = df['success'].astype(float)
        return df.groupby(['run_label', 'k'])['hit'].mean().reset_index()

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

    def compute_average_precision(self, df: pd.DataFrame) -> pd.DataFrame:
        def ap_for_group(group):
            sorted_group = group.sort_values('k')
            successes = sorted_group['success'].tolist()
            if not any(successes):
                return 0.0
            precision_at_k = []
            relevant_count = 0
            for i, success in enumerate(successes, 1):
                if success:
                    relevant_count += 1
                    precision_at_k.append(relevant_count / i)
            return sum(precision_at_k) / relevant_count if precision_at_k else 0.0
        
        return df.groupby(['run_label', 'query']).apply(
            lambda g: pd.Series({'ap': ap_for_group(g)})
        ).reset_index().groupby(['run_label'])['ap'].mean().reset_index()

    def compute_unique_courses_in_top_k(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.groupby(['run_label', 'k']).apply(
            lambda g: g['found_course'].nunique()
        ).reset_index(name='unique_courses')
        return result

    def compute_course_entropy(self, df: pd.DataFrame) -> pd.DataFrame:
        def entropy(course_list):
            counts = Counter(course_list)
            probs = [c / len(course_list) for c in counts.values()]
            return -sum(p * np.log2(p) for p in probs if p > 0)
        
        result = df.groupby(['run_label', 'k']).apply(
            lambda g: entropy(g['found_course'].tolist())
        ).reset_index(name='entropy')
        return result

    def compute_result_overlap(self, df: pd.DataFrame, method_a: str, method_b: str, k: int = 5) -> float:
        df_k = df[df['k'] == k]
        a_queries = df_k[df_k['run_label'] == method_a].set_index('query')['found_id']
        b_queries = df_k[df_k['run_label'] == method_b].set_index('query')['found_id']
        
        common_queries = set(a_queries.index) & set(b_queries.index)
        if not common_queries:
            return 0.0
        
        overlaps = sum(a_queries[q] == b_queries[q] for q in common_queries)
        return overlaps / len(common_queries)

    def compute_rank_correlation(self, df: pd.DataFrame) -> pd.DataFrame:
        def spearman_for_k(group, k1, k2):
            k1_data = group[group['k'] == k1].set_index('query')['found_id']
            k2_data = group[group['k'] == k2].set_index('query')['found_id']
            common_queries = set(k1_data.index) & set(k2_data.index)
            
            if not common_queries:
                return np.nan
            
            ranks_k1 = [list(k1_data.index).index(q) for q in common_queries]
            ranks_k2 = [list(k2_data.index).index(q) for q in common_queries]
            
            if len(ranks_k1) < 2:
                return np.nan
            
            return pd.Series({
                'spearman': np.corrcoef(ranks_k1, ranks_k2)[0, 1]
            })
        
        results = []
        for run_label, group in df.groupby(['run_label']):
            if 5 in group['k'].values and 10 in group['k'].values:
                corr = spearman_for_k(group, 5, 10)['spearman']
                results.append({'run_label': run_label, 'rank_correlation_5_10': corr})
        
        return pd.DataFrame(results)

    def compute_score_variance(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby(['run_label', 'k'])['score'].var().reset_index(name='score_variance')

    def compute_id_consistency(self, df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
        df_k = df[df['k'] == k]
        consistency = df_k.groupby(['run_label', 'query'])['found_id'].nunique()
        consistency = consistency.groupby('run_label').mean().reset_index(name='avg_id_consistency')
        return consistency

    def compute_latency_percentiles(self, df: pd.DataFrame) -> pd.DataFrame:
        percentiles = df.groupby(['run_label', 'k'])['latency_ms'].agg([
            ('p50', lambda x: np.percentile(x, 50)),
            ('p95', lambda x: np.percentile(x, 95)),
            ('p99', lambda x: np.percentile(x, 99))
        ]).reset_index()
        return percentiles

    def compute_success_by_query_length(self, df: pd.DataFrame) -> pd.DataFrame:
        df['query_length_bin'] = pd.cut(df['query_len'], bins=[0, 20, 50, 100, 500], labels=['short', 'medium', 'long', 'very_long'])
        return df.groupby(['run_label', 'k', 'query_length_bin'])['success'].mean().reset_index()

    def compute_success_by_query_prefix(self, df: pd.DataFrame) -> pd.DataFrame:
        def extract_prefix(query):
            query_lower = query.lower()
            if query_lower.startswith('course'):
                return 'course'
            elif query_lower.startswith('certificate'):
                return 'certificate'
            elif query_lower.startswith('office hours'):
                return 'office_hours'
            elif query_lower.startswith('homework'):
                return 'homework'
            else:
                return 'other'
        
        df['query_prefix'] = df['query'].apply(extract_prefix)
        return df.groupby(['run_label', 'k', 'query_prefix'])['success'].mean().reset_index()

    def _get_dynamic_ylim(self, data: pd.Series, padding_factor: float = 0.2, min_padding: float = 0.05) -> tuple:
        if data.dtype == 'bool':
            data = data.astype(float)
        
        min_val = data.min()
        max_val = data.max()
        
        if min_val == max_val:
            padding = min_padding
        else:
            padding = max((max_val - min_val) * padding_factor, min_padding)
        
        return (min_val - padding, max_val + padding)

    def plot_leaderboard(self, df: pd.DataFrame):
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='k', y='success', hue='run_label', marker='o')
        plt.title("Search Leaderboard: Recall@K")
        y_min, y_max = self._get_dynamic_ylim(df['success'])
        plt.ylim(y_min, y_max)
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
        y_min, y_max = self._get_dynamic_ylim(mrr_df['mrr'])
        plt.ylim(y_min, y_max)
        plt.ylabel("MRR")
        plt.xlabel("Experiment")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def plot_precision_at_k(self, precision_df: pd.DataFrame):
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=precision_df, x='k', y='precision', hue='run_label', marker='o')
        plt.title("Precision@K by Experiment")
        y_min, y_max = self._get_dynamic_ylim(precision_df['precision'])
        plt.ylim(y_min, y_max)
        plt.ylabel("Precision")
        plt.xlabel("K")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_ndcg_comparison(self, ndcg_df: pd.DataFrame):
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=ndcg_df, x='k', y='ndcg', hue='run_label', marker='o')
        plt.title("NDCG@K by Experiment")
        y_min, y_max = self._get_dynamic_ylim(ndcg_df['ndcg'])
        plt.ylim(y_min, y_max)
        plt.ylabel("NDCG")
        plt.xlabel("K")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_diversity_metrics(self, unique_df: pd.DataFrame, entropy_df: pd.DataFrame):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        sns.lineplot(data=unique_df, x='k', y='unique_courses', hue='run_label', marker='o', ax=axes[0])
        axes[0].set_title("Unique Courses in Top K Results")
        y_min = unique_df['unique_courses'].min() - 1
        y_max = unique_df['unique_courses'].max() + 1
        axes[0].set_ylim(y_min, y_max)
        axes[0].set_ylabel("Unique Courses")
        axes[0].set_xlabel("K")
        axes[0].grid(True, alpha=0.3)
        
        sns.lineplot(data=entropy_df, x='k', y='entropy', hue='run_label', marker='o', ax=axes[1])
        axes[1].set_title("Course Distribution Entropy")
        y_min, y_max = self._get_dynamic_ylim(entropy_df['entropy'])
        axes[1].set_ylim(y_min, y_max)
        axes[1].set_ylabel("Entropy (bits)")
        axes[1].set_xlabel("K")
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def plot_latency_percentiles(self, latency_df: pd.DataFrame):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        pivot_p50 = latency_df.pivot(index='k', columns='run_label', values='p50')
        pivot_p50.plot(kind='bar', ax=axes[0])
        axes[0].set_title("P50 Latency by Experiment")
        y_min, y_max = self._get_dynamic_ylim(latency_df['p50'])
        axes[0].set_ylim(y_min, y_max)
        axes[0].set_ylabel("Latency (ms)")
        axes[0].set_xlabel("K")
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        pivot_p95 = latency_df.pivot(index='k', columns='run_label', values='p95')
        pivot_p95.plot(kind='bar', ax=axes[1])
        axes[1].set_title("P95 Latency by Experiment")
        y_min, y_max = self._get_dynamic_ylim(latency_df['p95'])
        axes[1].set_ylim(y_min, y_max)
        axes[1].set_ylabel("Latency (ms)")
        axes[1].set_xlabel("K")
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()

    def plot_success_by_query_length(self, len_df: pd.DataFrame):
        for k in len_df['k'].unique():
            plt.figure(figsize=(12, 6))
            subset = len_df[len_df['k'] == k]
            sns.barplot(data=subset, x='query_length_bin', y='success', hue='run_label')
            plt.title(f"Success Rate by Query Length (K={k})")
            y_min, y_max = self._get_dynamic_ylim(subset['success'])
            plt.ylim(y_min, y_max)
            plt.ylabel("Recall")
            plt.xlabel("Query Length")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.show()

    def plot_success_by_query_prefix(self, prefix_df: pd.DataFrame, k: int = 5):
        subset = prefix_df[prefix_df['k'] == k]
        plt.figure(figsize=(12, 6))
        sns.barplot(data=subset, x='query_prefix', y='success', hue='run_label')
        plt.title(f"Success Rate by Query Prefix (K={k})")
        y_min, y_max = self._get_dynamic_ylim(subset['success'])
        plt.ylim(y_min, y_max)
        plt.ylabel("Recall")
        plt.xlabel("Query Type")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    def plot_answer_quality(self, df: pd.DataFrame):
        """Plot faithfulness vs relevancy by config."""
        quality_data = df[df['k'] == 5].copy()
        
        if 'faithful' not in quality_data.columns or 'relevant' not in quality_data.columns:
            print("\n⚠️ No quality metrics found. Run add_quality_metrics.py first.")
            print("   This adds 'faithful' and 'relevant' columns to your results.")
            return
        
        quality_data = quality_data.dropna(subset=['faithful', 'relevant'])
        
        if quality_data.empty:
            print("No quality metrics available.")
            return
        
        summary = quality_data.groupby('run_label').agg({
            'faithful': lambda x: (x == True).mean() if len(x) > 0 else 0,
            'relevant': lambda x: (x == True).mean() if len(x) > 0 else 0
        }).round(4) * 100
        
        summary.columns = ['Faithful %', 'Relevant %']
        
        print("\n" + "=" * 60)
        print("📊 ANSWER QUALITY BY CONFIG")
        print("=" * 60)
        print(summary.to_string())
        
        fig, ax = plt.subplots(figsize=(12, 6))
        summary.plot(kind='bar', ax=ax)
        ax.set_title('Faithfulness vs Relevancy by Config', fontsize=14)
        ax.set_ylabel('Percentage (%)')
        ax.set_xlabel('Experiment Configuration')
        ax.set_ylim(0, 105)
        ax.legend(['Faithful (answer grounded in context)', 'Relevant (answers the question)'])
        ax.grid(True, alpha=0.3)
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.0f%%', padding=3)
        
        plt.tight_layout()
        plt.show()
        
        return summary
    
    def plot_quality_gap(self, df: pd.DataFrame):
            """Plot the gap between recall and answer quality."""
            quality_data = df[df['k'] == 5].copy()
            
            if 'faithful' not in quality_data.columns or 'relevant' not in quality_data.columns:
                print("\n⚠️ No quality metrics found. Run evaluate_quality.py first.")
                print("   This adds 'faithful' and 'relevant' columns to your results.")
                return
            
            summary = quality_data.groupby('run_label').agg({
                'success': lambda x: (x == True).mean() if len(x) > 0 else 0,
                'faithful': lambda x: (x == True).mean() if len(x) > 0 else 0,
                'relevant': lambda x: (x == True).mean() if len(x) > 0 else 0
            }).round(4) * 100
            
            summary.columns = ['Recall %', 'Faithful %', 'Relevant %']
            summary['Gap (Recall - Relevant)'] = summary['Recall %'] - summary['Relevant %']
            
            print("\n" + "=" * 60)
            print("📊 RETRIEVAL VS ANSWER QUALITY GAP")
            print("=" * 60)
            print(summary.to_string())
            
            fig, ax = plt.subplots(figsize=(12, 6))
            summary[['Recall %', 'Relevant %']].plot(kind='bar', ax=ax)
            ax.set_title('Recall vs Answer Quality by Config', fontsize=14)
            ax.set_ylabel('Percentage (%)')
            ax.set_ylim(0, 105)
            ax.legend(['Recall (found correct doc)', 'Relevant (doc answers question)'])
            ax.grid(True, alpha=0.3)
            
            for container in ax.containers:
                ax.bar_label(container, fmt='%.0f%%', padding=3)
            
            plt.tight_layout()
            plt.show()
            
            return summary

    def get_quality_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return quality metrics summary table."""
        if 'faithful' not in df.columns or 'relevant' not in df.columns:
            return pd.DataFrame({'Error': ['No quality metrics found. Run add_quality_metrics.py first.']})
        
        quality_df = df[df['k'] == 5].copy()
        quality_df = quality_df.dropna(subset=['faithful', 'relevant'])
        
        if quality_df.empty:
            return pd.DataFrame({'Error': ['No quality metrics available.']})
        
        summary = quality_df.groupby('run_label').agg({
            'faithful': lambda x: (x == True).mean(),
            'relevant': lambda x: (x == True).mean(),
            'success': 'mean'
        }).round(4)
        
        summary.columns = ['Faithfulness', 'Relevancy', 'Recall@5']
        return summary.sort_values('Recall@5', ascending=False)

    def get_summary_table(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby(['run_label', 'k'])['success'].mean().unstack()

    def get_comprehensive_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        recall = self.compute_recall_at_k(df)
        recall = recall[recall['k'] == 5].set_index('run_label')['success'].rename('recall@5')
        
        mrr = self.compute_mrr(df).set_index('run_label')['mrr']
        
        precision = self.compute_precision_at_k(df)
        precision = precision[precision['k'] == 5].set_index('run_label')['precision'].rename('precision@5')
        
        hit = self.compute_hit_rate(df)
        hit = hit[hit['k'] == 5].set_index('run_label')['hit'].rename('hit_rate@5')
        
        unique = self.compute_unique_courses_in_top_k(df)
        unique = unique[unique['k'] == 5].set_index('run_label')['unique_courses'].rename('unique_courses@5')
        
        entropy = self.compute_course_entropy(df)
        entropy = entropy[entropy['k'] == 5].set_index('run_label')['entropy'].rename('entropy@5')
        
        latency = self.compute_latency_percentiles(df)
        latency_p95 = latency[latency['k'] == 5].set_index('run_label')['p95'].rename('p95_latency_ms')
        
        summary = pd.concat([recall, mrr, precision, hit, unique, entropy, latency_p95], axis=1)
        return summary.round(4)


def main():
    visualizer = RAGVisualizer()
    registry = visualizer.get_experiment_registry()
    
    if registry.empty:
        print("No experiment results found")
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
    
    precision_df = visualizer.compute_precision_at_k(df)
    visualizer.plot_precision_at_k(precision_df)
    
    ndcg_df = visualizer.compute_ndcg(df)
    visualizer.plot_ndcg_comparison(ndcg_df)
    
    unique_df = visualizer.compute_unique_courses_in_top_k(df)
    entropy_df = visualizer.compute_course_entropy(df)
    visualizer.plot_diversity_metrics(unique_df, entropy_df)
    
    latency_df = visualizer.compute_latency_percentiles(df)
    visualizer.plot_latency_percentiles(latency_df)
    
    len_df = visualizer.compute_success_by_query_length(df)
    visualizer.plot_success_by_query_length(len_df)
    
    prefix_df = visualizer.compute_success_by_query_prefix(df)
    visualizer.plot_success_by_query_prefix(prefix_df)
    
    if 'faithful' in df.columns:
        visualizer.plot_answer_quality(df)
        quality_summary = visualizer.get_quality_summary(df)
        print("\n=== Answer Quality Summary ===")
        print(quality_summary.to_string())
    
    summary = visualizer.get_comprehensive_summary(df)
    print("\n=== Comprehensive Summary (K=5) ===")
    print(summary.to_string())


if __name__ == "__main__":
    main()
