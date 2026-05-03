import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict

class RAGVisualizer:
    def __init__(self, results_dir: str = None):
        # 1. Get the directory where THIS script is actually located
        # Since this script is in experiments/results, its directory IS the results dir.
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # If user provides a path, use it, otherwise use the script's own folder
        self.results_dir = results_dir if results_dir else self.script_dir

    def get_experiment_registry(self) -> pd.DataFrame:
        """Returns all JSON results ordered by time."""
        search_pattern = os.path.join(self.results_dir, "*.json")
        files = glob.glob(search_pattern)
        
        if not files:
            # This prevents the KeyError by returning a DF with columns but no rows
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
                    "created_at": pd.to_datetime(stats.st_mtime, unit='s'), # Use mtime (modified)
                    "path": f
                })
            except Exception as e:
                print(f"Could not read {f}: {e}")
            
        df = pd.DataFrame(registry)
        return df.sort_values("created_at", ascending=False)

    def load_selected_experiments(self, filenames: List[str]) -> pd.DataFrame:
        all_dfs = []
        for fname in filenames:
            # Search for the file in the results_dir
            path = os.path.join(self.results_dir, fname)
            with open(path, 'r') as f:
                payload = json.load(f)
                temp_df = pd.DataFrame(payload['results'])
                temp_df['run_label'] = payload['metadata']['name']
                all_dfs.append(temp_df)
        return pd.concat(all_dfs, ignore_index=True)


    def plot_leaderboard(self, df: pd.DataFrame):
        """Generates the Recall@K comparison."""
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='k', y='success', hue='run_label', marker='o')
        plt.title("🚀 Search Leaderboard: Recall@K")
        plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()

    def get_summary_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns a text-based summary for sharing results."""
        summary = df.groupby(['run_label', 'k'])['success'].mean().unstack()
        return summary
