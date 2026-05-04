import sys
import os
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')

from src.stats import StatsCollector
from src.run_hard_stats import get_hard_eval_set
import glob

def main():
    print("=" * 60)
    print("RUNNING BENCHMARK ON HARD EVAL SET")
    print("=" * 60)
    
    # Load hard eval set
    eval_set = get_hard_eval_set()
    print(f"\n📊 Loaded {len(eval_set)} hard test queries")
    
    # Run on baseline config
    config_path = "experiments/configs/baseline_bm25.json"
    experiment_name = "baseline_bm25_hard"
    
    print(f"\n🧪 Running: {experiment_name}")
    collector = StatsCollector(config_path)
    result_file = collector.run_benchmark(eval_set, experiment_name)
    print(f"✅ Results saved to {result_file}")
    
    # Also run global config
    config_path = "experiments/configs/global_bm25.json"
    experiment_name = "global_bm25_hard"
    
    print(f"\n🧪 Running: {experiment_name}")
    collector = StatsCollector(config_path)
    result_file = collector.run_benchmark(eval_set, experiment_name)
    print(f"✅ Results saved to {result_file}")

if __name__ == "__main__":
    main()
