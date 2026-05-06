import sys
import os
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')

from src.stats import StatsCollector
from src.run_hard_stats import get_hard_eval_set
import glob

def main():
    print("=" * 60)
    print("RUNNING FULL BENCHMARK ON HARD EVAL SET")
    print("=" * 60)
    
    # Load hard eval set (90 paraphrased queries)
    eval_set = get_hard_eval_set()
    print(f"\n📊 Loaded {len(eval_set)} hard test queries")
    
    # Run on all configs
    configs = [
        "baseline_bm25",
        "global_bm25", 
        "vector_bm25",
        "hybrid_bm25"
    ]
    
    for config_name in configs:
        config_path = f"experiments/configs/{config_name}.json"
        experiment_name = f"{config_name}_hard_full"
        
        print(f"\n🧪 Running: {experiment_name}")
        collector = StatsCollector(config_path)
        result_file = collector.run_benchmark(eval_set, experiment_name)
        print(f"✅ Results saved to {result_file}")

if __name__ == "__main__":
    main()
