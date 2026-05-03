import sys
import os
import json
import glob
from elasticsearch import Elasticsearch

# Responsibility: Orchestration & Automation
from config_manager import load_config
from ingest_data import transform_documents, setup_index_and_ingest
from stats import StatsCollector
from run_stats import get_eval_set

def run_experiment(config_file: str, n_samples: int = 20):
    """Handles a single experiment execution."""
    # Derive name from filename (e.g., 'global_bm25.json' -> 'global_bm25')
    experiment_name = os.path.basename(config_file).replace(".json", "")
    settings = load_config(config_file)
    
    # 1. Sync Step (Optional: We re-index only for the first run or if forced)
    # For automation, we'll skip re-indexing here to save time, 
    # unless you explicitly pass the --reindex flag to the script.
    
    # 2. Benchmark
    print(f"\n🧪 [RUNNING] Experiment: {experiment_name}")
    doc_path = os.path.join(os.getcwd(), "documents.json")
    eval_set = get_eval_set(doc_path, n_per_course=n_samples)
    
    collector = StatsCollector(config_file)
    result_file = collector.run_benchmark(eval_set, experiment_name)
    print(f"✅ [FINISHED] {experiment_name} -> {result_file}")

if __name__ == "__main__":
    # 1. Setup paths
    config_dir = "experiments/configs"
    config_files = glob.glob(os.path.join(config_dir, "*.json"))
    
    if not config_files:
        print(f"❌ No configs found in {config_dir}. Please create .json files there.")
        sys.exit(1)

    # 2. Handle Re-indexing for the whole batch if requested
    if "--reindex" in sys.argv:
        print("🔄 [GLOBAL] Re-indexing before starting batch...")
        
        # FIX: Load the first actual file path from the list
        first_config_path = config_files[0]
        base_settings = load_config(first_config_path)
        
        # Fallback for index_name to prevent KeyError
        idx_name = base_settings.get('index_name', base_settings.get('index', 'course-questions'))
        es_host = base_settings.get('es_host', 'http://localhost:9200')
        
        es_client = Elasticsearch(es_host)
        
        # Load local ground truth
        doc_path = os.path.join(os.getcwd(), "documents.json")
        with open(doc_path, "r") as f:
            raw_data = json.load(f)
            
        from ingest_data import transform_documents, setup_index_and_ingest
        flattened = transform_documents(raw_data)
        setup_index_and_ingest(es_client, idx_name, flattened)

    # 3. Auto-run all configs found
    print(f"🚀 Found {len(config_files)} experiments. Starting batch run...")
    for config in sorted(config_files):
        run_experiment(config, n_samples=20)

    print("\n🏁 All experiments complete. Refresh your Notebook to see the Leaderboard!")
