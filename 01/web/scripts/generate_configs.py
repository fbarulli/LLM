import json
import os

def generate_experiment_configs():
    configs = [
        {
            "name": "baseline_bm25",
            "es_host": "http://localhost:9200",
            "boost_question": 20.0,
            "boost_text": 1.0,
            "search_type": "best_fields"
        },
        {
            "name": "high_question",
            "es_host": "http://localhost:9200",
            "boost_question": 50.0,
            "boost_text": 1.0,
            "search_type": "best_fields"
        },
        {
            "name": "high_text",
            "es_host": "http://localhost:9200",
            "boost_question": 1.0,
            "boost_text": 10.0,
            "search_type": "best_fields"
        },
        {
            "name": "balanced",
            "es_host": "http://localhost:9200",
            "boost_question": 5.0,
            "boost_text": 5.0,
            "search_type": "best_fields"
        },
        {
            "name": "cross_fields",
            "es_host": "http://localhost:9200",
            "boost_question": 20.0,
            "boost_text": 1.0,
            "search_type": "cross_fields"
        },
        {
            "name": "most_fields",
            "es_host": "http://localhost:9200",
            "boost_question": 20.0,
            "boost_text": 1.0,
            "search_type": "most_fields"
        }
    ]
    
    config_dir = "experiments/configs"
    os.makedirs(config_dir, exist_ok=True)
    
    names = []
    for config in configs:
        name = config["name"]
        names.append(name)
        config_copy = {k: v for k, v in config.items() if k != "name"}
        filepath = os.path.join(config_dir, f"{name}.json")
        with open(filepath, "w") as f:
            json.dump(config_copy, f, indent=4)
        print(f"Created: {filepath}")
    
    return names

if __name__ == "__main__":
    generate_experiment_configs()