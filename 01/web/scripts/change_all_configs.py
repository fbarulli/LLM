import json
import glob
import os

config_dir = "experiments/configs"
for filepath in glob.glob(os.path.join(config_dir, "*.json")):
    with open(filepath, 'r') as f:
        config = json.load(f)
    
    config["boost_question"] = 0.5
    config["boost_text"] = 1.0
    
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Updated: {filepath}")