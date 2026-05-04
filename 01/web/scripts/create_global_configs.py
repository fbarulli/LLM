import json
import glob
import os
import shutil

config_dir = "experiments/configs"
global_configs_dir = "experiments/configs/global_variants"
os.makedirs(global_configs_dir, exist_ok=True)

# Copy existing configs that aren't already global
config_files = glob.glob(os.path.join(config_dir, "*.json"))

for filepath in config_files:
    with open(filepath, 'r') as f:
        config = json.load(f)
    
    basename = os.path.basename(filepath)
    
    # Skip if already global
    if "global" in basename.lower():
        continue
    
    # Create new name with global_ prefix
    new_name = f"global_{basename}"
    new_path = os.path.join(global_configs_dir, new_name)
    
    with open(new_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Created: {new_path}")

print(f"\nCreated {len(glob.glob(os.path.join(global_configs_dir, '*.json')))} global configs")