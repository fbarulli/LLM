# /home/admin/LLM/LLM/01/web/cleanup_final.py

import os
import shutil

web_root = "/home/admin/LLM/LLM/01/web"
os.chdir(web_root)

# Create clean directories
for d in ["notebooks", "scripts"]:
    os.makedirs(d, exist_ok=True)

# Move notebooks to notebooks/
notebooks_to_move = ["experiments/results/eval_dashboard.ipynb"]
for nb in notebooks_to_move:
    if os.path.exists(nb):
        dst = os.path.join("notebooks", os.path.basename(nb))
        shutil.move(nb, dst)
        print(f"Moved: {nb} -> {dst}")

# Move scripts to scripts/
scripts_to_move = ["fix_imports.py", "run_all_experiments.sh", "run_eval.py", "quick_test.py", "prompt_manager.py", "comparisons.py", "app.py"]
for script in scripts_to_move:
    src = os.path.join(web_root, script)
    if os.path.exists(src):
        dst = os.path.join("scripts", script)
        shutil.move(src, dst)
        print(f"Moved: {src} -> {dst}")

# Clean __pycache__ from root
cache_dir = os.path.join(web_root, "__pycache__")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print(f"Removed: {cache_dir}")

print("\n✅ Cleanup complete")