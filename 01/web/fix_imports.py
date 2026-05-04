# /home/admin/LLM/LLM/01/web/fix_paths.py

import os
import shutil

def main():
    print("=" * 60)
    print("FIXING PATHS FOR STATS AND VISUALIZER")
    print("=" * 60)
    
    web_root = "/home/admin/LLM/LLM/01/web"
    os.chdir(web_root)
    
    # 1. Fix src/stats.py
    stats_path = os.path.join(web_root, "src", "stats.py")
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            content = f.read()
        
        # Replace the results_dir path
        old_path_line = 'current_file_dir = os.path.dirname(os.path.abspath(__file__))'
        new_path_line = 'current_file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up to web root'
        
        if old_path_line in content:
            content = content.replace(old_path_line, new_path_line)
            with open(stats_path, 'w') as f:
                f.write(content)
            print("✅ Fixed src/stats.py")
        else:
            print("⚠️ Could not find path line in src/stats.py - check manually")
    else:
        print(f"❌ src/stats.py not found at {stats_path}")
    
    # 2. Fix src/visualizer.py __init__ method
    visualizer_path = os.path.join(web_root, "src", "visualizer.py")
    if os.path.exists(visualizer_path):
        with open(visualizer_path, 'r') as f:
            content = f.read()
        
        # Check if already fixed
        if 'self.web_root' in content and 'os.path.dirname(os.path.dirname' in content:
            print("✅ src/visualizer.py already has correct path logic")
        else:
            # Replace the __init__ method
            old_init = '''    def __init__(self, results_dir: str = None):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.results_dir = results_dir if results_dir else self.script_dir'''
            
            new_init = '''    def __init__(self, results_dir: str = None):
        self.web_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.results_dir = results_dir if results_dir else os.path.join(self.web_root, "experiments", "results")'''
            
            if old_init in content:
                content = content.replace(old_init, new_init)
                with open(visualizer_path, 'w') as f:
                    f.write(content)
                print("✅ Fixed src/visualizer.py")
            else:
                print("⚠️ Could not find old __init__ in visualizer.py - check manually")
    else:
        print(f"❌ src/visualizer.py not found at {visualizer_path}")
    
    # 3. Delete wrong results directory
    wrong_results = os.path.join(web_root, "src", "experiments")
    if os.path.exists(wrong_results):
        shutil.rmtree(wrong_results)
        print(f"✅ Deleted wrong results directory: {wrong_results}")
    else:
        print("⚠️ No wrong results directory found at src/experiments")
    
    # 4. Ensure correct results directory exists
    correct_results = os.path.join(web_root, "experiments", "results")
    os.makedirs(correct_results, exist_ok=True)
    print(f"✅ Ensured correct results directory exists: {correct_results}")
    
    print("\n" + "=" * 60)
    print("FIX COMPLETE. Now run:")
    print("=" * 60)
    print("cd /home/admin/LLM/LLM/01/web")
    print("uv run pipeline.py --reindex")
    print("uv run pipeline.py")
    print()

if __name__ == "__main__":
    main()