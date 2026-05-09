---
id: a173b94288
question: How to use a specific python version (e.g. 3.11) from conda with pipenv?
sort_order: 56
---

First, avoid being in a virtual environment when using pipenv. You can point pipenv directly to the Python 3.11 interpreter from your Conda installation:

1. **Activate the Conda environment:**  
   ```bash
   conda activate env_name
   ```

2. **Get the Python path:**  
   ```bash
   which python
   ```

3. **Deactivate the Conda environment:**  
   ```bash
   conda deactivate
   ```

4. **Use pipenv with the Python path found in step 2:**  
   ```bash
   pipenv --python /path/to/python
   ```