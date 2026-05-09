---
id: cc4481d2fa
question: Video 3.2.1 - Various issues with Global Data Products
sort_order: 12
---

Refer to the following documentation for more details:

- [Project Management](https://docs.mage.ai/platform/projects/management)
- [Project Structure](https://docs.mage.ai/design/abstractions/project-structure)
- [Global Data Products Overview](https://docs.mage.ai/orchestration/global-data-products/overview)

## Issues and Solutions

### Running the GDP Block Takes Forever

**Exception:**

```
Pipeline run xx for global data product training_set: failed
AttributeError: 'NoneType' object has no attribute 'to_dict'
```

### Potential Causes and Solutions:

- **Ensure Project and Pipeline Matching**:
  
  Make sure the following configurations are correct:
  
  ```python
  "project": "unit_2_training",
  "repo_path": "/home/src/mlops/unit_2_training",
  ```

- **Restart Steps**:
  
  1. Interrupt and restart the Kernel from the Run menu.
  2. Bring Docker down and restart it via the script.

- **Recreate Everything (if above steps fail):**
  
  1. Remove connections from the `hyperparameter_tuning/sklearn` block in the Tree panel to its upstream blocks.
     - Click on the connector → Remove Connection.
  2. Remove the Global Data Product block from the Tree panel.
     - Right click → Delete Block (ignore dependencies).
  3. Click on All blocks, select Global Data Products, drag and drop this block to be the first in the pipeline.
  4. Rename the block to what is used in the video.
  5. Run the block to test it (Play button or Ctrl+Enter).

### Note

If helpful, repeat similar steps for the file in path "unit_3_observability." There is an ongoing attempt to replicate this process.

### Error with Creating Global Data Product on Mage

**Error:**

```
AttributeError: 'NoneType' object has no attribute 'to_dict'
```

**Solution:**

Global product is currently not cross-product. You will need to create the data preparation pipeline in `unit_2_training` and configure it to build.