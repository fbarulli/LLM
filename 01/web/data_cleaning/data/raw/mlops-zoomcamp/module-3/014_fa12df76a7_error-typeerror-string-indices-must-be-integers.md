---
id: fa12df76a7
question: 'Error: TypeError: string indices must be integers'
sort_order: 14
---

If you've removed and re-added blocks, especially due to issues with Global Data Products, try the following steps:

- Remove the connections from the `hyperparameter_tuning/sklearn` block in the Tree panel to its upstream blocks.
- Re-add these connections.
- Remember to save the pipeline using `Ctrl+S`.

### Video 3.2.8 Error:
#### Issue: ValueError: not enough values to unpack (expected 3, got 1)

Ensure your code follows this order:

- **data** → [training_set](http://localhost:6789/pipelines/xgboost_training/edit?sideview=tree#)
- **data_2** → [hyperparameter_tuning/xgboost](http://localhost:6789/pipelines/xgboost_training/edit?sideview=tree#)

If not, proceed with:

1. Remove the connections for the xgboost.
2. Reconnect starting with the training set, followed by `hyperparameter_tuning/xgboost`.
