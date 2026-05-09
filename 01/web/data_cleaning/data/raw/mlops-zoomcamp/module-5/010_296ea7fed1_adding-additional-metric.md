---
id: 296ea7fed1
question: Adding additional metric
sort_order: 10
---

### Problem Description

Getting “target columns” “prediction columns” not present errors after adding a metric.

### Solution Description

Make sure to read through the documentation on what is required or optional when adding the metric. For example, `DatasetCorrelationsMetric` doesn’t require any parameters because the metric evaluates for correlations among the features.