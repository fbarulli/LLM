---
id: 927d11d38f
question: 'Conda environment: You are using Windows. You then use Waitress instead
  of Gunicorn. After a few runs, suddenly the MLflow server fails to run.'
sort_order: 34
---

1. Uninstall Waitress and MLflow:

   ```bash
   pip uninstall waitress mlflow
   ```

2. Reinstall MLflow:

   ```bash
   pip install mlflow
   ```

By this time, you should have successfully built your Docker image, so you don't need to reinstall Waitress.

All good. Happy learning.