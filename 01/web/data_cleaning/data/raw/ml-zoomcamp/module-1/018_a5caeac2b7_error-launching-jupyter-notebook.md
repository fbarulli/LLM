---
id: a5caeac2b7
question: Error launching Jupyter notebook
sort_order: 18
---

If you encounter the error below when launching a Jupyter notebook in a new environment:

```bash
ImportError: cannot import name 'contextfilter' from 'jinja2' (anaconda\lib\site-packages\jinja2\__init__.py)
```

Follow these steps:

1. Switch to the main environment.
2. Run the following command:

   ```bash
   pip install nbconvert --upgrade
   ```