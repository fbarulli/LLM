---
id: 6b1c62c9cf
question: 'WARNING: mlflow.sklearn: Failed to log training dataset information to
  MLflow Tracking.'
sort_order: 44
---

### Problem

When using MLflow’s autolog function, you may encounter the following warning:

```
WARNING mlflow.sklearn: Failed to log training dataset information to MLflow Tracking. Reason: 'numpy.ndarray' object has no attribute 'toarray'
```

This occurs because the autolog function is attempting to log your dataset. MLflow expects the dataset to be in a `pd.DataFrame` format. If you’re following course code that provides a `numpy.ndarray`, MLflow fails as the `numpy.ndarray` is already an array.

### Solution

Since we are not processing datasets in this zoomcamp, use the following parameter in the autolog function to prevent logging datasets:

```python
log_datasets = False
```

---

### Problem

Error when running the mlflow server on AWS CLI with an S3 bucket and POSTGRES database:

#### Reproducible Command:

```bash
mlflow server -h 0.0.0.0 -p 5000 --backend-store-uri postgresql://<DB_USERNAME>:<DB_PASSWORD>@<DB_ENDPOINT>:<DB_PORT>/<DB_NAME> --default-artifact-root s3://<BUCKET_NAME>
```

#### Error Message:

```
urllib3 v2.0 only supports OpenSSL 1.1.1+, currently

ImportError: urllib3 v2.0 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'OpenSSL 1.0.2k-fips  26 Jan 2017'. See: [GitHub](https://github.com/urllib3/urllib3/issues/2168)
```

### Solution

Upgrade `mlflow` to address compatibility issues:

```bash
pip3 install --upgrade mlflow
```

### Resolution

This process will downgrade `urllib3` from version 2.0.3 to 1.26.16, ensuring compatibility with mlflow and `ssl` version 1.0.2. You should see the following output after the upgrade:

```bash
Installing collected packages: urllib3
Attempting uninstall: urllib3
Found existing installation: urllib3 2.0.3
Uninstalling urllib3-2.0.3:
Successfully uninstalled urllib3-2.0.3
Successfully installed urllib3-1.26.16
```