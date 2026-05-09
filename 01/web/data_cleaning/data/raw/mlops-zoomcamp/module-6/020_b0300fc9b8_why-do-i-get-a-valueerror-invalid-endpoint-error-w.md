---
id: b0300fc9b8
question: 'Why do I get a `ValueError: Invalid endpoint` error when using Boto3 with
  Docker Compose services?'
sort_order: 20
---

Boto3 does not support underscores (_) in service URLs. Naming your Docker Compose services with underscores will cause Boto3 to throw an error when connecting to the endpoint. (Source: [GitHub Issue](https://github.com/boto/boto3/issues/703))

### Incorrect Docker Compose configuration with underscores

```yaml
docker-compose.yml

version: '3.8'

services:
  backend_service:
    image: my_backend_image
    ...
  s3_service:
    image: localstack/localstack
    …
```

Rename your services to avoid using underscores. For example, change `s3_service` to `s3service`.

This way, when you run 

```python
client = boto3.client('s3', endpoint_url="http://s3service:4566")
```

You won’t get any error.

---

### Problem: Pre-commit fails with `RuntimeError: The Poetry configuration is invalid:`

```
data.extras.pipfile_deprecated_finder[2] must match pattern ^[a-zA-Z-_.0-9]+$
```

### Solution:

This is caused by a version mismatch between the `pre-commit-config.yaml` designated version for your package and the actual versions. Check the versions in `Pipfile.lock` and update as appropriate.