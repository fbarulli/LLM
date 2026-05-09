---
id: 76e6d9e8ce
question: 'ImportError: urllib3 v2.0 only supports OpenSSL 1.1.1+'
sort_order: 45
---

If you're encountering an error while running S3 buckets, ensure to resolve the dependencies issue by downgrading urllib3 to a compatible version:

```bash
pip3 install "urllib3<1.27"
```