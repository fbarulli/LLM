---
id: bcedbb12fe
question: 'Could not install packages due to an OSError: [WinError 5] Access is denied'
sort_order: 8
---

When I ran the command:

```bash
pip install grpcio==1.42.0 tensorflow-serving-api==2.7.0
```

to install the libraries on a Windows machine, I encountered the following error:

```
ERROR: Could not install packages due to an OSError: [WinError 5] Access is denied: 'C:\Users\Asia\anaconda3\Lib\site-packages\google\protobuf\internal\_api_implementation.cp39-win_amd64.pyd'
Consider using the `--user` option or check the permissions.
```

I was able to successfully install the libraries using the following command:

```bash
pip --user install grpcio==1.42.0 tensorflow-serving-api==2.7.0
```
