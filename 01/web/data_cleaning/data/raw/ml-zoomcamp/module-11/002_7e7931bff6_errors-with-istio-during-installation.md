---
id: 7e7931bff6
question: Errors with istio during installation
sort_order: 2
---

Running the following command:

```bash
curl -s "https://raw.githubusercontent.com/kserve/kserve/release-0.9/hack/quick_install.sh" | bash
```

Fails with errors due to Istio failing to update resources when using `kubectl` version greater than 1.25.0.

Check your `kubectl` version with:

```bash
kubectl version
```


1. Download the "quick_install.bash" script without executing it:
   
   ```bash
   curl -O https://raw.githubusercontent.com/kserve/kserve/release-0.9/hack/quick_install.sh
   ```

2. Edit the downloaded script to update the versions of Istio and Knative according to the [recommended version matrix on the KServe website](https://kserve.github.io/website/master/admin/serverless/serverless/#recommended-version-matrix).

3. Run the modified bash script.

By following these steps, you should avoid the installation errors related to Istio.