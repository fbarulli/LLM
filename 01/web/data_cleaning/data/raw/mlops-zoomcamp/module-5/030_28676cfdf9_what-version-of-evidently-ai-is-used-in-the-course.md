---
id: 28676cfdf9
question: What version of Evidently AI is used in the course?
sort_order: 30
---

In the video (current cohort: 2025), the Evidently version used is 0.4.17. However, any version up to 0.6.7 will work with the code provided in the video and the repository.

Note that newer versions have changed the APIs, so the code in the video may not run with versions beyond 0.6.7.

### Error: Failed to create provisioner when running `docker-compose up –build`

```
✗ Failed to create provisioner: Failed to read dashboards config: could not parse provisioning config file: dashboards.yaml error: read /etc/grafana/provisioning/dashboards/dashboards.yaml: is a directory
```

To resolve this error in your `docker-compose.yml` file, update the Grafana `volumes`:

- Change from a YML file reference to a directory reference.
- Instead of specifying `/etc/grafana/provisioning/dashboards/dashboards.yaml`, use `/etc/grafana/provisioning/dashboards/dashboards`.
- Apply this change to all file names in the Grafana `volumes` section.