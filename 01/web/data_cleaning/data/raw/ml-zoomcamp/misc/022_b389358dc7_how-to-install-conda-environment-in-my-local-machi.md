---
id: b389358dc7
question: How to install Conda environment in my local machine?
sort_order: 22
---

You don’t install a conda environment. First, you create it, then you activate it.

**Step 1: How to create a conda environment?**

In a terminal, write the command (ml-zoomcamp is the name of the environment):

```bash
conda create -n ml-zoomcamp
```

**Step 2: How to activate a conda environment?**

```bash
conda activate ml-zoomcamp
```

You can verify that it worked if you see `(ml-zoomcamp)` prepended to your command prompt.

**Note:**

The answer above assumes Anaconda has already been installed on your local machine. If this is not the case, you can download it from [Anaconda’s download page](https://www.anaconda.com/download). After installing it, you can verify it succeeded with the following command in a terminal:

```bash
conda --version
```