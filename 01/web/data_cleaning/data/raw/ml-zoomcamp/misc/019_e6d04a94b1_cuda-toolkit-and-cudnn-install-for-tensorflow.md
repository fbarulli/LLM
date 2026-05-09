---
id: e6d04a94b1
question: CUDA toolkit and cuDNN Install for Tensorflow
sort_order: 19
---

Install Nvidia drivers: [https://www.nvidia.com/download/index.aspx](https://www.nvidia.com/download/index.aspx).

Windows:

- Install Anaconda prompt: [https://www.anaconda.com/](https://www.anaconda.com/)
- Two options:
  1. Install package `tensorflow-gpu` in Anaconda.
  2. Install Tensorflow [with pip](https://www.tensorflow.org/install/pip#windows-native)

WSL/Linux:

- WSL: Use the Windows Nvidia drivers, do not modify them.
- Two options:
  1. Install the Tensorflow [with pip](https://www.tensorflow.org/install/pip#linux_1)
     - Make sure to follow step 4 to install CUDA by environment.
     - Also run:

     ```bash
     echo 'export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
     ```
  2. Install CUDA toolkit 11.x.x: [link](https://developer.nvidia.com/cuda-toolkit-archive)
  
- Install cuDNN: [link](https://developer.nvidia.com/rdp/cudnn-download)

Now you should be able to perform training/inference with GPU in Tensorflow.
