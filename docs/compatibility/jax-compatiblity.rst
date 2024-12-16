.. meta::
    :description: JAX compatibility
    :keywords: GPU, JAX compatibility

*******************************************************************************
JAX compatibility
*******************************************************************************

JAX provides a NumPy-like API, which combines automatic differentiation and the
Accelerated Linear Algebra (XLA) compiler to achieve high-performance machine
learning at scale.

JAX uses composable transformations of Python+NumPy through just-in-time (JIT)
compilation, automatic vectorization, and parallelization.

To learn about JAX, including profiling and optimizations, refer to the
`JAX documentation <https://jax.readthedocs.io/en/latest/notebooks/quickstart.html>`_.

JAX already upstreamed to official repository and the JAX team also
releases JAX with ROCm support, which means the JAX has two different
release cycle with ROCm support:

- Official JAX release:

  - Support the latest stable PyTorch releases with the latest or one minor
    version behind ROCm.

  - `Official JAX repository <https://github.com/pytorch/pytorch>`_

  - `Nightly and latest stable version installation guide <https://jax.readthedocs.io/en/latest/installation.html#installation/>`_.

  - `Previous versions installation guide <https://jax.readthedocs.io/en/latest/installation.html#installing-older-jaxlib-wheels/>`_

- ROCm JAX release:

  - Support the one or two minor version behind PyTorch releases with the latest
    ROCm.

  - `Docker images <https://hub.docker.com/r/rocm/jax>`_ with preinstalled
    PyTorch and ROCm.

  - `ROCm PyTorch repository <https://github.com/rocm/jax>`_

  - :doc:`ROCm PyTorch installation guide <rocm-install-on-linux:install/3rd-party/jax-install>`

ROCm release compatibility matrix
===============================================================================

This table shows the compatibility between various JAX versions and the
corresponding ROCm and Python versions, ensuring optimal performance and
compatibility for your projects:

.. list-table::
    :header-rows: 1
    :name: jax-rocm-compatibility

    * - JAX versions
      - ROCm version
      - Python version
    * - 0.4.35
      - 6.2.4, 6.1.3, 6.0.3
      - 3.12.7, 3.11.10, 3.10.15
    * - 0.4.34
      - 6.2.3, 6.1.3, 6.0.3
      - 3.12.6, 3.11.10, 3.10.15
    * - 0.4.33
      - 6.2.3, 6.1.3, 6.0.3
      - 3.12.6, 3.11.10, 3.10.15
    * - 0.4.31
      - 6.2.3, 6.1.3, 6.0.3
      - 3.12.6, 3.11.10, 3.10.15
    * - 0.4.30
      - 6.1.1, 6.0.2 
      - 3.11.9, 3.10.14, 3.9.19
    * - 0.4.29
      - 6.1.1, 6.0.2 
      - 3.11.9, 3.10.14, 3.9.19

Docker image compatibility
-------------------------------------------------------------------------------

AMD validates and publishes ready-made `JAX
<https://hub.docker.com/r/rocm/jax/>`_ images with ROCm backends on Docker Hub.
The following Docker image tags and associated inventories are validated for
ROCm 6.2.

.. list-table:: PyTorch docker image components
    :header-rows: 1

    * - Docker image
      - ROCm
      - JAX
      - Linux
      - Python
    * - `rocm6.2.4-jax0.4.35-py3.12.7 <https://hub.docker.com/layers/rocm/jax-community/rocm6.2.4-jax0.4.35-py3.12.7/images/sha256-a6032d89c07573b84c44e42c637bf9752b1b7cd2a222d39344e603d8f4c63beb?context=explore>`_
    - `6.2.4 <https://repo.radeon.com/rocm/apt/6.2.4/>`_
    - `0.4.35 <https://github.com/ROCm/jax/releases/tag/rocm-jax-v0.4.35>`_
    - **Ubuntu 22.04**
    - `3.12.7 <https://www.python.org/downloads/release/python-3127/>`_
  * - `rocm6.2.4-jax0.4.35-py3.11.10 <https://hub.docker.com/layers/rocm/jax-community/rocm6.2.4-jax0.4.35-py3.11.10/images/sha256-d462f7e445545fba2f3b92234a21beaa52fe6c5f550faabcfdcd1bf53486d991?context=explore>`_
    - `6.2.4 <https://repo.radeon.com/rocm/apt/6.2.4/>`_
    - `0.4.35 <https://github.com/ROCm/jax/releases/tag/rocm-jax-v0.4.35>`_
    - **Ubuntu 22.04**
    - `3.11.10 <https://www.python.org/downloads/release/python-31110/>`_
  * - `rocm6.2.4-jax0.4.35-py3.10.15 <https://hub.docker.com/layers/rocm/jax-community/rocm6.2.4-jax0.4.35-py3.10.15/images/sha256-6f2d4d0f529378d9572f0e8cfdcbc101d1e1d335bd626bb3336fff87814e9d60?context=explore>`_
    - `6.2.4 <https://repo.radeon.com/rocm/apt/6.2.4/>`_
    - `0.4.35 <https://github.com/ROCm/jax/releases/tag/rocm-jax-v0.4.35>`_
    - **Ubuntu 22.04**
    - `3.10.15 <https://www.python.org/downloads/release/python-31015/>`_
  * - `rocm6.2.3-jax0.4.34-py3.12.6 <https://hub.docker.com/layers/rocm/jax-community/rocm6.2.3-jax0.4.34-py3.12.6/images/sha256-c9063cc512bc6385721bb00790bf6a013a01b86940aa11d45359f23c9d995c1e?context=explore>`_
    - `6.2.3 <https://repo.radeon.com/rocm/apt/6.2.3/>`_
    - `0.4.34 <https://github.com/ROCm/jax/releases/tag/rocm-jax-v0.4.34>`_
    - **Ubuntu 20.04**
    - `3.12.6 <https://www.python.org/downloads/release/python-3126/>`_
  * - `rocm6.2.3-jax0.4.34-py3.11.10 <https://hub.docker.com/layers/rocm/jax-community/rocm6.2.3-jax0.4.34-py3.11.10/images/sha256-243e9bcbefb8f8af8b167b214ac5356542f01328182bc0772dddcd0cdaf55072?context=explore>`_
    - `6.2.3 <https://repo.radeon.com/rocm/apt/6.2.3/>`_
    - `0.4.34 <https://github.com/ROCm/jax/releases/tag/rocm-jax-v0.4.34>`_
    - **Ubuntu 20.04**
    - `3.11.10 <https://www.python.org/downloads/release/python-31110/>`_
  * - `rocm6.2.3-jax0.4.34-py3.10.15 <https://hub.docker.com/layers/rocm/jax-community/rocm6.2.3-jax0.4.34-py3.10.15/images/sha256-97c2e0c7462de0bd1586e1fead4ffad60774b8e3ce3037ba3c7c47fa06e7ae73?context=explore>`_
    - `6.2.3 <https://repo.radeon.com/rocm/apt/6.2.3/>`_
    - `0.4.34 <https://github.com/ROCm/jax/releases/tag/rocm-jax-v0.4.34>`_
    - **Ubuntu 20.04**
    - `3.10.15 <https://www.python.org/downloads/release/python-31015/>`_
  * - `rocm6.2.3-jax0.4.33-py3.12.6 <https://hub.docker.com/layers/rocm/jax-community/rocm6.2.3-jax0.4.33-py3.12.6/images/sha256-8cb16b1fba8f949da23690195c5fe8d450fc05ea4e01aabda8160e9a8ca1d238?context=explore>`_
    - `6.2.3 <https://repo.radeon.com/rocm/apt/6.2.3/>`_
    - `0.4.33 <https://github.com/ROCm/jax/releases/tag/rocm-jax-v0.4.33>`_
    - **Ubuntu 20.04**
    - `3.12.6 <https://www.python.org/downloads/release/python-3126/>`_
  * - `rocm6.2.3-jax0.4.33-py3.11.10 <https://hub.docker.com/layers/rocm/jax-community/rocm6.2.3-jax0.4.33-py3.11.10/images/sha256-abc8167fd2b612b28b655f8b995ed1c08a27157c3fc73c85399c82ae8bf3f7d0?context=explore>`_
    - `6.2.3 <https://repo.radeon.com/rocm/apt/6.2.3/>`_
    - `0.4.33 <https://github.com/ROCm/jax/releases/tag/rocm-jax-v0.4.33>`_
    - **Ubuntu 20.04**
    - `3.11.10 <https://www.python.org/downloads/release/python-31110/>`_
  * - `rocm6.2.3-jax0.4.33-py3.10.15 <https://hub.docker.com/layers/rocm/jax-community/rocm6.2.3-jax0.4.33-py3.10.15/images/sha256-217f2fbaef52c9f7fd6253d28886cdaa694923a8c1fbc28c6c283e1e4eb1cc77?context=explore>`_
    - `6.2.3 <https://repo.radeon.com/rocm/apt/6.2.3/>`_
    - `0.4.33 <https://github.com/ROCm/jax/releases/tag/rocm-jax-v0.4.33>`_
    - **Ubuntu 20.04**
    - `3.10.15 <https://www.python.org/downloads/release/python-31015/>`_
  * - `rocm6.2.3-jax0.4.31-py3.12.6 <https://hub.docker.com/layers/rocm/jax-community/rocm6.2.3-jax0.4.31-py3.12.6/images/sha256-595679a21f2ac332bf38197a2cf5cd411dff59f2616cf9802fb1700f96fa5906?context=explore>`_
    - `6.2.3 <https://repo.radeon.com/rocm/apt/6.2.3/>`_
    - `0.4.31 <https://github.com/ROCm/jax/releases/tag/rocm-jax-v0.4.31>`_
    - **Ubuntu 20.04**
    - `3.12.6 <https://www.python.org/downloads/release/python-3126/>`_
  * - `rocm6.2.3-jax0.4.31-py3.11.10 <https://hub.docker.com/layers/rocm/jax-community/rocm6.2.3-jax0.4.31-py3.11.10/images/sha256-0c38612d0f4d34fb66e3a7132564f068b1bd22599f347bca7007efbc3b709165?context=explore>`_
    - `6.2.3 <https://repo.radeon.com/rocm/apt/6.2.3/>`_
    - `0.4.31 <https://github.com/ROCm/jax/releases/tag/rocm-jax-v0.4.31>`_
    - **Ubuntu 20.04**
    - `3.11.10 <https://www.python.org/downloads/release/python-31110/>`_
  * - `rocm6.2.3-jax0.4.31-py3.10.15 <https://hub.docker.com/layers/rocm/jax-community/rocm6.2.3-jax0.4.31-py3.10.15/images/sha256-b7e7b68ba0fb293e66bc7aa1187a0a641e25276151237de56d567625caac1dde?context=explore>`_
    - `6.2.3 <https://repo.radeon.com/rocm/apt/6.2.3/>`_
    - `0.4.31 <https://github.com/ROCm/jax/releases/tag/rocm-jax-v0.4.31>`_
    - **Ubuntu 20.04**
    - `3.10.15 <https://www.python.org/downloads/release/python-31015/>`_

Supported features
===============================================================================

.. list-table::
    :header-rows: 1

    * - module
      - Since JAX
      - Since ROCm
    * - ``jax.numpy``
      - 0.1.56
      - 5.0.0
    * - ``jax.scipy``
      - 0.1.56
      - 5.0.0
    * - ``jax.lax``
      - 0.1.57
      - 5.0.0
    * - ``jax.random``
      - 0.1.58
      - 5.0.0
    * - ``jax.sharding``
      - 0.3.20
      - 5.1.0
    * - ``jax.debug``
      - 0.3.11
      - 5.1.0
    * - ``jax.dlpack``
      - 0.1.57
      - 5.0.0
    * - ``jax.distributed``
      - 0.1.74
      - 5.0.0
    * - ``jax.dtypes``
      - 0.1.66
      - 5.0.0
    * - ``jax.flatten_util``
      - 0.1.72
      - 5.0.0
    * - ``jax.image``
      - 0.1.57
      - 5.0.0
    * - ``jax.nn``
      - 0.1.56
      - 5.0.0
    * - ``jax.ops``
      - 0.1.57
      - 5.0.0
    * - ``jax.profiler``
      - 0.1.57
      - 5.0.0
    * - ``jax.stages``
      - 0.3.4
      - 5.0.0
    * - ``jax.tree``
      - 0.4.26
      - 5.6.0
    * - ``jax.tree_util``
      - 0.1.65
      - 5.0.0
    * - ``jax.typing``
      - 0.3.18
      - 5.1.0
    * - ``jax.export``
      - 0.4.30
      - 6.0.0
    * - ``jax.extend``
      - 0.4.15
      - 5.5.0
    * - ``jax.example_libraries``
      - 0.1.74
      - 5.0.0
    * - ``jax.experimental``
      - 0.1.56
      - 5.0.0
    * - ``jax.lib``
      - 0.4.6
      - 5.3.0

``jax.scipy``
-------------------------------------------------------------------------------

A SciPy-like API for scientific computing.

.. list-table::
    :header-rows: 1

    * - module
      - Since JAX
      - Since ROCm
    * - ``jax.scipy.cluster``
      - 0.3.11
      - 5.1.0
    * - ``jax.scipy.fft``
      - 0.1.71
      - 5.0.0
    * - ``jax.scipy.integrate``
      - 0.4.15
      - 5.5.0
    * - ``jax.scipy.interpolate``
      - 0.1.76
      - 5.0.0
    * - ``jax.scipy.linalg``
      - 0.1.56
      - 5.0.0
    * - ``jax.scipy.ndimage``
      - 0.1.56
      - 5.0.0
    * - ``jax.scipy.optimize``
      - 0.1.57
      - 5.0.0
    * - ``jax.scipy.signal``
      - 0.1.56
      - 5.0.0
    * - ``jax.scipy.spatial.transform``
      - 0.4.12
      - 5.4.0
    * - ``jax.scipy.sparse.linalg``
      - 0.1.56
      - 5.0.0
    * - ``jax.scipy.special``
      - 0.1.56
      - 5.0.0
    * - ``jax.scipy.stats``
      - 0.1.56
      - 5.0.0

``jax.scipy.stats``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - module
     - Since JAX
     - Since ROCm
   * - ``jax.scipy.stats.bernouli``
     - 0.1.56
     - 5.0.0
   * - ``jax.scipy.stats.beta``
     - 0.1.56
     - 5.0.0
   * - ``jax.scipy.stats.betabinom``
     - 0.1.61
     - 5.0.0
   * - ``jax.scipy.stats.binom``
     - 0.4.14
     - 5.4.0
   * - ``jax.scipy.stats.cauchy``
     - 0.1.56
     - 5.0.0
   * - ``jax.scipy.stats.chi2``
     - 0.1.61
     - 5.0.0
   * - ``jax.scipy.stats.dirichlet``
     - 0.1.56
     - 5.0.0
   * - ``jax.scipy.stats.expon``
     - 0.1.56
     - 5.0.0
   * - ``jax.scipy.stats.gamma``
     - 0.1.56
     - 5.0.0
   * - ``jax.scipy.stats.gennorm``
     - 0.3.15
     - 5.2.0
   * - ``jax.scipy.stats.geom``
     - 0.1.56
     - 5.0.0
   * - ``jax.scipy.stats.laplace``
     - 0.1.56
     - 5.0.0
   * - ``jax.scipy.stats.logistic``
     - 0.1.56
     - 5.0.0
   * - ``jax.scipy.stats.multinomial``
     - 0.3.18
     - 5.1.0
   * - ``jax.scipy.stats.multivariate_normal``
     - 0.1.56
     - 5.0.0
   * - ``jax.scipy.stats.nbinom``
     - 0.1.72
     - 5.0.0
   * - ``jax.scipy.stats.norm``
     - 0.1.56
     - 5.0.0
   * - ``jax.scipy.stats.pareto``
     - 0.1.56
     - 5.0.0
   * - ``jax.scipy.stats.poisson``
     - 0.1.56
     - 5.0.0
   * - ``jax.scipy.stats.t``
     - 0.1.56
     - 5.0.0
   * - ``jax.scipy.stats.truncnorm``
     - 0.4.0
     - 5.3.0
   * - ``jax.scipy.stats.uniform``
     - 0.1.56
     - 5.0.0
   * - ``jax.scipy.stats.vonmises``
     - 0.4.2
     - 5.3.0
   * - ``jax.scipy.stats.wrapcauchy``
     - 0.4.20
     - 5.6.0

``jax.extend``
-------------------------------------------------------------------------------

Modules for JAX extensions.

.. list-table::
    :header-rows: 1

    * - module
      - Since JAX
      - Since ROCm
    * - ``jax.extend.ffi``
      - 0.4.30
      - 6.0.0
    * - ``jax.extend.linear_util``
      - 0.4.17
      - 5.6.0
    * - ``jax.extend.mlir``
      - 0.4.26
      - 5.6.0
    * - ``jax.extend.random``
      - 0.4.15
      - 5.5.0

``jax.experimental``
-------------------------------------------------------------------------------

Experimental modules and APIs.

.. list-table::
    :header-rows: 1

    * - module
      - Since JAX
      - Since ROCm
    * - ``jax.experimental.checkify``
      - 0.1.75
      - 5.0.0
    * - ``jax.experimental.compilation_cache.compilation_cache``
      - 0.1.68
      - 5.0.0
    * - ``jax.experimental.custom_partitioning``
      - 0.4.0
      - 5.3.0
    * - ``jax.experimental.jet``
      - 0.1.56
      - 5.0.0
    * - ``jax.experimental.key_reuse``
      - 0.4.26
      - 5.6.0
    * - ``jax.experimental.mesh_utils``
      - 0.1.76
      - 5.0.0
    * - ``jax.experimental.multihost_utils``
      - 0.3.2
      - 5.0.0
    * - ``jax.experimental.pallas``
      - 0.4.15
      - 5.5.0
    * - ``jax.experimental.pjit``
      - 0.1.61
      - 5.0.0
    * - ``jax.experimental.serialize_executable``
      - 0.4.0
      - 5.3.0
    * - ``jax.experimental.shard_map``
      - 0.4.3
      - 5.3.0
    * - ``jax.experimental.sparse``
      - 0.1.75
      - 5.0.0

.. list-table::
    :header-rows: 1

    * - API
      - Since JAX
      - Since ROCm
    * - ``jax.experimental.enable_x64``
      - 0.1.60
      - 5.0.0
    * - ``jax.experimental.disable_x64``
      - 0.1.60
      - 5.0.0

``jax.experimental.pallas``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Module for Pallas, a JAX extension for custom kernels.

.. list-table::
    :header-rows: 1

    * - module
      - Since JAX
      - Since ROCm
    * - ``jax.experimental.pallas.mosaic_gpu``
      - 0.4.31
      - 6.1.3
    * - ``jax.experimental.pallas.tpu``
      - 0.4.15
      - 5.5.0
    * - ``jax.experimental.pallas.triton``
      - 0.4.32
      - 6.1.3

``jax.experimental.sparse``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Experimental support for sparse matrix operations.

.. list-table::
    :header-rows: 1

    * - module
      - Since JAX
      - Since ROCm
    * - ``jax.experimental.sparse.linalg``
      - 0.3.15
      - 5.2.0

.. list-table::
    :header-rows: 1

    * - sparse data structure API
      - Since JAX
      - Since ROCm
    * - ``jax.experimental.sparse.BCOO``
      - 0.1.72
      - 5.0.0
    * - ``jax.experimental.sparse.BCSR``
      - 0.3.20
      - 5.1.0
    * - ``jax.experimental.sparse.CSR``
      - 0.1.75
      - 5.0.0
    * - ``jax.experimental.sparse.NM``
      - 0.4.27
      - 5.6.0
    * - ``jax.experimental.sparse.COO``
      - 0.1.75
      - 5.0.0

Use cases and recommendations
================================================================================

The page `ROCm for AI: Train a Model <https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/train-a-model.html>`_ 
provides guidance on how to leverage the ROCm platform for training AI models.
It covers the steps, tools, and best practices for optimizing training workflows
on AMD GPUs using PyTorch features.

The `Single-GPU Fine-Tuning and Inference page <https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/single-gpu-fine-tuning-and-inference.html>`_
describes how to use the ROCm platform for fine-tuning and inference of machine
learning models, particularly Large Language Models (LLMs), on systems with a
single AMD GPU. The page provides a detailed guide for setting up, optimizing,
and executing fine-tuning and inference workflows in such environments.

The `Multi-GPU Fine-Tuning and Inference page <https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/single-gpu-fine-tuning-and-inference.html>`_
describe fine-tuning and inference of machine learning models on system with
multi GPU cases.

The `MI300X Workload Optimization page <https://rocm.docs.amd.com/en/latest/how-to/tuning-guides/mi300x/workload.html>`_
provides detailed guidance on optimizing workloads for the AMD Instinct MI300X
accelerator using ROCm. The page is aimed at helping users achieve optimal
performance for deep learning and other high-performance computing tasks on the
MI300X GPU.

The `AI PyTorch Inception page <https://rocm.docs.amd.com/en/latest/conceptual/ai-pytorch-inception.html>`_ 
describes how PyTorch integrates with ROCm for AI workloads It outlines the use
of PyTorch on the ROCm platform and focuses on how to efficiently leverage AMD's
GPU hardware for training and inference tasks in AI applications.

For more use cases and recommendations, please check `ROCm JAX blog posts <https://rocm.blogs.amd.com/blog/tag/jax.html>`_

