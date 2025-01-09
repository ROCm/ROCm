.. meta::
   :description: JAX compatibility
   :keywords: GPU, JAX compatibility

*******************************************************************************
JAX compatibility
*******************************************************************************

JAX provides a NumPy-like API, which combines automatic differentiation and the
Accelerated Linear Algebra (XLA) compiler to achieve high-performance machine
learning at scale.

JAX uses composable transformations of Python and NumPy through just-in-time (JIT) compilation,
automatic vectorization, and parallelization. To learn about JAX, including profiling and
optimizations, see the official `JAX documentation
<https://jax.readthedocs.io/en/latest/notebooks/quickstart.html>`_.

ROCm support for JAX is upstreamed and users can build the official source code with ROCm
support:

- ROCm JAX release:

  - Offers AMD-validated and community :ref:`Docker images <jax-docker-compat>` with ROCm and JAX pre-installed.

  - ROCm JAX repository: `<https://github.com/ROCm/jax>`__

  - See the :doc:`ROCm JAX installation guide <rocm-install-on-linux:install/3rd-party/jax-install>`
    to get started.

- Official JAX release:

  - Official JAX repository: `<https://github.com/jax-ml/jax>`__

  - See the `AMD GPU (Linux) installation section
    <https://jax.readthedocs.io/en/latest/installation.html#amd-gpu-linux>`_ in the JAX
    documentation.

.. note::

   AMD releases official `ROCm JAX Docker images <https://hub.docker.com/r/rocm/jax>`_
   quarterly alongside new ROCm releases. These images undergo full AMD testing.
   `Community ROCm JAX Docker images <https://hub.docker.com/r/rocm/jax-community>`_
   follow upstream JAX releases and use the latest available ROCm version.

.. _jax-docker-compat:

Docker image compatibility
================================================================================

.. |docker-icon| raw:: html

   <i class="fab fa-docker"></i>

AMD validates and publishes ready-made `JAX <https://hub.docker.com/r/rocm/jax/>`_
images with ROCm backends on Docker Hub. The following Docker image tags and
associated inventories are validated for
`ROCm 6.3.1 <https://repo.radeon.com/rocm/apt/6.3.1/>`_. Click the |docker-icon|
icon to view the image on Docker Hub.

.. list-table:: JAX Docker image components
    :header-rows: 1

    * - Docker image
      - JAX
      - Linux
      - Python
    * - .. raw:: html

           <a href="https://hub.docker.com/layers/rocm/jax/rocm6.3.1-jax0.4.31-py3.12/images/sha256-085a0cd5207110922f1fca684933a9359c66d42db6c5aba4760ed5214fdabde0"><i class="fab fa-docker fa-lg"></i> rocm/jax</a>

      - `0.4.31 <https://github.com/ROCm/jax/releases/tag/rocm-jax-v0.4.31>`_
      - Ubuntu 24.04
      - `3.12.7 <https://www.python.org/downloads/release/python-3127/>`_
    * - .. raw:: html

           <a href="https://hub.docker.com/layers/rocm/jax/rocm6.3.1-jax0.4.31-py3.10/images/sha256-f88eddad8f47856d8640b694da4da347ffc1750d7363175ab7dc872e82b43324"><i class="fab fa-docker fa-lg"></i> rocm/jax</a>

      - `0.4.31 <https://github.com/ROCm/jax/releases/tag/rocm-jax-v0.4.31>`_
      - Ubuntu 22.04
      - `3.10.14 <https://www.python.org/downloads/release/python-31014/>`_

AMD publishes community `JAX <https://hub.docker.com/r/rocm/jax-community>`_
images with ROCm backends on Docker Hub. The following Docker image tags and
associated inventories are tested for `ROCm 6.2.4 <https://repo.radeon.com/rocm/apt/6.2.4/>`_.

.. list-table:: JAX community Docker image components
    :header-rows: 1

    * - Docker image
      - JAX
      - Linux
      - Python
    * - .. raw:: html

           <a href="https://hub.docker.com/layers/rocm/jax-community/rocm6.2.4-jax0.4.35-py3.12.7/images/sha256-a6032d89c07573b84c44e42c637bf9752b1b7cd2a222d39344e603d8f4c63beb?context=explore"><i class="fab fa-docker fa-lg"></i> rocm/jax-community</a>

      - `0.4.35 <https://github.com/ROCm/jax/releases/tag/rocm-jax-v0.4.35>`_
      - Ubuntu 22.04
      - `3.12.7 <https://www.python.org/downloads/release/python-3127/>`_
    * - .. raw:: html

           <a href="https://hub.docker.com/layers/rocm/jax-community/rocm6.2.4-jax0.4.35-py3.11.10/images/sha256-d462f7e445545fba2f3b92234a21beaa52fe6c5f550faabcfdcd1bf53486d991?context=explore"><i class="fab fa-docker fa-lg"></i> rocm/jax-community</a>

      - `0.4.35 <https://github.com/ROCm/jax/releases/tag/rocm-jax-v0.4.35>`_
      - Ubuntu 22.04
      - `3.11.10 <https://www.python.org/downloads/release/python-31110/>`_
    * - .. raw:: html

           <a href="https://hub.docker.com/layers/rocm/jax-community/rocm6.2.4-jax0.4.35-py3.10.15/images/sha256-6f2d4d0f529378d9572f0e8cfdcbc101d1e1d335bd626bb3336fff87814e9d60?context=explore"><i class="fab fa-docker fa-lg"></i> rocm/jax-community</a>

      - `0.4.35 <https://github.com/ROCm/jax/releases/tag/rocm-jax-v0.4.35>`_
      - Ubuntu 22.04
      - `3.10.15 <https://www.python.org/downloads/release/python-31015/>`_

Critical ROCm libraries for JAX
================================================================================

The functionality of JAX with ROCm is determined by its underlying library
dependencies. These critical ROCm components affect the capabilities,
performance, and feature set available to developers.

.. list-table::
    :header-rows: 1

    * - ROCm library
      - Version
      - Purpose
      - Used in
    * - `hipBLAS <https://github.com/ROCm/hipBLAS>`_
      - 2.3.0
      - Provides GPU-accelerated Basic Linear Algebra Subprograms (BLAS) for
        matrix and vector operations.
      - Matrix multiplication in ``jax.numpy.matmul``, ``jax.lax.dot`` and
        ``jax.lax.dot_general``, operations like ``jax.numpy.dot``, which
        involve vector and matrix computations and batch matrix multiplications
        ``jax.numpy.einsum`` with matrix-multiplication patterns algebra
        operations.
    * - `hipBLASLt <https://github.com/ROCm/hipBLASLt>`_
      - 0.10.0
      - hipBLASLt is an extension of hipBLAS, providing additional
        features like epilogues fused into the matrix multiplication kernel or
        use of integer tensor cores.
      - Matrix multiplication in ``jax.numpy.matmul`` or ``jax.lax.dot``, and
        the XLA (Accelerated Linear Algebra) use hipBLASLt for optimized matrix
        operations, mixed-precision support, and hardware-specific
        optimizations.
    * - `hipCUB <https://github.com/ROCm/hipCUB>`_
      - 3.3.0
      - Provides a C++ template library for parallel algorithms for reduction,
        scan, sort and select.
      - Reduction functions (``jax.numpy.sum``, ``jax.numpy.mean``, 
        ``jax.numpy.prod``, ``jax.numpy.max`` and ``jax.numpy.min``), prefix sum
        (``jax.numpy.cumsum``, ``jax.numpy.cumprod``) and sorting
        (``jax.numpy.sort``, ``jax.numpy.argsort``).
    * - `hipFFT <https://github.com/ROCm/hipFFT>`_
      - 1.0.17
      - Provides GPU-accelerated Fast Fourier Transform (FFT) operations.
      - Used in functions like ``jax.numpy.fft``.
    * - `hipRAND <https://github.com/ROCm/hipRAND>`_
      - 2.11.0
      - Provides fast random number generation for GPUs.
      - The ``jax.random.uniform``, ``jax.random.normal``,
        ``jax.random.randint`` and ``jax.random.split``.
    * - `hipSOLVER <https://github.com/ROCm/hipSOLVER>`_
      - 2.3.0
      - Provides GPU-accelerated solvers for linear systems, eigenvalues, and
        singular value decompositions (SVD).
      - Solving linear systems (``jax.numpy.linalg.solve``), matrix
        factorizations, SVD (``jax.numpy.linalg.svd``) and eigenvalue problems 
        (``jax.numpy.linalg.eig``).
    * - `hipSPARSE <https://github.com/ROCm/hipSPARSE>`_
      - 3.1.2
      - Accelerates operations on sparse matrices, such as sparse matrix-vector
        or matrix-matrix products.
      - Sparse matrix multiplication (``jax.numpy.matmul``), sparse
        matrix-vector and matrix-matrix products
        (``jax.experimental.sparse.dot``), sparse linear system solvers and
        sparse data handling.
    * - `hipSPARSELt <https://github.com/ROCm/hipSPARSELt>`_
      - 0.2.2
      - Accelerates operations on sparse matrices, such as sparse matrix-vector
        or matrix-matrix products.
      - Sparse matrix multiplication (``jax.numpy.matmul``), sparse
        matrix-vector and matrix-matrix products
        (``jax.experimental.sparse.dot``) and sparse linear system solvers.
    * - `MIOpen <https://github.com/ROCm/MIOpen>`_
      - 3.3.0
      - Optimized for deep learning primitives such as convolutions, pooling,
        normalization, and activation functions.
      - Speeds up convolutional neural networks (CNNs), recurrent neural
        networks (RNNs), and other layers. Used in operations like
        ``jax.nn.conv``, ``jax.nn.relu``, and ``jax.nn.batch_norm``.
    * - `RCCL <https://github.com/ROCm/rccl>`_
      - 2.21.5
      - Optimized for multi-GPU communication for operations like  all-reduce,
        broadcast, and scatter.
      - Distribute computations across multiple GPU with ``pmap`` and
        ``jax.distributed``. XLA automatically uses rccl when executing
        operations across multiple GPUs on AMD hardware.
    * - `rocThrust <https://github.com/ROCm/rocThrust>`_
      - 3.3.0
      - Provides a C++ template library for parallel algorithms like sorting,
        reduction, and scanning.
      - Reduction operations like ``jax.numpy.sum``, ``jax.pmap`` for
        distributed training, which involves parallel reductions or
        operations like ``jax.numpy.cumsum`` can use rocThrust.

Supported and unsupported features
===============================================================================

The following table maps GPU-accelerated JAX modules to their supported
ROCm and JAX versions.

.. list-table::
    :header-rows: 1

    * - Module
      - Description
      - Since JAX
      - Since ROCm
    * - ``jax.numpy``
      - Implements the NumPy API, using the primitives in ``jax.lax``.
      - 0.1.56
      - 5.0.0
    * - ``jax.scipy``
      - Provides GPU-accelerated and differentiable implementations of many
        functions from the SciPy library, leveraging JAX's transformations
        (e.g., ``grad``, ``jit``, ``vmap``).
      - 0.1.56
      - 5.0.0
    * - ``jax.lax``
      - A library of primitives operations that underpins libraries such as
        ``jax.numpy.`` Transformation rules, such as Jacobian-vector product
        (JVP) and batching rules, are typically defined as transformations on
        ``jax.lax`` primitives.
      - 0.1.57
      - 5.0.0
    * - ``jax.random``
      - Provides a number of routines for deterministic generation of sequences
        of pseudorandom numbers.
      - 0.1.58
      - 5.0.0
    * - ``jax.sharding``
      - Allows to define partitioning and distributing arrays across multiple
        devices.
      - 0.3.20
      - 5.1.0
    * - ``jax.dlpack``
      - For exchanging tensor data between JAX and other libraries that support the
        DLPack standard.
      - 0.1.57
      - 5.0.0
    * - ``jax.distributed``
      - Enables the scaling of computations across multiple devices on a single
        machine or across multiple machines.
      - 0.1.74
      - 5.0.0
    * - ``jax.dtypes``
      - Provides utilities for working with and managing data types in JAX
        arrays and computations.
      - 0.1.66
      - 5.0.0 
    * - ``jax.image``
      - Contains image manipulation functions like resize, scale and translation.
      - 0.1.57
      - 5.0.0
    * - ``jax.nn``
      - Contains common functions for neural network libraries.
      - 0.1.56
      - 5.0.0
    * - ``jax.ops``
      - Computes the minimum, maximum, sum or product within segments of an
        array.
      - 0.1.57
      - 5.0.0
    * - ``jax.profiler``
      - Contains JAX’s tracing and time profiling features.
      - 0.1.57
      - 5.0.0
    * - ``jax.stages``
      - Contains interfaces to stages of the compiled execution process.
      - 0.3.4
      - 5.0.0
    * - ``jax.tree``
      - Provides utilities for working with tree-like container data structures.
      - 0.4.26
      - 5.6.0
    * - ``jax.tree_util``
      - Provides utilities for working with nested data structures, or
        ``pytrees``.
      - 0.1.65
      - 5.0.0
    * - ``jax.typing``
      - Provides JAX-specific static type annotations.
      - 0.3.18
      - 5.1.0
    * - ``jax.extend``
      - Provides modules for access to JAX internal machinery module. The
        ``jax.extend`` module defines a library view of some of JAX’s internal
        components.
      - 0.4.15
      - 5.5.0
    * - ``jax.example_libraries``
      - Serves as a collection of example code and libraries that demonstrate
        various capabilities of JAX.
      - 0.1.74
      - 5.0.0
    * - ``jax.experimental``
      - Namespace for experimental features and APIs that are in development or
        are not yet fully stable for production use.
      - 0.1.56
      - 5.0.0
    * - ``jax.lib``
      - Set of internal tools and types for bridging between JAX’s Python
        frontend and its XLA backend.
      - 0.4.6
      - 5.3.0
    * - ``jax_triton``
      - Library that integrates the Triton deep learning compiler with JAX.
      - jax_triton 0.2.0 
      - 6.2.4

jax.scipy module
-------------------------------------------------------------------------------

A SciPy-like API for scientific computing.

.. list-table::
    :header-rows: 1

    * - Module
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

jax.scipy.stats module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Module
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

jax.extend module
-------------------------------------------------------------------------------

Modules for JAX extensions.

.. list-table::
    :header-rows: 1

    * - Module
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

jax.experimental module
-------------------------------------------------------------------------------

Experimental modules and APIs.

.. list-table::
    :header-rows: 1

    * - Module
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

jax.experimental.pallas module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Module for Pallas, a JAX extension for custom kernels.

.. list-table::
    :header-rows: 1

    * - Module
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

jax.experimental.sparse module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Experimental support for sparse matrix operations.

.. list-table::
    :header-rows: 1

    * - Module
      - Since JAX
      - Since ROCm
    * - ``jax.experimental.sparse.linalg``
      - 0.3.15
      - 5.2.0
    * - ``jax.experimental.sparse.sparsify``
      - 0.3.25
      - ❌

.. list-table::
    :header-rows: 1

    * - ``sparse`` data structure API
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

Unsupported JAX features
------------------------

The following are GPU-accelerated JAX features not currently supported by
ROCm.

.. list-table::
    :header-rows: 1

    * - Feature
      - Description
      - Since JAX
    * - Mixed Precision with TF32
      - Mixed precision with TF32 is used for matrix multiplications,
        convolutions, and other linear algebra operations, particularly in
        deep learning workloads like CNNs and transformers.
      - 0.2.25
    * - RNN support
      - Currently only LSTM with double bias is supported with float32 input
        and weight.
      - 0.3.25
    * - XLA int4 support
      - 4-bit integer (int4) precision in the XLA compiler.
      - 0.4.0
    * - ``jax.experimental.sparsify``
      - Converts a dense matrix to a sparse matrix representation.
      - Experimental

Use cases and recommendations
================================================================================

* The `nanoGPT in JAX <https://rocm.blogs.amd.com/artificial-intelligence/nanoGPT-JAX/README.html>`_
  blog explores the implementation and training of a Generative Pre-trained
  Transformer (GPT) model in JAX, inspired by Andrej Karpathy’s PyTorch-based
  nanoGPT. By comparing how essential GPT components—such as self-attention
  mechanisms and optimizers—are realized in PyTorch and JAX, also highlight
  JAX’s unique features.

* The `Optimize GPT Training: Enabling Mixed Precision Training in JAX using
  ROCm on AMD GPUs <https://rocm.blogs.amd.com/artificial-intelligence/jax-mixed-precision/README.html>`_
  blog post provides a comprehensive guide on enhancing the training efficiency
  of GPT models by implementing mixed precision techniques in JAX, specifically
  tailored for AMD GPUs utilizing the ROCm platform.

* The `Supercharging JAX with Triton Kernels on AMD GPUs <https://rocm.blogs.amd.com/artificial-intelligence/jax-triton/README.html>`_
  blog demonstrates how to develop a custom fused dropout-activation kernel for
  matrices using Triton, integrate it with JAX, and benchmark its performance
  using ROCm.

* The `Distributed fine-tuning with JAX on AMD GPUs <https://rocm.blogs.amd.com/artificial-intelligence/distributed-sft-jax/README.html>`_
  outlines the process of fine-tuning a Bidirectional Encoder Representations
  from Transformers (BERT)-based large language model (LLM) using JAX for a text
  classification task. The blog post discuss techniques for parallelizing the
  fine-tuning across multiple AMD GPUs and assess the model's performance on a
  holdout dataset. During the fine-tuning, a BERT-base-cased transformer model
  and the General Language Understanding Evaluation (GLUE) benchmark dataset was
  used on a multi-GPU setup.

* The `MI300X workload optimization guide <https://rocm.docs.amd.com/en/latest/how-to/tuning-guides/mi300x/workload.html>`_
  provides detailed guidance on optimizing workloads for the AMD Instinct MI300X
  accelerator using ROCm. The page is aimed at helping users achieve optimal
  performance for deep learning and other high-performance computing tasks on
  the MI300X GPU.

For more use cases and recommendations, see `ROCm JAX blog posts <https://rocm.blogs.amd.com/blog/tag/jax.html>`_.
