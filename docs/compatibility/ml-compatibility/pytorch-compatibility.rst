.. meta::
    :description: PyTorch compatibility
    :keywords: GPU, PyTorch compatibility

********************************************************************************
PyTorch compatibility
********************************************************************************

`PyTorch <https://pytorch.org/>`_ is an open-source tensor library designed for
deep learning. PyTorch on ROCm provides mixed-precision and large-scale training
using `MIOpen <https://github.com/ROCm/MIOpen>`_ and
`RCCL <https://github.com/ROCm/rccl>`_ libraries.

ROCm support for PyTorch is upstreamed into the official PyTorch repository. Due
to independent compatibility considerations, this results in two distinct
release cycles for PyTorch on ROCm:

- ROCm PyTorch release:

  - Provides the latest version of ROCm but doesn't immediately support the latest stable PyTorch
    version.

  - Offers :ref:`Docker images <pytorch-docker-compat>` with ROCm and PyTorch
    pre-installed.

  - ROCm PyTorch repository: `<https://github.com/ROCm/pytorch>`__

  - See the :doc:`ROCm PyTorch installation guide <rocm-install-on-linux:install/3rd-party/pytorch-install>` to get started.

- Official PyTorch release:

  - Provides the latest stable version of PyTorch but doesn't immediately support the latest ROCm version.

  - Official PyTorch repository: `<https://github.com/pytorch/pytorch>`__

  - See the `Nightly and latest stable version installation guide <https://pytorch.org/get-started/locally/>`_
    or `Previous versions <https://pytorch.org/get-started/previous-versions/>`_ to get started.

The upstream PyTorch includes an automatic HIPification solution that automatically generates HIP
source code from the CUDA backend. This approach allows PyTorch to support ROCm without requiring
manual code modifications.

ROCm's development is aligned with the stable release of PyTorch while upstream PyTorch testing uses
the stable release of ROCm to maintain consistency.

.. _pytorch-docker-compat:

Docker image compatibility
================================================================================

.. |docker-icon| raw:: html

   <i class="fab fa-docker"></i>

AMD validates and publishes ready-made `PyTorch <https://hub.docker.com/r/rocm/pytorch>`_
images with ROCm backends on Docker Hub. The following Docker image tags and
associated inventories are validated for `ROCm 6.3.0 <https://repo.radeon.com/rocm/apt/6.3/>`_.
Click |docker-icon| to see the image on Docker Hub.

.. list-table:: PyTorch Docker image components
    :header-rows: 1
    :class: docker-image-compatibility

    * - Docker
      - PyTorch
      - Ubuntu
      - Python
      - Apex
      - torchvision
      - TensorBoard
      - MAGMA
      - UCX
      - OMPI
      - OFED

    * - .. raw:: html

           <a href="https://hub.docker.com/layers/rocm/pytorch/rocm6.3_ubuntu24.04_py3.12_pytorch_release_2.4.0/images/sha256-98ddf20333bd01ff749b8092b1190ee369a75d3b8c71c2fac80ffdcb1a98d529?context=explore"><i class="fab fa-docker fa-lg"></i></a>

      - `2.4.0 <https://github.com/ROCm/pytorch/tree/release/2.4>`_
      - 24.04
      - `3.12 <https://www.python.org/downloads/release/python-3128/>`_
      - `1.4.0 <https://github.com/ROCm/apex/tree/release/1.4.0>`_
      - `0.19.0 <https://github.com/pytorch/vision/tree/v0.19.0>`_
      - `2.13.0 <https://github.com/tensorflow/tensorboard/tree/2.13>`_
      - `master <https://bitbucket.org/icl/magma/src/master/>`_
      - `1.10.0 <https://github.com/openucx/ucx/tree/v1.10.0>`_
      - `4.0.7 <https://github.com/open-mpi/ompi/tree/v4.0.7>`_
      - `5.3-1.0.5.0 <https://content.mellanox.com/ofed/MLNX_OFED-5.3-1.0.5.0/MLNX_OFED_LINUX-5.3-1.0.5.0-ubuntu20.04-x86_64.tgz>`_

    * - .. raw:: html

           <a href="https://hub.docker.com/layers/rocm/pytorch/rocm6.3_ubuntu22.04_py3.10_pytorch_release_2.4.0/images/sha256-402c9b4f1a6b5a81c634a1932b56cbe01abb699cfcc7463d226276997c6cf8ea?context=explore"><i class="fab fa-docker fa-lg"></i></a>

      - `2.4.0 <https://github.com/ROCm/pytorch/tree/release/2.4>`_
      - 22.04
      - `3.10 <https://www.python.org/downloads/release/python-31016/>`_
      - `1.4.0 <https://github.com/ROCm/apex/tree/release/1.4.0>`_
      - `0.19.0 <https://github.com/pytorch/vision/tree/v0.19.0>`_
      - `2.13.0 <https://github.com/tensorflow/tensorboard/tree/2.13>`_
      - `master <https://bitbucket.org/icl/magma/src/master/>`_
      - `1.10.0 <https://github.com/openucx/ucx/tree/v1.10.0>`_
      - `4.0.7 <https://github.com/open-mpi/ompi/tree/v4.0.7>`_
      - `5.3-1.0.5.0 <https://content.mellanox.com/ofed/MLNX_OFED-5.3-1.0.5.0/MLNX_OFED_LINUX-5.3-1.0.5.0-ubuntu20.04-x86_64.tgz>`_

    * - .. raw:: html

           <a href="https://hub.docker.com/layers/rocm/pytorch/rocm6.3_ubuntu22.04_py3.9_pytorch_release_2.4.0/images/sha256-e0608b55d408c3bfe5c19fdd57a4ced3e0eb3a495b74c309980b60b156c526dd?context=explore"><i class="fab fa-docker fa-lg"></i></a>

      - `2.4.0 <https://github.com/ROCm/pytorch/tree/release/2.4>`_
      - 22.04
      - `3.9 <https://www.python.org/downloads/release/python-3918/>`_
      - `1.4.0 <https://github.com/ROCm/apex/tree/release/1.4.0>`_
      - `0.19.0 <https://github.com/pytorch/vision/tree/v0.19.0>`_
      - `2.13.0 <https://github.com/tensorflow/tensorboard/tree/2.13>`_
      - `master <https://bitbucket.org/icl/magma/src/master/>`_
      - `1.10.0 <https://github.com/openucx/ucx/tree/v1.10.0>`_
      - `4.0.7 <https://github.com/open-mpi/ompi/tree/v4.0.7>`_
      - `5.3-1.0.5.0 <https://content.mellanox.com/ofed/MLNX_OFED-5.3-1.0.5.0/MLNX_OFED_LINUX-5.3-1.0.5.0-ubuntu20.04-x86_64.tgz>`_

    * - .. raw:: html

           <a href="https://hub.docker.com/layers/rocm/pytorch/rocm6.3_ubuntu22.04_py3.10_pytorch_release_2.3.0/images/sha256-652cf25263d05b1de548222970aeb76e60b12de101de66751264709c0d0ff9d8?context=explore"><i class="fab fa-docker fa-lg"></i></a>

      - `2.3.0 <https://github.com/ROCm/pytorch/tree/release/2.3>`_
      - 22.04
      - `3.10 <https://www.python.org/downloads/release/python-31016/>`_
      - `1.3.0 <https://github.com/ROCm/apex/tree/release/1.3.0>`_
      - `0.18.0 <https://github.com/pytorch/vision/tree/v0.18.0>`_
      - `2.13.0 <https://github.com/tensorflow/tensorboard/tree/2.13>`_
      - `master <https://bitbucket.org/icl/magma/src/master/>`_
      - `1.14.1 <https://github.com/openucx/ucx/tree/v1.14.1>`_
      - `4.1.5 <https://github.com/open-mpi/ompi/tree/v4.1.5>`_
      - `5.3-1.0.5.0 <https://content.mellanox.com/ofed/MLNX_OFED-5.3-1.0.5.0/MLNX_OFED_LINUX-5.3-1.0.5.0-ubuntu20.04-x86_64.tgz>`_

    * - .. raw:: html

           <a href="https://hub.docker.com/layers/rocm/pytorch/rocm6.3_ubuntu22.04_py3.10_pytorch_release_2.2.1/images/sha256-051976f26beab8f9aa65d999e3ad546c027b39240a0cc3ee81b114a9024f2912?context=explore"><i class="fab fa-docker fa-lg"></i></a>

      - `2.2.1 <https://github.com/ROCm/pytorch/tree/release/2.2>`_
      - 22.04
      - `3.10 <https://www.python.org/downloads/release/python-31016/>`_
      - `1.2.0 <https://github.com/ROCm/apex/tree/release/1.2.0>`_
      - `0.17.1 <https://github.com/pytorch/vision/tree/v0.17.1>`_
      - `2.13.0 <https://github.com/tensorflow/tensorboard/tree/2.13>`_
      - `master <https://bitbucket.org/icl/magma/src/master/>`_
      - `1.14.1 <https://github.com/openucx/ucx/tree/v1.14.1>`_
      - `4.1.5 <https://github.com/open-mpi/ompi/tree/v4.1.5>`_
      - `5.3-1.0.5.0 <https://content.mellanox.com/ofed/MLNX_OFED-5.3-1.0.5.0/MLNX_OFED_LINUX-5.3-1.0.5.0-ubuntu20.04-x86_64.tgz>`_

    * - .. raw:: html

           <a href="https://hub.docker.com/layers/rocm/pytorch/rocm6.3_ubuntu20.04_py3.9_pytorch_release_2.2.1/images/sha256-88c839a364d109d3748c100385bfa100d28090d25118cc723fd0406390ab2f7e?context=explore"><i class="fab fa-docker fa-lg"></i></a>

      - `2.2.1 <https://github.com/ROCm/pytorch/tree/release/2.2>`_
      - 20.04
      - `3.9 <https://www.python.org/downloads/release/python-3921/>`_
      - `1.2.0 <https://github.com/ROCm/apex/tree/release/1.2.0>`_
      - `0.17.1 <https://github.com/pytorch/vision/tree/v0.17.1>`_
      - `2.13.0 <https://github.com/tensorflow/tensorboard/tree/2.13.0>`_
      - `master <https://bitbucket.org/icl/magma/src/master/>`_
      - `1.10.0 <https://github.com/openucx/ucx/tree/v1.10.0>`_
      - `4.0.3 <https://github.com/open-mpi/ompi/tree/v4.0.3>`_
      - `5.3-1.0.5.0 <https://content.mellanox.com/ofed/MLNX_OFED-5.3-1.0.5.0/MLNX_OFED_LINUX-5.3-1.0.5.0-ubuntu20.04-x86_64.tgz>`_

    * - .. raw:: html

           <a href="https://hub.docker.com/layers/rocm/pytorch/rocm6.3_ubuntu22.04_py3.9_pytorch_release_1.13.1/images/sha256-994424ed07a63113f79dd9aa72159124c00f5fbfe18127151e6658f7d0b6f821?context=explore"><i class="fab fa-docker fa-lg"></i></a>

      - `1.13.1 <https://github.com/ROCm/pytorch/tree/release/1.13>`_
      - 22.04
      - `3.9 <https://www.python.org/downloads/release/python-3921/>`_
      - `1.0.0 <https://github.com/ROCm/apex/tree/release/1.0.0>`_
      - `0.14.0 <https://github.com/pytorch/vision/tree/v0.14.0>`_
      - `2.18.0 <https://github.com/tensorflow/tensorboard/tree/2.18>`_
      - `master <https://bitbucket.org/icl/magma/src/master/>`_
      - `1.14.1 <https://github.com/openucx/ucx/tree/v1.14.1>`_
      - `4.1.5 <https://github.com/open-mpi/ompi/tree/v4.1.5>`_
      - `5.3-1.0.5.0 <https://content.mellanox.com/ofed/MLNX_OFED-5.3-1.0.5.0/MLNX_OFED_LINUX-5.3-1.0.5.0-ubuntu20.04-x86_64.tgz>`_

    * - .. raw:: html

           <a href="https://hub.docker.com/layers/rocm/pytorch/rocm6.3_ubuntu20.04_py3.9_pytorch_release_1.13.1/images/sha256-7b8139fe40a9aeb4bca3aecd15c22c1fa96e867d93479fa3a24fdeeeeafa1219?context=explore"><i class="fab fa-docker fa-lg"></i></a>

      - `1.13.1 <https://github.com/ROCm/pytorch/tree/release/1.13>`_
      - 20.04
      - `3.9 <https://www.python.org/downloads/release/python-3921/>`_
      - `1.0.0 <https://github.com/ROCm/apex/tree/release/1.0.0>`_
      - `0.14.0 <https://github.com/pytorch/vision/tree/v0.14.0>`_
      - `2.18.0 <https://github.com/tensorflow/tensorboard/tree/2.18>`_
      - `master <https://bitbucket.org/icl/magma/src/master/>`_
      - `1.10.0 <https://github.com/openucx/ucx/tree/v1.10.0>`_
      - `4.0.3 <https://github.com/open-mpi/ompi/tree/v4.0.3>`_
      - `5.3-1.0.5.0 <https://content.mellanox.com/ofed/MLNX_OFED-5.3-1.0.5.0/MLNX_OFED_LINUX-5.3-1.0.5.0-ubuntu20.04-x86_64.tgz>`_

Critical ROCm libraries for PyTorch
================================================================================

The functionality of PyTorch with ROCm is determined by its underlying library
dependencies. These critical ROCm components affect the capabilities,
performance, and feature set available to developers.

.. list-table::
    :header-rows: 1

    * - ROCm library
      - Version
      - Purpose
      - Used in
    * - `Composable Kernel <https://github.com/ROCm/composable_kernel>`_
      - 1.1.0
      - Enables faster execution of core operations like matrix multiplication
        (GEMM), convolutions and transformations.
      - Speeds up ``torch.permute``, ``torch.view``, ``torch.matmul``,
        ``torch.mm``, ``torch.bmm``, ``torch.nn.Conv2d``, ``torch.nn.Conv3d``
        and ``torch.nn.MultiheadAttention``. 
    * - `hipBLAS <https://github.com/ROCm/hipBLAS>`_
      - 2.3.0
      - Provides GPU-accelerated Basic Linear Algebra Subprograms (BLAS) for
        matrix and vector operations.
      - Supports operations like matrix multiplication, matrix-vector products,
        and tensor contractions. Utilized in both dense and batched linear
        algebra operations.
    * - `hipBLASLt <https://github.com/ROCm/hipBLASLt>`_
      - 0.10.0
      - hipBLASLt is an extension of the hipBLAS library, providing additional
        features like epilogues fused into the matrix multiplication kernel or
        use of integer tensor cores.
      - It accelerates operations like ``torch.matmul``, ``torch.mm``, and the
        matrix multiplications used in convolutional and linear layers.
    * - `hipCUB <https://github.com/ROCm/hipCUB>`_
      - 3.3.0
      - Provides a C++ template library for parallel algorithms for reduction,
        scan, sort and select.
      - Supports operations like ``torch.sum``, ``torch.cumsum``, ``torch.sort``
        and ``torch.topk``. Operations on sparse tensors or tensors with
        irregular shapes often involve scanning, sorting, and filtering, which
        hipCUB handles efficiently.
    * - `hipFFT <https://github.com/ROCm/hipFFT>`_
      - 1.0.17
      - Provides GPU-accelerated Fast Fourier Transform (FFT) operations.
      - Used in functions like the ``torch.fft`` module.
    * - `hipRAND <https://github.com/ROCm/hipRAND>`_
      - 2.11.0
      - Provides fast random number generation for GPUs.
      - The ``torch.rand``, ``torch.randn`` and stochastic layers like 
        ``torch.nn.Dropout``.
    * - `hipSOLVER <https://github.com/ROCm/hipSOLVER>`_
      - 2.3.0
      - Provides GPU-accelerated solvers for linear systems, eigenvalues, and
        singular value decompositions (SVD).
      - Supports functions like ``torch.linalg.solve``,
        ``torch.linalg.eig``, and ``torch.linalg.svd``.
    * - `hipSPARSE <https://github.com/ROCm/hipSPARSE>`_
      - 3.1.2
      - Accelerates operations on sparse matrices, such as sparse matrix-vector
        or matrix-matrix products.
      - Sparse tensor operations ``torch.sparse``.
    * - `hipSPARSELt <https://github.com/ROCm/hipSPARSELt>`_
      - 0.2.2
      - Accelerates operations on sparse matrices, such as sparse matrix-vector
        or matrix-matrix products.
      - Sparse tensor operations ``torch.sparse``.
    * - `hipTensor <https://github.com/ROCm/hipTensor>`_
      - 1.4.0
      - Optimizes for high-performance tensor operations, such as contractions.
      - Accelerates tensor algebra, especially in deep learning and scientific
        computing.
    * - `MIOpen <https://github.com/ROCm/MIOpen>`_
      - 3.3.0
      - Optimizes deep learning primitives such as convolutions, pooling,
        normalization, and activation functions.
      - Speeds up convolutional neural networks (CNNs), recurrent neural
        networks (RNNs), and other layers. Used in operations like
        ``torch.nn.Conv2d``, ``torch.nn.ReLU``, and ``torch.nn.LSTM``.
    * - `MIGraphX <https://github.com/ROCm/AMDMIGraphX>`_
      - 2.11.0
      - Adds graph-level optimizations, ONNX models and mixed precision support
        and enable Ahead-of-Time (AOT) Compilation.
      - Speeds up inference models and executes ONNX models for
        compatibility with other frameworks.
        ``torch.nn.Conv2d``, ``torch.nn.ReLU``, and ``torch.nn.LSTM``.
    * - `MIVisionX <https://github.com/ROCm/MIVisionX>`_
      - 3.1.0
      - Optimizes acceleration for computer vision and AI workloads like
        preprocessing, augmentation, and inferencing.
      - Faster data preprocessing and augmentation pipelines for datasets like
        ImageNet or COCO and easy to integrate into PyTorch's ``torch.utils.data``
        and ``torchvision`` workflows.
    * - `rocAL <https://github.com/ROCm/rocAL>`_
      - 2.1.0
      - Accelerates the data pipeline by offloading intensive preprocessing and
        augmentation tasks. rocAL is part of MIVisionX.
      - Easy to integrate into PyTorch's ``torch.utils.data`` and
        ``torchvision`` data load workloads.
    * - `RCCL <https://github.com/ROCm/rccl>`_
      - 2.21.5
      - Optimizes for multi-GPU communication for operations like AllReduce and
        Broadcast.
      - Distributed data parallel training (``torch.nn.parallel.DistributedDataParallel``).
        Handles communication in multi-GPU setups.
    * - `rocDecode <https://github.com/ROCm/rocDecode>`_
      - 0.8.0
      - Provides hardware-accelerated data decoding capabilities, particularly
        for image, video, and other dataset formats.
      - Can be integrated in ``torch.utils.data``, ``torchvision.transforms``
        and ``torch.distributed``.
    * - `rocJPEG <https://github.com/ROCm/rocJPEG>`_
      - 0.6.0
      - Provides hardware-accelerated JPEG image decoding and encoding.
      - GPU accelerated ``torchvision.io.decode_jpeg`` and
        ``torchvision.io.encode_jpeg`` and can be integrated in
        ``torch.utils.data`` and ``torchvision``.
    * - `RPP <https://github.com/ROCm/RPP>`_
      - 1.9.1
      - Speeds up data augmentation, transformation, and other preprocessing steps.
      - Easy to integrate into PyTorch's ``torch.utils.data`` and
        ``torchvision`` data load workloads.
    * - `rocThrust <https://github.com/ROCm/rocThrust>`_
      - 3.3.0
      - Provides a C++ template library for parallel algorithms like sorting,
        reduction, and scanning.
      - Utilized in backend operations for tensor computations requiring
        parallel processing.
    * - `rocWMMA <https://github.com/ROCm/rocWMMA>`_
      - 1.6.0
      - Accelerates warp-level matrix-multiply and matrix-accumulate to speed up matrix
        multiplication (GEMM) and accumulation operations with mixed precision
        support.
      - Linear layers (``torch.nn.Linear``), convolutional layers
        (``torch.nn.Conv2d``), attention layers, general tensor operations that
        involve matrix products, such as ``torch.matmul``, ``torch.bmm``, and
        more.

Supported and unsupported features
================================================================================

The following section maps GPU-accelerated PyTorch features to their supported
ROCm and PyTorch versions.

torch
--------------------------------------------------------------------------------

`torch <https://pytorch.org/docs/stable/index.html>`_ is the central module of
PyTorch, providing data structures for multi-dimensional tensors and
implementing mathematical operations on them. It also includes utilities for
efficient serialization of tensors and arbitrary data types, along with various
other tools.

Tensor data types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The data type of a tensor is specified using the ``dtype`` attribute or argument, and PyTorch supports a wide range of data types for different use cases.

The following table lists `torch.Tensor <https://pytorch.org/docs/stable/tensors.html>`_'s single data types:

.. list-table::
    :header-rows: 1

    * - Data type
      - Description
      - Since PyTorch
      - Since ROCm
    * - ``torch.float8_e4m3fn``
      - 8-bit floating point, e4m3
      - 2.3
      - 5.5
    * - ``torch.float8_e5m2``
      - 8-bit floating point, e5m2
      - 2.3
      - 5.5
    * - ``torch.float16`` or ``torch.half``
      - 16-bit floating point
      - 0.1.6
      - 2.0
    * - ``torch.bfloat16``
      - 16-bit floating point
      - 1.6
      - 2.6
    * - ``torch.float32`` or ``torch.float``
      - 32-bit floating point
      - 0.1.12_2
      - 2.0
    * - ``torch.float64`` or ``torch.double``
      - 64-bit floating point
      - 0.1.12_2
      - 2.0
    * - ``torch.complex32`` or ``torch.chalf``
      - PyTorch provides native support for 32-bit complex numbers
      - 1.6
      - 2.0
    * - ``torch.complex64`` or ``torch.cfloat``
      - PyTorch provides native support for 64-bit complex numbers
      - 1.6
      - 2.0
    * - ``torch.complex128`` or ``torch.cdouble``
      - PyTorch provides native support for 128-bit complex numbers
      - 1.6
      - 2.0
    * - ``torch.uint8``
      - 8-bit integer (unsigned)
      - 0.1.12_2
      - 2.0
    * - ``torch.uint16``
      - 16-bit integer (unsigned)
      - 2.3
      - Not natively supported
    * - ``torch.uint32``
      - 32-bit integer (unsigned)
      - 2.3
      - Not natively supported
    * - ``torch.uint64``
      - 32-bit integer (unsigned)
      - 2.3
      - Not natively supported
    * - ``torch.int8``
      - 8-bit integer (signed)
      - 1.12
      - 5.0
    * - ``torch.int16`` or ``torch.short``
      - 16-bit integer (signed)
      - 0.1.12_2
      - 2.0
    * - ``torch.int32`` or ``torch.int``
      - 32-bit integer (signed)
      - 0.1.12_2
      - 2.0
    * - ``torch.int64`` or ``torch.long``
      - 64-bit integer (signed)
      - 0.1.12_2
      - 2.0
    * - ``torch.bool``
      - Boolean
      - 1.2
      - 2.0
    * - ``torch.quint8``
      - Quantized 8-bit integer (unsigned)
      - 1.8
      - 5.0
    * - ``torch.qint8``
      - Quantized 8-bit integer (signed)
      - 1.8
      - 5.0
    * - ``torch.qint32``
      - Quantized 32-bit integer (signed)
      - 1.8
      - 5.0
    * - ``torch.quint4x2``
      - Quantized 4-bit integer (unsigned)
      - 1.8
      - 5.0

.. note::

  Unsigned types aside from ``uint8`` are currently only have limited support in
  eager mode (they primarily exist to assist usage with ``torch.compile``).

  The :doc:`ROCm precision support page <rocm:reference/precision-support>`
  collected the native HW support of different data types.

torch.cuda
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``torch.cuda`` in PyTorch is a module that provides utilities and functions for
managing and utilizing AMD and NVIDIA GPUs. It enables GPU-accelerated
computations, memory management, and efficient execution of tensor operations,
leveraging ROCm and CUDA as the underlying frameworks.

.. list-table::
    :header-rows: 1

    * - Feature
      - Description
      - Since PyTorch
      - Since ROCm
    * - Device management
      - Utilities for managing and interacting with GPUs.
      - 0.4.0
      - 3.8
    * - Tensor operations on GPU
      - Performs tensor operations such as addition and matrix multiplications on
        the GPU.
      - 0.4.0
      - 3.8
    * - Streams and events
      - Streams allow overlapping computation and communication for optimized
        performance. Events enable synchronization.
      - 1.6.0
      - 3.8
    * - Memory management
      - Functions to manage and inspect memory usage like
        ``torch.cuda.memory_allocated()``, ``torch.cuda.max_memory_allocated()``,
        ``torch.cuda.memory_reserved()`` and ``torch.cuda.empty_cache()``.
      - 0.3.0
      - 1.9.2
    * - Running process lists of memory management
      - Returns a human-readable printout of the running processes and their GPU
        memory use for a given device with functions like
        ``torch.cuda.memory_stats()`` and ``torch.cuda.memory_summary()``.
      - 1.8.0
      - 4.0
    * - Communication collectives
      - Set of APIs that enable efficient communication between multiple GPUs,
        allowing for distributed computing and data parallelism.
      - 1.9.0
      - 5.0
    * - ``torch.cuda.CUDAGraph``
      - Graphs capture sequences of GPU operations to minimize kernel launch
        overhead and improve performance.
      - 1.10.0
      - 5.3
    * - TunableOp
      - A mechanism that allows certain operations to be more flexible and
        optimized for performance. It enables automatic tuning of kernel
        configurations and other settings to achieve the best possible
        performance based on the specific hardware (GPU) and workload.
      - 2.0
      - 5.4
    * - NVIDIA Tools Extension (NVTX)
      - Integration with NVTX for profiling and debugging GPU performance using
        NVIDIA's Nsight tools.
      - 1.8.0
      - ❌
    * - Lazy loading NVRTC
      - Delays JIT compilation with NVRTC until the code is explicitly needed.
      - 1.13.0
      - ❌
    * - Jiterator (beta)
      - Jiterator allows asynchronous data streaming into computation streams
        during training loops.
      - 1.13.0
      - 5.2

.. Need to validate and extend.

torch.backends.cuda
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``torch.backends.cuda`` is a PyTorch module that provides configuration options
and flags to control the behavior of CUDA or ROCm operations. It is part of the
PyTorch backend configuration system, which allows users to fine-tune how
PyTorch interacts with the CUDA or ROCm environment.

.. list-table::
    :header-rows: 1

    * - Feature
      - Description
      - Since PyTorch
      - Since ROCm
    * - ``cufft_plan_cache``
      - Manages caching of GPU FFT plans to optimize repeated FFT computations.
      - 1.7.0
      - 5.0
    * - ``matmul.allow_tf32``
      - Enables or disables the use of TensorFloat-32 (TF32) precision for
        faster matrix multiplications on GPUs with Tensor Cores.
      - 1.10.0
      - ❌
    * - ``matmul.allow_fp16_reduced_precision_reduction``
      - Reduced precision reductions (e.g., with fp16 accumulation type) are
        allowed with fp16 GEMMs.
      - 2.0
      - ❌
    * - ``matmul.allow_bf16_reduced_precision_reduction``
      - Reduced precision reductions are allowed with bf16 GEMMs.
      - 2.0
      - ❌
    * - ``enable_cudnn_sdp``
      - Globally enables cuDNN SDPA's kernels within SDPA.
      - 2.0
      - ❌
    * - ``enable_flash_sdp``
      - Globally enables or disables FlashAttention for SDPA.
      - 2.1
      - ❌
    * - ``enable_mem_efficient_sdp``
      - Globally enables or disables Memory-Efficient Attention for SDPA.
      - 2.1
      - ❌
    * - ``enable_math_sdp``
      - Globally enables or disables the PyTorch C++ implementation within SDPA.
      - 2.1
      - ❌

.. Need to validate and extend.

torch.backends.cudnn
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Supported ``torch`` options include:

.. list-table::
    :header-rows: 1

    * - Option
      - Description
      - Since PyTorch
      - Since ROCm
    * - ``allow_tf32``
      - TensorFloat-32 tensor cores may be used in cuDNN convolutions on NVIDIA
        Ampere or newer GPUs.
      - 1.12.0
      - ❌
    * - ``deterministic``
      - A bool that, if True, causes cuDNN to only use deterministic
        convolution algorithms.
      - 1.12.0
      - 6.0

Automatic mixed precision: torch.amp
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PyTorch that automates the process of using both 16-bit (half-precision,
float16) and 32-bit (single-precision, float32) floating-point types in model
training and inference.

.. list-table::
    :header-rows: 1

    * - Feature
      - Description
      - Since PyTorch
      - Since ROCm
    * - Autocasting
      - Instances of autocast serve as context managers or decorators that allow
        regions of your script to run in mixed precision.
      - 1.9
      - 2.5
    * - Gradient scaling
      - To prevent underflow, “gradient scaling” multiplies the network’s
        loss(es) by a scale factor and invokes a backward pass on the scaled
        loss(es). Gradients flowing backward through the network are then
        scaled by the same factor. In other words, gradient values have a
        larger magnitude, so they don’t flush to zero.
      - 1.9
      - 2.5
    * - CUDA op-specific behavior
      - These ops always go through autocasting whether they are invoked as part
        of a ``torch.nn.Module``, as a function, or as a ``torch.Tensor`` method. If
        functions are exposed in multiple namespaces, they go through
        autocasting regardless of the namespace.
      - 1.9
      - 2.5

Distributed library features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The PyTorch distributed library includes a collective of parallelism modules, a
communications layer, and infrastructure for launching and debugging large
training jobs. See :ref:`rocm-for-ai-pytorch-distributed` for more information.

The Distributed Library feature in PyTorch provides tools and APIs for building
and running distributed machine learning workflows. It allows training models
across multiple processes, GPUs, or nodes in a cluster, enabling efficient use
of computational resources and scalability for large-scale tasks.

.. list-table::
    :header-rows: 1

    * - Feature
      - Description
      - Since PyTorch
      - Since ROCm
    * - TensorPipe
      - A point-to-point communication library integrated into
        PyTorch for distributed training. It is designed to handle tensor data
        transfers efficiently between different processes or devices, including
        those on separate machines.
      - 1.8
      - 5.4
    * - Gloo
      - Designed for multi-machine and multi-GPU setups, enabling
        efficient communication and synchronization between processes. Gloo is
        one of the default backends for PyTorch's Distributed Data Parallel
        (DDP) and RPC frameworks, alongside other backends like NCCL and MPI.
      - 1.0
      - 2.0

torch.compiler
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
    :header-rows: 1

    * - Feature
      - Description
      - Since PyTorch
      - Since ROCm
    * - ``torch.compiler`` (AOT Autograd)
      - Autograd captures not only the user-level code, but also backpropagation,
        which results in capturing the backwards pass “ahead-of-time”. This
        enables acceleration of both forwards and backwards pass using
        ``TorchInductor``.
      - 2.0
      - 5.3
    * - ``torch.compiler`` (TorchInductor)
      - The default ``torch.compile`` deep learning compiler that generates fast
        code for multiple accelerators and backends. You need to use a backend
        compiler to make speedups through ``torch.compile`` possible. For AMD,
        NVIDIA, and Intel GPUs, it leverages OpenAI Triton as the key building block.
      - 2.0
      - 5.3

torchaudio
--------------------------------------------------------------------------------

The `torchaudio <https://pytorch.org/audio/stable/index.html>`_ library provides
utilities for processing audio data in PyTorch, such as audio loading,
transformations, and feature extraction.

To ensure GPU-acceleration with ``torchaudio.transforms``, you need to move audio
data (waveform tensor) explicitly to GPU using ``.to('cuda')``.

The following ``torchaudio`` features are GPU-accelerated.

.. list-table::
    :header-rows: 1

    * - Feature
      - Description
      - Since torchaudio version
      - Since ROCm
    * - ``torchaudio.transforms.Spectrogram``
      - Generates spectrogram of an input waveform using STFT.
      - 0.6.0
      - 4.5
    * - ``torchaudio.transforms.MelSpectrogram``
      - Generates the mel-scale spectrogram of raw audio signals.
      - 0.9.0
      - 4.5
    * - ``torchaudio.transforms.MFCC``
      - Extract of MFCC features.
      - 0.9.0
      - 4.5
    * - ``torchaudio.transforms.Resample``
      - Resamples a signal from one frequency to another.
      - 0.9.0
      - 4.5

torchvision
--------------------------------------------------------------------------------

The `torchvision <https://pytorch.org/vision/stable/index.html>`_ library
provide datasets, model architectures, and common image transformations for
computer vision.

The following ``torchvision`` features are GPU-accelerated.

.. list-table::
    :header-rows: 1

    * - Feature
      - Description
      - Since torchvision version
      - Since ROCm
    * - ``torchvision.transforms.functional``
      - Provides GPU-compatible transformations for image preprocessing like
        resize, normalize, rotate and crop.
      - 0.2.0
      - 4.0
    * - ``torchvision.ops``
      - GPU-accelerated operations for object detection and segmentation tasks.
        ``torchvision.ops.roi_align``, ``torchvision.ops.nms`` and
        ``box_convert``.
      - 0.6.0
      - 3.3
    * - ``torchvision.models`` with ``.to('cuda')``
      - ``torchvision`` provides several pre-trained models (ResNet, Faster
        R-CNN, Mask R-CNN, ...) that can run on CUDA for faster inference and
        training.
      - 0.1.6
      - 2.x
    * - ``torchvision.io``
      - Enables video decoding and frame extraction using GPU acceleration with NVIDIA’s
        NVDEC and nvJPEG (rocJPEG) on CUDA-enabled GPUs.
      - 0.4.0
      - 6.3

torchtext
--------------------------------------------------------------------------------

The `torchtext <https://pytorch.org/text/stable/index.html>`_ library provides
utilities for processing and working with text data in PyTorch, including
tokenization, vocabulary management, and text embeddings. torchtext supports
preprocessing pipelines and integration with PyTorch models, simplifying the
implementation of natural language processing (NLP) tasks.

To leverage GPU acceleration in torchtext, you need to move tensors
explicitly to the GPU using ``.to('cuda')``.

* torchtext does not implement its own kernels. ROCm support is enabled by linking against ROCm libraries.

* Only official release exists.

torchtune
--------------------------------------------------------------------------------

The `torchtune <https://pytorch.org/torchtune/stable/index.html>`_ library for
authoring, fine-tuning and experimenting with LLMs.

* Usage: It works out-of-the-box, enabling developers to fine-tune ROCm PyTorch solutions.

* Only official release exists.

torchserve
--------------------------------------------------------------------------------

The `torchserve <https://pytorch.org/torchserve/>`_ is a PyTorch domain library
for common sparsity and parallelism primitives needed for large-scale recommender
systems.

* torchtext does not implement its own kernels. ROCm support is enabled by linking against ROCm libraries.

* Only official release exists.

torchrec
--------------------------------------------------------------------------------

The `torchrec <https://pytorch.org/torchrec/>`_ is a PyTorch domain library for
common sparsity and parallelism primitives needed for large-scale recommender
systems.

* torchrec does not implement its own kernels. ROCm support is enabled by linking against ROCm libraries.

* Only official release exists.

Unsupported PyTorch features
----------------------------

The following are GPU-accelerated PyTorch features not currently supported by ROCm.

.. list-table::
    :widths: 30, 60, 10
    :header-rows: 1

    * - Feature
      - Description
      - Since PyTorch
    * - APEX batch norm
      - Use APEX batch norm instead of PyTorch batch norm.
      - 1.6.0
    * - ``torch.backends.cuda`` / ``matmul.allow_tf32``
      - A bool that controls whether TensorFloat-32 tensor cores may be used in
        matrix multiplications.
      - 1.7
    * - ``torch.cuda`` / NVIDIA Tools Extension (NVTX)
      - Integration with NVTX for profiling and debugging GPU performance using
        NVIDIA's Nsight tools.
      - 1.7.0
    * - ``torch.cuda`` / Lazy loading NVRTC
      - Delays JIT compilation with NVRTC until the code is explicitly needed.
      - 1.8.0
    * - ``torch-tensorrt``
      - Integrate TensorRT library for optimizing and deploying PyTorch models.
        ROCm does not have equialent library for TensorRT.
      - 1.9.0
    * - ``torch.backends`` / ``cudnn.allow_tf32``
      - TensorFloat-32 tensor cores may be used in cuDNN convolutions.
      - 1.10.0
    * - ``torch.backends.cuda`` / ``matmul.allow_fp16_reduced_precision_reduction``
      - Reduced precision reductions with fp16 accumulation type are
        allowed with fp16 GEMMs.
      - 2.0
    * - ``torch.backends.cuda`` / ``matmul.allow_bf16_reduced_precision_reduction``
      - Reduced precision reductions are allowed with bf16 GEMMs.
      - 2.0
    * - ``torch.nn.functional`` / ``scaled_dot_product_attention`` 
      - Flash attention backend for SDPA to accelerate attention computation in
        transformer-based models.
      - 2.0
    * - ``torch.backends.cuda`` / ``enable_cudnn_sdp``
      - Globally enables cuDNN SDPA's kernels within SDPA.
      - 2.0
    * - ``torch.backends.cuda`` / ``enable_flash_sdp``
      - Globally enables or disables FlashAttention for SDPA.
      - 2.1
    * - ``torch.backends.cuda`` / ``enable_mem_efficient_sdp``
      - Globally enables or disables Memory-Efficient Attention for SDPA.
      - 2.1
    * - ``torch.backends.cuda`` / ``enable_math_sdp``
      - Globally enables or disables the PyTorch C++ implementation within SDPA.
      - 2.1
    * - Dynamic parallelism
      - PyTorch itself does not directly expose dynamic parallelism as a core
        feature. Dynamic parallelism allow GPU threads to launch additional
        threads which can be reached using custom operations via the
        ``torch.utils.cpp_extension`` module.
      - Not a core feature
    * - Unified memory support in PyTorch
      - Unified Memory is not directly exposed in PyTorch's core API, it can be
        utilized effectively through custom CUDA extensions or advanced
        workflows.
      - Not a core feature

Use cases and recommendations
================================================================================

* :doc:`Using ROCm for AI: training a model </how-to/rocm-for-ai/train-a-model>` provides
  guidance on how to leverage the ROCm platform for training AI models. It covers the steps, tools, and best practices
  for optimizing training workflows on AMD GPUs using PyTorch features.

* :doc:`Single-GPU fine-tuning and inference </how-to/llm-fine-tuning-optimization/single-gpu-fine-tuning-and-inference>`
  describes and demonstrates how to use the ROCm platform for the fine-tuning and inference of
  machine learning models, particularly large language models (LLMs), on systems with a single AMD
  Instinct MI300X accelerator. This page provides a detailed guide for setting up, optimizing, and
  executing fine-tuning and inference workflows in such environments.

* :doc:`Multi-GPU fine-tuning and inference optimization </how-to/llm-fine-tuning-optimization/multi-gpu-fine-tuning-and-inference>`
  describes and demonstrates the fine-tuning and inference of machine learning models on systems
  with multi MI300X accelerators.

* The :doc:`Instinct MI300X workload optimization guide </how-to/tuning-guides/mi300x/workload>` provides detailed
  guidance on optimizing workloads for the AMD Instinct MI300X accelerator using ROCm. This guide is aimed at helping
  users achieve optimal performance for deep learning and other high-performance computing tasks on the MI300X
  accelerator.

* The :doc:`Inception with PyTorch documentation </conceptual/ai-pytorch-inception>`
  describes how PyTorch integrates with ROCm for AI workloads It outlines the use of PyTorch on the ROCm platform and
  focuses on how to efficiently leverage AMD GPU hardware for training and inference tasks in AI applications.

For more use cases and recommendations, see `ROCm PyTorch blog posts <https://rocm.blogs.amd.com/blog/tag/pytorch.html>`_.
