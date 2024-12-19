.. meta::
    :description: TensorFlow compatibility
    :keywords: GPU, TensorFlow compatibility

*******************************************************************************
TensorFlow compatibility
*******************************************************************************

`TensorFlow <https://www.tensorflow.org/>`_ is an open-source library for
solving machine learning, deep learning, and AI problems. It can solve many
problems across different sectors and industries, but primarily focuses on
neural network training and inference. It is one of the most popular and
in-demand frameworks and is very active in open-source contribution and
development.

ROCm support for TensorFlow is upstreamed into the official TensorFlow
repository. Due to independent compatibility considerations, this results in
two distinct release cycles for TensorFlow on ROCm:

- ROCm TensorFlow release:

  - Provides the latest version of ROCm but doesn't immediately support the
    latest stable TensorFlow version.

  - Offers `Docker images <https://hub.docker.com/r/rocm/tensorflow>`_ with
    ROCm and TensorFlow pre-installed.

  - ROCm TensorFlow repository: `<https://github.com/ROCm/tensorflow-upstream>`_

  - See the :doc:`ROCm TensorFlow installation guide <rocm-install-on-linux:install/3rd-party/tensorflow-install>`
    to get started.

- Official TensorFlow release:

  - Official TensorFlow repository: `<https://github.com/tensorflow/tensorflow>`_

  - See the `Previous versions <https://www.tensorflow.org/versions>`_.

.. note::

  The official Tensorflow documentation does not mentioning the ROCm support and
  only the ROCm documentation provide installation guide.

Docker image compatibility
===============================================================================

AMD validates and publishes ready-made `Tensorflow
<https://hub.docker.com/r/rocm/tensorflow>`_ images with ROCm backends on
Docker Hub. The following Docker image tags and associated inventories are
validated for `ROCm 6.3.0 <https://repo.radeon.com/rocm/apt/6.3/>`_.

.. list-table:: TensorFlow docker image components
    :header-rows: 1

    * - Docker image
      - TensorFlow
      - Dev
      - Python
      - TensorBoard 

    * - .. raw:: html

           <a href="https://hub.docker.com/layers/rocm/tensorflow/rocm6.3-py3.10-tf2.15.0-runtime/images/sha256-37e0ab694ac0c65afbf34e32e115122d1c2af37e8095740ac1c951e48faed4e7?context=explore"><i class="fab fa-docker fa-lg"></i></a>

      - `tensorflow-rocm 2.15.1 <https://repo.radeon.com/rocm/manylinux/rocm-rel-6.3/tensorflow_rocm-2.15.1-cp310-cp310-manylinux_2_28_x86_64.whl>`_
      - runtime
      - `Python 3.10 <https://www.python.org/downloads/release/python-31016/>`_
      - `TensorBoard 2.15.2 <https://github.com/tensorflow/tensorboard/tree/2.15.2>`_

    * - .. raw:: html

           <a href="https://hub.docker.com/layers/rocm/tensorflow/rocm6.3-py3.10-tf2.15.0-dev/images/sha256-f1c633cbcebb9e34660c06bff5aa22dee82a9e2a4919ba923deb32216edce5db?context=explore"><i class="fab fa-docker fa-lg"></i></a>

      - `tensorflow-rocm 2.15.1 <https://repo.radeon.com/rocm/manylinux/rocm-rel-6.3/tensorflow_rocm-2.15.1-cp310-cp310-manylinux_2_28_x86_64.whl>`_
      - dev
      - `Python 3.10 <https://www.python.org/downloads/release/python-31016/>`_
      - `TensorBoard 2.15.2 <https://github.com/tensorflow/tensorboard/tree/2.15.2>`_

ROCm critical libraries for Tensorflow
===============================================================================

TensorFlow depends on multiple components, and the supported features of those
components can affect the TensorFlow ROCm supported feature set. The version
mentioned refers to the first TensorFlow version where the ROCm library was
introduced as a dependency.

.. list-table::
    :widths: 25, 10, 35, 30
    :header-rows: 1

    * - ROCm library
      - Version
      - Purpose
      - Used in
    * - `hipBLAS <https://github.com/ROCm/hipBLAS>`_
      - 2.3.0
      - Provides GPU-accelerated Basic Linear Algebra Subprograms (BLAS) for
        matrix and vector operations.
      - Accelerates operations like ``tf.matmul``, ``tf.linalg.matmul``, and
        other matrix multiplications commonly used in neural network layers.
    * - `hipBLASLt <https://github.com/ROCm/hipBLASLt>`_
      - 0.10.0
      - Extends `hipBLAS` with additional optimizations like fused kernels and
        integer tensor cores.
      - Optimizes matrix multiplications and linear algebra operations used in
        layers like dense, convolutional, and RNNs in TensorFlow.
    * - `hipCUB <https://github.com/ROCm/hipCUB>`_
      - 3.3.0
      - Provides a C++ template library for parallel algorithms for reduction,
        scan, sort and select.
      - Supports operations like ``tf.reduce_sum``, ``tf.cumsum``, ``tf.sort``
        and other tensor operations in TensorFlow, especially those involving
        scanning, sorting, and filtering.
    * - `hipFFT <https://github.com/ROCm/hipFFT>`_
      - 1.0.17
      - Accelerates Fast Fourier Transforms (FFT) for signal processing tasks.
      - Used for operations like signal processing, image filtering, and
        certain types of neural networks requiring FFT-based transformations.
    * - `hipSOLVER <https://github.com/ROCm/hipSOLVER>`_
      - 2.3.0
      - Provides GPU-accelerated direct linear solvers for dense and sparse
        systems.
      - Optimizes linear algebra functions such as solving systems of linear
        equations, often used in optimization and training tasks.
    * - `hipSPARSE <https://github.com/ROCm/hipSPARSE>`_
      - 3.1.2
      - Optimizes sparse matrix operations for efficient computations on sparse
        data.
      - Accelerates sparse matrix operations in models with sparse weight
        matrices or activations, commonly used in neural networks.
    * - `MIOpen <https://github.com/ROCm/MIOpen>`_
      - 3.3.0
      - Provides optimized deep learning primitives such as convolutions,
        pooling,
        normalization, and activation functions.
      - Speeds up convolutional neural networks (CNNs) and other layers. Used
        in TensorFlow for layers like ``tf.nn.conv2d``, ``tf.nn.relu``, and
        ``tf.nn.lstm_cell``.
    * - `RCCL <https://github.com/ROCm/rccl>`_
      - 2.21.5
      - Optimizes for multi-GPU communication for operations like AllReduce and
        Broadcast.
      - Distributed data parallel training (``tf.distribute.MirroredStrategy``).
        Handles communication in multi-GPU setups.

Supported and unsupported features
===============================================================================

The data type of a tensor is specified using the ``dtype`` attribute or
argument, and TensorFlow supports a wide range of data types for different use
cases.

The single data types of `tf.dtypes <https://www.tensorflow.org/api_docs/python/tf/dtypes>`_

.. list-table::
    :header-rows: 1

    * - Data type
      - Description
      - Since TensorFlow
      - Since ROCm
    * - bfloat16
      - 16-bit bfloat (brain floating point).
      - 1.0.0
      - [Insert ROCm Version]
    * - bool
      - Boolean.
      - 1.0.0
      - [Insert ROCm Version]
    * - complex128
      - 128-bit complex.
      - 1.0.0
      - [Insert ROCm Version]
    * - complex64
      - 64-bit complex.
      - 1.0.0
      - [Insert ROCm Version]
    * - double
      - 64-bit (double precision) floating-point.
      - 1.0.0
      - [Insert ROCm Version]
    * - float16
      - 16-bit (half precision) floating-point.
      - 1.0.0
      - [Insert ROCm Version]
    * - float32
      - 32-bit (single precision) floating-point.
      - 1.0.0
      - [Insert ROCm Version]
    * - float64
      - 64-bit (double precision) floating-point.
      - 1.0.0
      - [Insert ROCm Version]
    * - half
      - 16-bit (half precision) floating-point.
      - 2.0.0
      - [Insert ROCm Version]
    * - int16
      - Signed 16-bit integer.
      - 1.0.0
      - [Insert ROCm Version]
    * - int32
      - Signed 32-bit integer.
      - 1.0.0
      - [Insert ROCm Version]
    * - int64
      - Signed 64-bit integer.
      - 1.0.0
      - [Insert ROCm Version]
    * - int8
      - Signed 8-bit integer.
      - 1.0.0
      - [Insert ROCm Version]
    * - qint16
      - Signed quantized 16-bit integer.
      - 1.0.0
      - [Insert ROCm Version]
    * - qint32
      - Signed quantized 32-bit integer.
      - 1.0.0
      - [Insert ROCm Version]
    * - qint8
      - Signed quantized 8-bit integer.
      - 1.0.0
      - [Insert ROCm Version]
    * - quint16
      - Unsigned quantized 16-bit integer.
      - 1.0.0
      - [Insert ROCm Version]
    * - quint8
      - Unsigned quantized 8-bit integer.
      - 1.0.0
      - [Insert ROCm Version]
    * - resource
      - Handle to a mutable, dynamically allocated resource.
      - 1.0.0
      - [Insert ROCm Version]
    * - string
      - Variable-length string, represented as byte array.
      - 1.0.0
      - [Insert ROCm Version]
    * - uint16
      - Unsigned 16-bit (word) integer.
      - 1.0.0
      - [Insert ROCm Version]
    * - uint32
      - Unsigned 32-bit (dword) integer.
      - 1.5.0
      - [Insert ROCm Version]
    * - uint64
      - Unsigned 64-bit (qword) integer.
      - 1.5.0
      - [Insert ROCm Version]
    * - uint8
      - Unsigned 8-bit (byte) integer.
      - 1.0.0
      - [Insert ROCm Version]
    * - variant
      - Data of arbitrary type (known at runtime).
      - 1.4.0
      - [Insert ROCm Version]

Unsupported Tensorflow features
===============================================================================

The following are GPU-acclerated JAX features not currently supported by ROCm.

.. list-table::
    :header-rows: 1

    * - Data Type
      - Description
      - Since PyTorch
    * - Mixed Precision with TF32
      - Mixed precision with TF32 is used for matrix multiplications,
        convolutions, and other linear algebra operations, particularly in
        deep learning workloads like CNNs and transformers.
      - 
    * - RNN support
      - Currently only LSTM with double bias is supported with float32 input
        and weight.
      - 
    * - XLA int4 support
      - 4-bit integer (int4) precision in the XLA compiler.
      - 
    * - Graph support
      - Does not expose Graphs as a standalone feature, its reliance on XLA for
        computation allows Graph solutions to be used internally for GPU
        workloads.
      - 
    * - Semi-structured sparsity
      - Semi-structured sparsity typically involves setting values to zero in
        certain parts of a tensor or matrix according to patterns that are
        either predefined or learned.
      - 

Use cases and recommendations
===============================================================================

* The `Training a Neural Collaborative Filtering (NCF) Recommender on an AMD GPU
  <https://rocm.blogs.amd.com/artificial-intelligence/ncf/README.html>`_ blog post
  discusses training an NCF recommender system using Tensorflow. It explains how
  NCF improves traditional collaborative filtering methods by leveraging neural
  networks to model non-linear user-item interactions. The post outlines the
  implementation using the recommenders library, focusing on the use of implicit
  data (e.g., user interactions like viewing or purchasing) and how it addresses
  challenges like the lack of negative values. 


* The `Creating a PyTorch/TensorFlow code environment on AMD GPUs
  <https://rocm.blogs.amd.com/software-tools-optimization/pytorch-tensorflow-env/README.html>`_
  blog post provides instructions for creating a machine learning environment for
  PyTorch and TensorFlow on AMD GPUs using ROCm. It covers steps like installing
  the libraries, cloning code repositories, installing dependencies, and
  troubleshooting potential issues with CUDA-based code. Additionally, it
  explains how to HIPify code (port CUDA code to HIP) and manage Docker images for
  a better experience on AMD GPUs. This guide aims to help data scientists and
  ML practitioners adapt their code for AMD GPUs.

For more use cases and recommendations, see `ROCm Tensorflow blog posts <https://rocm.blogs.amd.com/blog/tag/tensorflow.html>`_
