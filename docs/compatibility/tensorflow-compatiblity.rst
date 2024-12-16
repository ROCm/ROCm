.. meta::
    :description: TensorFlow compatibility
    :keywords: GPU, TensorFlow compatibility

*******************************************************************************
TensorFlow compatibility
*******************************************************************************

TensorFlow is an open-source library for solving machine learning, deep
learning, and AI problems. It can solve many problems across different sectors
and industries, but primarily focuses on neural network training and inference.
It is one of the most popular and in-demand frameworks and is very active in
open-source contribution and development.

`TensorFlow <https://www.tensorflow.org/>`_ is an open-source library designed
for solving machine learning, deep learning, and AI problems. It can solve many
problems across different sectors and industries, but primarily focuses on
neural network training and inference. TensorFlow on ROCm provides
mixed-precision and large-scale training using
`MIOpen <https://github.com/ROCm/MIOpen>`_ and
`RCCL <https://github.com/ROCm/rccl>`_ libraries.

ROCm support for TensorFlow is upstreamed into the official TensorFlow
repository. Due to independent compatibility considerations, this results in
two distinct release cycles for TensorFlow on ROCm:

- ROCm TensorFlow release:

  - Provides the latest version of ROCm but doesn't immediately support the
    latest stable TensorFlow version.
  - Offers `Docker images <https://hub.docker.com/r/rocm/tensorflow>`_ with
    ROCm and TensorFlow pre-installed.
  - ROCm TensorFlow repository: `<https://github.com/ROCm/tensorflow-upstream>`__
  - See the :doc:`ROCm TensorFlow installation guide <rocm-install-on-linux:install/3rd-party/tensorflow-install>`
    to get started.

- Official TensorFlow release:

  - Provides the latest stable version of TensorFlow but doesn't immediately
    support the latest ROCm version.
  - Official TensorFlow repository: `<https://github.com/tensorflow/tensorflow>`__
  - See the `Nightly and latest stable version installation guide <https://www.tensorflow.org/install>`_
    or `Previous versions <https://www.tensorflow.org/versions>`_ to get started.

The upstream TensorFlow includes an automatic hipification solution that
automatically generates HIP source code from the CUDA backend. This approach
allows TensorFlow to support ROCm without requiring manual code modifications.

ROCm's development is aligned with the stable release of TensorFlow while
upstream TensorFlow testing uses the stable release of ROCm to maintain
consistency.

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
     - Linux
     - Python

   * - `rocm6.3-py3.10-tf0.37.1-dev <https://hub.docker.com/layers/rocm/tensorflow/rocm6.3-py3.10-tf0.37.1-dev/images/sha256-d1b63d8df056f9f1cc5d1454406ce7e6a1decf18ed9fe42e5df44f3e29587f85>`_
     - `0.37.1 <https://github.com/tensorflow/tensorflow/tree/v0.37.1>`_
     - **Ubuntu 22.04**
     - `3.10 <https://www.python.org/downloads/release/python-31013/>`_

   * - `rocm6.3-py3.10-tf2.15.0-dev <https://hub.docker.com/layers/rocm/tensorflow/rocm6.3-py3.10-tf2.15.0-dev/images/sha256-f1c633cbcebb9e34660c06bff5aa22dee82a9e2a4919ba923deb32216edce5db>`_
     - `2.15.0 <https://github.com/tensorflow/tensorflow/tree/v2.15.0>`_
     - **Ubuntu 22.04**
     - `3.10 <https://www.python.org/downloads/release/python-31013/>`_

   * - `rocm6.3-py3.10-tf2.15.0-runtime <https://hub.docker.com/layers/rocm/tensorflow/rocm6.3-py3.10-tf2.15.0-runtime/images/sha256-37e0ab694ac0c65afbf34e32e115122d1c2af37e8095740ac1c951e48faed4e7>`_
     - `2.15.0 <https://github.com/tensorflow/tensorflow/tree/v2.15.0>`_
     - **Ubuntu 22.04**
     - `3.10 <https://www.python.org/downloads/release/python-31013/>`_

ROCm critical libraries for Tensorflow
===============================================================================

TensorFlow depends on multiple components, and the supported features of those
components can affect the TensorFlow ROCm supported feature set.
The version mentioned refers to the first TensorFlow version where the ROCm
library was introduced as a dependency.

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
    * - `rocBLAS <https://github.com/ROCm/rocBLAS>`_
      - 4.3.0
      - Optimized BLAS library for AMD GPUs.
      - Linear algebra operations in TensorFlow and other ML frameworks
    * - `rocFFT <https://github.com/ROCm/rocFFT>`_
      - 1.0.31
      - Fast Fourier Transform library for AMD GPUs.
      - Signal processing, scientific computing
    * - `rocPRIM <https://github.com/ROCm/rocPRIM>`_
      - 3.3.0
      - Provides optimized parallel primitives.
      - Parallel algorithms, data processing
    * - `rocRAND <https://github.com/ROCm/rocRAND>`_
      - 3.2.0
      - Random number generation library for AMD GPUs.
      - Stochastic processes, statistical sampling
    * - `rocSOLVER <https://github.com/ROCm/rocSOLVER>`_
      - 3.27.0
      - Provides LAPACK functionalities for AMD GPUs.
      - Solving linear equations, matrix factorizations


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


Use cases and recommendations
===============================================================================


