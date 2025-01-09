.. meta::
    :description: TensorFlow compatibility
    :keywords: GPU, TensorFlow compatibility

*******************************************************************************
TensorFlow compatibility
*******************************************************************************

`TensorFlow <https://www.tensorflow.org/>`_ is an open-source library for
solving machine learning, deep learning, and AI problems. It can solve many
problems across different sectors and industries but primarily focuses on
neural network training and inference. It is one of the most popular and
in-demand frameworks and is very active in open-source contribution and
development.

The `official TensorFlow repository <http://github.com/tensorflow/tensorflow>`_
includes full ROCm support. AMD maintains a TensorFlow `ROCm repository
<http://github.com/rocm/tensorflow-upstream>`_ in order to quickly add bug
fixes, updates, and support for the latest ROCM versions.

- ROCm TensorFlow release:

  - Offers :ref:`Docker images <tensorflow-docker-compat>` with
    ROCm and TensorFlow pre-installed.

  - ROCm TensorFlow repository: `<https://github.com/ROCm/tensorflow-upstream>`_

  - See the :doc:`ROCm TensorFlow installation guide <rocm-install-on-linux:install/3rd-party/tensorflow-install>`
    to get started.

- Official TensorFlow release:

  - Official TensorFlow repository: `<https://github.com/tensorflow/tensorflow>`_

  - See the `TensorFlow API versions <https://www.tensorflow.org/versions>`_ list.

  .. note::

     The official TensorFlow documentation does not cover ROCm support. Use the
     ROCm documentation for installation instructions for Tensorflow on ROCm.
     See :doc:`rocm-install-on-linux:install/3rd-party/tensorflow-install`.

.. _tensorflow-docker-compat:

Docker image compatibility
===============================================================================

.. |docker-icon| raw:: html

   <i class="fab fa-docker"></i>

AMD validates and publishes ready-made `TensorFlow
<https://hub.docker.com/r/rocm/tensorflow>`_ images with ROCm backends on
Docker Hub. The following Docker image tags and associated inventories are
validated for `ROCm 6.3.1 <https://repo.radeon.com/rocm/apt/6.3.1/>`_. Click
the |docker-icon| icon to view the image on Docker Hub.

.. list-table:: TensorFlow Docker image components
    :header-rows: 1

    * - Docker image
      - TensorFlow
      - Dev
      - Python
      - TensorBoard 

    * - .. raw:: html

           <a href="https://hub.docker.com/layers/rocm/tensorflow/rocm6.3.1-py3.12-tf2.17.0-dev/images/sha256-804121ee4985718277ba7dcec53c57bdade130a1ef42f544b6c48090ad379c17"><i class="fab fa-docker fa-lg"></i> rocm/tensorflow</a>

      - `tensorflow-rocm 2.17.0 <https://repo.radeon.com/rocm/manylinux/rocm-rel-6.3/tensorflow_rocm-2.17.0-cp312-cp312-manylinux_2_28_x86_64.whl>`_
      - dev
      - `Python 3.12 <https://www.python.org/downloads/release/python-3124/>`_
      - `TensorBoard 2.17.1 <https://github.com/tensorflow/tensorboard/tree/2.17.1>`_

    * - .. raw:: html

           <a href="https://hub.docker.com/layers/rocm/tensorflow/rocm6.3.1-py3.10-tf2.17.0-dev/images/sha256-776837ffa945913f6c466bfe477810a11453d21d5b6afb200be1c36e48fbc08e"><i class="fab fa-docker fa-lg"></i> rocm/tensorflow</a>

      - `tensorflow-rocm 2.17.0 <https://repo.radeon.com/rocm/manylinux/rocm-rel-6.3/tensorflow_rocm-2.17.0-cp310-cp310-manylinux_2_28_x86_64.whl>`_
      - dev
      - `Python 3.10 <https://www.python.org/downloads/release/python-31012/>`_
      - `TensorBoard 2.17.0 <https://github.com/tensorflow/tensorboard/tree/2.17.0>`_

    * - .. raw:: html

           <a href="https://hub.docker.com/layers/rocm/tensorflow/rocm6.3.1-py3.12-tf2.16.2-dev/images/sha256-c793e1483e30809c3c28fc5d7805bedc033c73da224f839fff370717cb100944"><i class="fab fa-docker fa-lg"></i> rocm/tensorflow</a>

      - `tensorflow-rocm 2.16.2 <https://repo.radeon.com/rocm/manylinux/rocm-rel-6.3/tensorflow_rocm-2.16.2-cp312-cp312-manylinux_2_28_x86_64.whl>`_
      - dev
      - `Python 3.12 <https://www.python.org/downloads/release/python-3124/>`_
      - `TensorBoard 2.16.2 <https://github.com/tensorflow/tensorboard/tree/2.16.2>`_

    * - .. raw:: html

           <a href="https://hub.docker.com/layers/rocm/tensorflow/rocm6.3.1-py3.10-tf2.16.0-dev/images/sha256-263e78414ae85d7bcd52a025a94131d0a279872a45ed632b9165336dfdcd4443"><i class="fab fa-docker fa-lg"></i> rocm/tensorflow</a>

      - `tensorflow-rocm 2.16.2 <https://repo.radeon.com/rocm/manylinux/rocm-rel-6.3/tensorflow_rocm-2.16.2-cp310-cp310-manylinux_2_28_x86_64.whl>`_
      - dev
      - `Python 3.10 <https://www.python.org/downloads/release/python-31012/>`_
      - `TensorBoard 2.16.2 <https://github.com/tensorflow/tensorboard/tree/2.16.2>`_

    * - .. raw:: html

           <a href="https://hub.docker.com/layers/rocm/tensorflow/rocm6.3.1-py3.10-tf2.15.0-dev/images/sha256-479046a8477ca701a9494a813ab17e8ab4f6baa54641e65dc8d07629f1e6a880"><i class="fab fa-docker fa-lg"></i> rocm/tensorflow</a>

      - `tensorflow-rocm 2.15.1 <https://repo.radeon.com/rocm/manylinux/rocm-rel-6.3/tensorflow_rocm-2.15.1-cp310-cp310-manylinux_2_28_x86_64.whl>`_
      - dev
      - `Python 3.10 <https://www.python.org/downloads/release/python-31012/>`_
      - `TensorBoard 2.15.2 <https://github.com/tensorflow/tensorboard/tree/2.15.2>`_

Critical ROCm libraries for TensorFlow
===============================================================================

TensorFlow depends on multiple components and the supported features of those
components can affect the TensorFlow ROCm supported feature set. The versions
in the following table refer to the first TensorFlow version where the ROCm
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
      - Extends hipBLAS with additional optimizations like fused kernels and
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
    * - `rocThrust <https://github.com/ROCm/rocThrust>`_
      - 3.3.0
      - Provides a C++ template library for parallel algorithms like sorting,
        reduction, and scanning.
      - Reduction operations like ``tf.reduce_sum``, ``tf.cumsum`` for computing
        the cumulative sum of elements along a given axis or ``tf.unique`` to
        finds unique elements in a tensor can use rocThrust.

Supported and unsupported features
===============================================================================

The following section maps supported data types and GPU-accelerated TensorFlow
features to their minimum supported ROCm and TensorFlow versions.

Data types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The data type of a tensor is specified using the ``dtype`` attribute or
argument, and TensorFlow supports a wide range of data types for different use
cases.

The basic, single data types of `tf.dtypes <https://www.tensorflow.org/api_docs/python/tf/dtypes>`_
are as follows:

.. list-table::
    :header-rows: 1

    * - Data type
      - Description
      - Since TensorFlow
      - Since ROCm
    * - ``bfloat16``
      - 16-bit bfloat (brain floating point).
      - 1.0.0
      - 1.7
    * - ``bool``
      - Boolean.
      - 1.0.0
      - 1.7
    * - ``complex128``
      - 128-bit complex.
      - 1.0.0
      - 1.7
    * - ``complex64``
      - 64-bit complex.
      - 1.0.0
      - 1.7
    * - ``double``
      - 64-bit (double precision) floating-point.
      - 1.0.0
      - 1.7
    * - ``float16``
      - 16-bit (half precision) floating-point.
      - 1.0.0
      - 1.7
    * - ``float32``
      - 32-bit (single precision) floating-point.
      - 1.0.0
      - 1.7
    * - ``float64``
      - 64-bit (double precision) floating-point.
      - 1.0.0
      - 1.7
    * - ``half``
      - 16-bit (half precision) floating-point.
      - 2.0.0
      - 2.0
    * - ``int16``
      - Signed 16-bit integer.
      - 1.0.0
      - 1.7
    * - ``int32``
      - Signed 32-bit integer.
      - 1.0.0
      - 1.7
    * - ``int64``
      - Signed 64-bit integer.
      - 1.0.0
      - 1.7
    * - ``int8``
      - Signed 8-bit integer.
      - 1.0.0
      - 1.7
    * - ``qint16``
      - Signed quantized 16-bit integer.
      - 1.0.0
      - 1.7
    * - ``qint32``
      - Signed quantized 32-bit integer.
      - 1.0.0
      - 1.7
    * - ``qint8``
      - Signed quantized 8-bit integer.
      - 1.0.0
      - 1.7
    * - ``quint16``
      - Unsigned quantized 16-bit integer.
      - 1.0.0
      - 1.7
    * - ``quint8``
      - Unsigned quantized 8-bit integer.
      - 1.0.0
      - 1.7
    * - ``resource``
      - Handle to a mutable, dynamically allocated resource.
      - 1.0.0
      - 1.7
    * - ``string``
      - Variable-length string, represented as byte array.
      - 1.0.0
      - 1.7
    * - ``uint16``
      - Unsigned 16-bit (word) integer.
      - 1.0.0
      - 1.7
    * - ``uint32``
      - Unsigned 32-bit (dword) integer.
      - 1.5.0
      - 1.7
    * - ``uint64``
      - Unsigned 64-bit (qword) integer.
      - 1.5.0
      - 1.7
    * - ``uint8``
      - Unsigned 8-bit (byte) integer.
      - 1.0.0
      - 1.7
    * - ``variant``
      - Data of arbitrary type (known at runtime).
      - 1.4.0
      - 1.7

Features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This table provides an overview of key features in TensorFlow and their
availability in ROCm.

.. list-table::
    :header-rows: 1

    * - Module
      - Description
      - Since TensorFlow
      - Since ROCm
    * - ``tf.linalg`` (Linear Algebra)
      - Operations for matrix and tensor computations, such as
        ``tf.linalg.matmul`` (matrix multiplication), ``tf.linalg.inv``
        (matrix inversion) and ``tf.linalg.cholesky`` (Cholesky decomposition).
        These leverage GPUs for high-performance linear algebra operations.
      - 1.4
      - 1.8.2
    * - ``tf.nn`` (Neural Network Operations)
      - GPU-accelerated building blocks for deep learning models, such as 2D
        convolutions with ``tf.nn.conv2d``, max pooling operations with
        ``tf.nn.max_pool``, activation functions like ``tf.nn.relu`` or softmax
        for output layers with ``tf.nn.softmax``.
      - 1.0
      - 1.8.2
    * - ``tf.image`` (Image Processing)
      - GPU-accelerated functions for image preprocessing and augmentations,
        such as resize images with ``tf.image.resize``, flip images horizontally
        with ``tf.image.flip_left_right`` and adjust image brightness randomly
        with ``tf.image.random_brightness``.
      - 1.1
      - 1.8.2
    * - ``tf.keras`` (High-Level API)
      - GPU acceleration for Keras layers and models, including dense layers
        (``tf.keras.layers.Dense``), convolutional layers
        (``tf.keras.layers.Conv2D``) and recurrent layers
        (``tf.keras.layers.LSTM``).
      - 1.4
      - 1.8.2
    * - ``tf.math`` (Mathematical Operations)
      - GPU-accelerated mathematical operations, such as sum across dimensions
        with ``tf.math.reduce_sum``, elementwise exponentiation with
        ``tf.math.exp`` and sigmoid activation (``tf.math.sigmoid``).
      - 1.5
      - 1.8.2
    * - ``tf.signal`` (Signal Processing)
      - Functions for spectral analysis and signal transformations.
      - 1.13
      - 2.1
    * - ``tf.data`` (Data Input Pipeline)
      - GPU-accelerated data preprocessing for efficient input pipelines, 
        Prefetching with ``tf.data.experimental.AUTOTUNE``. GPU-enabled
        transformations like map and batch. 
      - 1.4
      - 1.8.2
    * - ``tf.distribute`` (Distributed Training)
      - Enabling to scale computations across multiple devices on a single
        machine or across multiple machines.
      - 1.13
      - 2.1
    * - ``tf.random`` (Random Number Generation)
      - GPU-accelerated random number generation
      - 1.12
      - 1.9.2
    * - ``tf.TensorArray`` (Dynamic Array Operations)
      - Enables dynamic tensor manipulation on GPUs.
      - 1.0
      - 1.8.2
    * - ``tf.sparse`` (Sparse Tensor Operations)
      - GPU-accelerated sparse matrix manipulations.
      - 1.9
      - 1.9.0
    * - ``tf.experimental.numpy``
      - GPU-accelerated NumPy-like API for numerical computations.
      - 2.4
      - 4.1.1
    * - ``tf.RaggedTensor``
      - Handling of variable-length sequences and ragged tensors with GPU
        support.
      - 1.13
      - 2.1
    * - ``tf.function`` with XLA (Accelerated Linear Algebra)
      - Enable GPU-accelerated functions in optimization.
      - 1.14
      - 2.4
    * - ``tf.quantization``
      - Quantized operations for inference, accelerated on GPUs.
      - 1.12 
      - 1.9.2

Distributed library features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Enables developers to scale computations across multiple devices on a single machine or
across multiple machines.

.. list-table::
   :header-rows: 1

   * - Feature
     - Description
     - Since TensorFlow
     - Since ROCm
   * - ``MultiWorkerMirroredStrategy``
     - Synchronous training across multiple workers using mirrored variables.
     - 2.0
     - 3.0
   * - ``MirroredStrategy``
     - Synchronous training across multiple GPUs on one machine.
     - 1.5
     - 2.5
   * - ``TPUStrategy``
     - Efficiently trains models on Google TPUs.
     - 1.9
     - ❌
   * - ``ParameterServerStrategy``
     - Asynchronous training using parameter servers for variable management.
     - 2.1
     - 4.0
   * - ``CentralStorageStrategy``
     - Keeps variables on a single device and performs computation on multiple
       devices.
     - 2.3
     - 4.1
   * - ``CollectiveAllReduceStrategy``
     - Synchronous training across multiple devices and hosts.
     - 1.14
     - 3.5
   * - Distribution Strategies API
     - High-level API to simplify distributed training configuration and
       execution.
     - 1.10
     - 3.0

Unsupported TensorFlow features
===============================================================================

The following are GPU-accelerated TensorFlow features not currently supported by
ROCm.

.. list-table::
    :header-rows: 1

    * - Feature
      - Description
      - Since TensorFlow
    * - Mixed Precision with TF32
      - Mixed precision with TF32 is used for matrix multiplications,
        convolutions, and other linear algebra operations, particularly in
        deep learning workloads like CNNs and transformers.
      - 2.4
    * - ``tf.distribute.TPUStrategy``
      - Efficiently trains models on Google TPUs.
      - 1.9

Use cases and recommendations
===============================================================================

* The `Training a Neural Collaborative Filtering (NCF) Recommender on an AMD
  GPU <https://rocm.blogs.amd.com/artificial-intelligence/ncf/README.html>`_
  blog post discusses training an NCF recommender system using TensorFlow. It
  explains how NCF improves traditional collaborative filtering methods by
  leveraging neural networks to model non-linear user-item interactions. The
  post outlines the implementation using the recommenders library, focusing on
  the use of implicit data (for example, user interactions like viewing or
  purchasing) and how it addresses challenges like the lack of negative values.

* The `Creating a PyTorch/TensorFlow code environment on AMD GPUs
  <https://rocm.blogs.amd.com/software-tools-optimization/pytorch-tensorflow-env/README.html>`_
  blog post provides instructions for creating a machine learning environment
  for PyTorch and TensorFlow on AMD GPUs using ROCm. It covers steps like
  installing the libraries, cloning code repositories, installing dependencies,
  and troubleshooting potential issues with CUDA-based code. Additionally, it
  explains how to HIPify code (port CUDA code to HIP) and manage Docker images
  for a better experience on AMD GPUs. This guide aims to help data scientists
  and ML practitioners adapt their code for AMD GPUs.

For more use cases and recommendations, see the `ROCm Tensorflow blog posts <https://rocm.blogs.amd.com/blog/tag/tensorflow.html>`_.
