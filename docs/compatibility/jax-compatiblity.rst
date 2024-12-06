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

Release notes
===============================================================================

.. list-table::
    :header-rows: 1
    :name: jax-rocm-compatibility

    * - JAX versions
      - ROCm version
    * - 0.3.2--0.3.4
      - 5.0.0
    * - 0.3.5--0.3.14
      - 5.1.0
    * - 0.3.15--0.3.21
      - 5.2.0
    * - 0.3.22--0.4.3
      - 5.3.0
    * - 0.4.4--0.4.12
      - 5.4.0
    * - 0.4.13--0.4.15
      - 5.5.0
    * - 0.4.16--0.4.23
      - 5.6.0
    * - 0.4.24--0.4.30
      - 6.0.0
    * - 0.4.31--0.4.35
      - 6.1.3

Supported features
===============================================================================

.. list-table::
    :header-rows: 1

    * - module
      - minimum JAX version
    * - ``jax.numpy``
      - 0.1.56
    * - ``jax.scipy``
      - 0.1.56
    * - ``jax.lax``
      - 0.1.57
    * - ``jax.random``
      - 0.1.58
    * - ``jax.sharding``
      - 0.3.20
    * - ``jax.debug``
      - 0.3.11
    * - ``jax.dlpack``
      - 0.1.57
    * - ``jax.distributed``
      - 0.1.74
    * - ``jax.dtypes``
      - 0.1.66
    * - ``jax.flatten_util``
      - 0.1.72
    * - ``jax.image``
      - 0.1.57
    * - ``jax.nn``
      - 0.1.56
    * - ``jax.ops``
      - 0.1.57
    * - ``jax.profiler``
      - 0.1.57
    * - ``jax.stages``
      - 0.3.4
    * - ``jax.tree``
      - 0.4.26
    * - ``jax.tree_util``
      - 0.1.65
    * - ``jax.typing``
      - 0.3.18
    * - ``jax.export``
      - 0.4.30
    * - ``jax.extend``
      - 0.4.15
    * - ``jax.example_libraries``
      - 0.1.74
    * - ``jax.experimental``
      - 0.1.56
    * - ``jax.lib``
      - 0.4.6

``jax.scipy``
-------------------------------------------------------------------------------

A SciPy-like API for scientific computing.

.. list-table::
    :header-rows: 1

    * - module
      - minimum JAX version
    * - ``jax.scipy.cluster``
      - 0.3.11
    * - ``jax.scipy.fft``
      - 0.1.71
    * - ``jax.scipy.integrate``
      - 0.4.15
    * - ``jax.scipy.interpolate``
      - 0.1.76
    * - ``jax.scipy.linalg``
      - 0.1.56
    * - ``jax.scipy.ndimage``
      - 0.1.56
    * - ``jax.scipy.optimize``
      - 0.1.57
    * - ``jax.scipy.signal``
      - 0.1.56
    * - ``jax.scipy.spatial.transform``
      - 0.4.12
    * - ``jax.scipy.sparse.linalg``
      - 0.1.56
    * - ``jax.scipy.special``
      - 0.1.56
    * - ``jax.scipy.stats``
      - 0.1.56

``jax.scipy.stats``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - module
     - minimum JAX version
   * - ``jax.scipy.stats.bernouli``
     - 0.1.56
   * - ``jax.scipy.stats.beta``
     - 0.1.56
   * - ``jax.scipy.stats.betabinom``
     - 0.1.61
   * - ``jax.scipy.stats.binom``
     - 0.4.14
   * - ``jax.scipy.stats.cauchy``
     - 0.1.56
   * - ``jax.scipy.stats.chi2``
     - 0.1.61
   * - ``jax.scipy.stats.dirichlet``
     - 0.1.56
   * - ``jax.scipy.stats.expon``
     - 0.1.56
   * - ``jax.scipy.stats.gamma``
     - 0.1.56
   * - ``jax.scipy.stats.gennorm``
     - 0.3.15
   * - ``jax.scipy.stats.geom``
     - 0.1.56
   * - ``jax.scipy.stats.laplace``
     - 0.1.56
   * - ``jax.scipy.stats.logistic``
     - 0.1.56
   * - ``jax.scipy.stats.multinomial``
     - 0.3.18
   * - ``jax.scipy.stats.multivariate_normal``
     - 0.1.56
   * - ``jax.scipy.stats.nbinom``
     - 0.1.72
   * - ``jax.scipy.stats.norm``
     - 0.1.56
   * - ``jax.scipy.stats.pareto``
     - 0.1.56
   * - ``jax.scipy.stats.poisson``
     - 0.1.56
   * - ``jax.scipy.stats.t``
     - 0.1.56
   * - ``jax.scipy.stats.truncnorm``
     - 0.4.0
   * - ``jax.scipy.stats.uniform``
     - 0.1.56
   * - ``jax.scipy.stats.vonmises``
     - 0.4.2
   * - ``jax.scipy.stats.wrapcauchy``
     - 0.4.20

``jax.extend``
-------------------------------------------------------------------------------

Modules for JAX extensions.

.. list-table::
    :header-rows: 1

    * - module
      - minimum JAX version
    * - ``jax.extend.ffi``
      - 0.4.30	
    * - ``jax.extend.linear_util``
      - 0.4.17
    * - ``jax.extend.mlir``
      - 0.4.26
    * - ``jax.extend.random``
      - 0.4.15

``jax.experimental``
-------------------------------------------------------------------------------

Experimental modules and APIs.

.. list-table::
    :header-rows: 1

    * - module
      - minimum JAX version
    * - ``jax.experimental.checkify``
      - 0.1.75
    * - ``jax.experimental.compilation_cache.compilation_cache``
      - 0.1.68
    * - ``jax.experimental.custom_partitioning``
      - 0.4.0
    * - ``jax.experimental.jet``
      - 0.1.56
    * - ``jax.experimental.key_reuse``
      - 0.4.26
    * - ``jax.experimental.mesh_utils``
      - 0.1.76
    * - ``jax.experimental.multihost_utils``
      - 0.3.2
    * - ``jax.experimental.pallas``
      - 0.4.15
    * - ``jax.experimental.pjit``
      - 0.1.61
    * - ``jax.experimental.serialize_executable``
      - 0.4.0
    * - ``jax.experimental.shard_map``
      - 0.4.3
    * - ``jax.experimental.sparse``
      - 0.1.75

.. list-table::
    :header-rows: 1

    * - API
      - minimum JAX version
    * - ``jax.experimental.enable_x64``
      - 0.1.60
    * - ``jax.experimental.disable_x64``
      - 0.1.60

``jax.experimental.pallas``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Module for Pallas, a JAX extension for custom kernels.

.. list-table::
    :header-rows: 1

    * - module
      - minimum JAX version
    * - ``jax.experimental.pallas.mosaic_gpu``
      - 0.4.31
    * - ``jax.experimental.pallas.tpu``
      - 0.4.15
    * - ``jax.experimental.pallas.triton``
      - 0.4.32

``jax.experimental.sparse``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Experimental support for sparse matrix operations.

.. list-table::
    :header-rows: 1

    * - module
      - minimum JAX version
    * - ``jax.experimental.sparse.linalg``
      - 0.3.15

.. list-table::
    :header-rows: 1

    * - sparse data structure API
      - minimum JAX version
    * - ``jax.experimental.sparse.BCOO``
      - 0.1.72
    * - ``jax.experimental.sparse.BCSR``
      - 0.3.20
    * - ``jax.experimental.sparse.CSR``
      - 0.1.75
    * - ``jax.experimental.sparse.NM``
      - 0.4.27
    * - ``jax.experimental.sparse.COO``
      - 0.1.75

Run unit tests
===============================================================================

Run unit tests to validate the JAX installation fully.

.. note::

  You must run the following command from the JAX home directory.

.. code-block:: bash

  python3 ./build/rocm/run_single_gpu.py -c


In a multi-GPU environment:

.. code-block:: bash

  ./build/rocm/run_multi_gpu.sh -c


Alternatively you can run tests as:

.. code-block:: bash

  pytest tests/

With multiple GPUs the tests can run in parallel. Use ``pytest-xdist`` and set
the ``XLA_PYTHON_CLIENT_ALLOCATOR`` environment variable to ``platform``:

.. code-block:: bash

  export XLA_PYTHON_CLIENT_ALLOCATOR=platform pytest -n 8 --tb=short tests/

where ``-n 8`` is the number of parallel worker processes to use when running
the tests.

JAX benchmark
===============================================================================

Navigate to the JAX directory and execute a specific JAX benchmark scripts for
ROCm:

.. code-block:: bash

  python3 jax/benchmarks/path_to_specific_benchmark.py


For benchmarking your JAX ported code and more detailed description, the
`JAX benchmark page
<https://jax.readthedocs.io/en/latest/faq.html#benchmarking-jax-code>`_
gives detailed instructions.
