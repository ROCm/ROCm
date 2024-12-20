.. meta::
    :description: This page lists supported graph safe ROCm libraries.
    :keywords: AMD, ROCm, HIP, hipGRAPH

********************************************************************************
Graph-safe support for ROCm libraries
********************************************************************************

HIP graph-safe libraries operate safely in HIP execution graphs.
:ref:`hip:how_to_HIP_graph` are an alternative way of executing tasks on a GPU
that can provide performance benefits over launching kernels using the standard
method via streams.

Functions and routines from graph-safe libraries shouldn’t result in issues like
race conditions, deadlocks, or unintended dependencies.

The following table shows whether a ROCm library is graph-safe.

.. list-table::
    :header-rows: 1

    *
      - ROCm library
      - Graph safe support
    * 
      - `Composable Kernel <https://github.com/ROCm/composable_kernel>`_
      - ❌
    * 
      - `hipBLAS <https://github.com/ROCm/hipBLAS>`_
      - ✅
    * 
      - `hipBLASLt <https://github.com/ROCm/hipBLASLt>`_
      - ⚠️
    * 
      - `hipCUB <https://github.com/ROCm/hipCUB>`_
      - ✅
    * 
      - `hipFFT <https://github.com/ROCm/hipFFT>`_
      - ✅ (see :ref:`details <hipfft:hip-graph-support-for-hipfft>`)
    * 
      - `hipRAND <https://github.com/ROCm/hipRAND>`_
      - ✅
    * 
      - `hipSOLVER <https://github.com/ROCm/hipSOLVER>`_
      - ⚠️ (experimental)
    * 
      - `hipSPARSE <https://github.com/ROCm/hipSPARSE>`_
      - ✅
    * 
      - `hipSPARSELt <https://github.com/ROCm/hipSPARSELt>`_
      - ⚠️ (experimental)
    * 
      - `hipTensor <https://github.com/ROCm/hipTensor>`_
      - ❌
    * 
      - `MIOpen <https://github.com/ROCm/MIOpen>`_
      - ❌
    * 
      - `RCCL <https://github.com/ROCm/rccl>`_
      - ✅
    * 
      - `rocAL <https://github.com/ROCm/rocAL>`_
      - ❌
    * 
      - `rocALUTION <https://github.com/ROCm/rocALUTION>`_
      - ❌
    * 
      - `rocBLAS <https://github.com/ROCm/rocBLAS>`_
      - ✅ (see :doc:`details <rocblas:reference/beta-features>`)
    * 
      - `rocDecode <https://github.com/ROCm/rocDecode>`_
      - ❌
    * 
      - `rocFFT <https://github.com/ROCm/rocFFT>`_
      - ✅ (see :ref:`details <rocfft:hip-graph-support-for-rocfft>`)
    * 
      - `rocHPCG <https://github.com/ROCm/rocHPCG>`_
      - ❌
    * 
      - `rocJPEG <https://github.com/ROCm/rocJPEG>`_
      - ❌
    * 
      - `rocPRIM <https://github.com/ROCm/rocPRIM>`_
      - ✅
    * 
      - `rocRAND <https://github.com/ROCm/rocRAND>`_
      - ✅
    * 
      - `rocSOLVER <https://github.com/ROCm/rocSOLVER>`_
      - ⚠️ (experimental)
    * 
      - `rocSPARSE <https://github.com/ROCm/rocSPARSE>`_
      - ⚠️ (experimental)
    * 
      - `rocThrust <https://github.com/ROCm/rocThrust>`_
      - ❌ (see :doc:`details <rocthrust:hipgraph-support>`)
    * 
      - `rocWMMA <https://github.com/ROCm/rocWMMA>`_
      - ❌
    * 
      - `RPP <https://github.com/ROCm/rpp>`_
      - ⚠️
    * 
      - `Tensile <https://github.com/ROCm/Tensile>`_
      - ✅

✅: full support

⚠️: partial support

❌: not supported
