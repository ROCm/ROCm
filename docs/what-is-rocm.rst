 .. meta::
  :description: What is ROCm
  :keywords: ROCm components, ROCm projects, introduction, ROCm, AMD, runtimes, compilers, tools, libraries, API

***********************************************************
What is ROCm?
***********************************************************

ROCm is a software stack, composed primarily of open-source software, that
provides the tools for programming AMD Graphics Processing Units (GPUs), from
low-level kernels to high-level end-user applications.

.. image:: data/rocm-software-stack-6_3_2.jpg
  :width: 800
  :alt: AMD's ROCm software stack and enabling technologies.
  :align: center

Specifically, ROCm provides the tools for
:doc:`HIP (Heterogeneous-computing Interface for Portability) <hip:index>`,
OpenCL and OpenMP. These include compilers, libraries for high-level
functions, debuggers, profilers and runtimes.

ROCm components
===============================================

ROCm consists of the following components. For information on the license associated with each component,
see :doc:`ROCm licensing <./about/license>`.

Libraries
-----------------------------------------------

Machine Learning & Computer Vision
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
  :header: "Component", "Description"

  ":doc:`Composable Kernel <composable_kernel:index>`", "Provides a programming model for writing performance critical kernels for machine learning workloads across multiple architectures"
  ":doc:`MIGraphX <amdmigraphx:index>`", "Graph inference engine that accelerates machine learning model inference"
  ":doc:`MIOpen <miopen:index>`", "An open source deep-learning library"
  ":doc:`MIVisionX <mivisionx:index>`", "Set of comprehensive computer vision and machine learning libraries, utilities, and applications"
  ":doc:`ROCm Performance Primitives (RPP) <rpp:index>`", "Comprehensive high-performance computer vision library for AMD processors with HIP/OpenCL/CPU back-ends"
  ":doc:`rocAL <rocal:index>`", "An augmentation library designed to decode and process images and videos"
  ":doc:`rocDecode <rocdecode:index>`", "High-performance SDK for access to video decoding features on AMD GPUs"
  ":doc:`rocJPEG <rocjpeg:index>`", "Library for decoding JPG images on AMD GPUs"
  ":doc:`rocPyDecode <rocpydecode:index>`", "Provides access to rocDecode APIs in both Python and C/C++ languages"

Communication
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
  :header: "Component", "Description"

  ":doc:`RCCL <rccl:index>`", "Standalone library that provides multi-GPU and multi-node collective communication primitives"

Math
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
  :header: "Component", "Description"

  "`half <https://github.com/ROCm/half/>`_", "C++ header-only library that provides an IEEE 754 conformant, 16-bit half-precision floating-point type, along with corresponding arithmetic operators, type conversions, and common mathematical functions"
  ":doc:`hipBLAS <hipblas:index>`", "BLAS-marshaling library that supports :doc:`rocBLAS <rocblas:index>` and cuBLAS backends"
  ":doc:`hipBLASLt <hipblaslt:index>`", "Provides general matrix-matrix operations with a flexible API and extends functionalities beyond traditional BLAS library"
  ":doc:`hipFFT <hipfft:index>`", "Fast Fourier transforms (FFT)-marshalling library that supports rocFFT or cuFFT backends"
  ":doc:`hipfort <hipfort:index>`", "Fortran interface library for accessing GPU Kernels"
  ":doc:`hipRAND <hiprand:index>`", "Ports CUDA applications that use the cuRAND library into the HIP layer"
  ":doc:`hipSOLVER <hipsolver:index>`", "An LAPACK-marshalling library that supports :doc:`rocSOLVER <rocsolver:index>` and cuSOLVER backends"
  ":doc:`hipSPARSE <hipsparse:index>`", "SPARSE-marshalling library that supports :doc:`rocSPARSE <rocsparse:index>` and cuSPARSE backends"
  ":doc:`hipSPARSELt <hipsparselt:index>`", "SPARSE-marshalling library with multiple supported backends"
  ":doc:`rocALUTION <rocalution:index>`", "Sparse linear algebra library for exploring fine-grained parallelism on ROCm runtime and toolchains"
  ":doc:`rocBLAS <rocblas:index>`", "BLAS implementation (in the HIP programming language) on the ROCm runtime and toolchains"
  ":doc:`rocFFT <rocfft:index>`", "Software library for computing fast Fourier transforms (FFTs) written in HIP"
  ":doc:`rocRAND <rocrand:index>`", "Provides functions that generate pseudorandom and quasirandom numbers"
  ":doc:`rocSOLVER <rocsolver:index>`", "An implementation of LAPACK routines on ROCm software, implemented in the HIP programming language and optimized for AMD's latest discrete GPUs"
  ":doc:`rocSPARSE <rocsparse:index>`", "Exposes a common interface that provides BLAS for sparse computation implemented on ROCm runtime and toolchains (in the HIP programming language)"
  ":doc:`rocWMMA <rocwmma:index>`", "C++ library for accelerating mixed-precision matrix multiply-accumulate (MMA) operations"
  ":doc:`Tensile <tensile:src/index>`", "Creates benchmark-driven backend libraries for GEMMs, GEMM-like problems, and general N-dimensional tensor contractions"

Primitives
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
  :header: "Component", "Description"

  ":doc:`hipCUB <hipcub:index>`", "Thin header-only wrapper library on top of :doc:`rocPRIM <rocprim:index>` or CUB that allows project porting using the CUB library to the HIP layer"
  ":doc:`hipTensor <hiptensor:index>`", "AMD's C++ library for accelerating tensor primitives based on the composable kernel library"
  ":doc:`rocPRIM <rocprim:index>`", "Header-only library for HIP parallel primitives"
  ":doc:`rocThrust <rocthrust:index>`", "Parallel algorithm library"

Tools
-----------------------------------------------

System Management
^^^^^^^^^^^^^^^^^

.. csv-table::
  :header: "Component", "Description"

  ":doc:`AMD SMI <amdsmi:index>`", "C library for Linux that provides a user space interface for applications to monitor and control AMD devices"
  ":doc:`ROCm Data Center Tool <rdc:index>`", "Simplifies administration and addresses key infrastructure challenges in AMD GPUs in cluster and data-center environments"
  ":doc:`rocminfo <rocminfo:index>`", "Reports system information"
  ":doc:`ROCm SMI <rocm_smi_lib:index>`", "C library for Linux that provides a user space interface for applications to monitor and control GPU applications"
  ":doc:`ROCm Validation Suite <rocmvalidationsuite:index>`", "Detects and troubleshoots common problems affecting AMD GPUs running in a high-performance computing environment"

Performance
^^^^^^^^^^^

.. csv-table::
  :header: "Component", "Description"

  ":doc:`ROCm Bandwidth Test <rocm_bandwidth_test:index>`", "Captures the performance characteristics of buffer copying and kernel read/write operations"
  ":doc:`ROCm Compute Profiler <rocprofiler-compute:index>`", "Kernel-level profiling for machine learning and high performance computing (HPC) workloads"
  ":doc:`ROCm Systems Profiler <rocprofiler-systems:index>`", "Comprehensive profiling and tracing of applications running on the CPU or the CPU and GPU"
  ":doc:`ROCProfiler <rocprofiler:index>`", "Profiling tool for HIP applications"
  ":doc:`ROCprofiler-SDK <rocprofiler-sdk:index>`", "Toolkit for developing analysis tools for profiling and tracing GPU compute applications. This toolkit is in beta and subject to change"
  ":doc:`ROCTracer <roctracer:index>`", "Intercepts runtime API calls and traces asynchronous activity"

Development
^^^^^^^^^^^

.. csv-table::
  :header: "Component", "Description"

  ":doc:`HIPIFY <hipify:index>`", "Translates CUDA source code into portable HIP C++"
  ":doc:`ROCm CMake <rocmcmakebuildtools:index>`", "Collection of CMake modules for common build and development tasks"
  ":doc:`ROCdbgapi <rocdbgapi:index>`", "ROCm debugger API library"
  ":doc:`ROCm Debugger (ROCgdb) <rocgdb:index>`", "Source-level debugger for Linux, based on the GNU Debugger (GDB)"
  ":doc:`ROCr Debug Agent <rocr_debug_agent:index>`", "Prints the state of all AMD GPU wavefronts that caused a queue error by sending a SIGQUIT signal to the process while the program is running"

Compilers
-----------------------------------------------

.. csv-table::
  :header: "Component", "Description"

  ":doc:`HIPCC <hipcc:index>`", "Compiler driver utility that calls Clang or NVCC and passes the appropriate include and library options for the target compiler and HIP infrastructure"
  ":doc:`ROCm compilers <llvm-project:index>`", "ROCm LLVM compiler infrastructure"
  "`FLANG <https://github.com/ROCm/flang/>`_", "An out-of-tree Fortran compiler targeting LLVM"

Runtimes
-----------------------------------------------

.. csv-table::
  :header: "Component", "Description"

  ":doc:`AMD Compute Language Runtime (CLR) <hip:understand/amd_clr>`", "Contains source code for AMD's compute language runtimes: HIP and OpenCL"
  ":doc:`HIP <hip:index>`", "AMD's GPU programming language extension and the GPU runtime"
  ":doc:`ROCR-Runtime <rocr-runtime:index>`", "User-mode API interfaces and libraries necessary for host applications to launch compute kernels on available HSA ROCm kernel agents"
