.. meta::
    :description: ROCm compatibility matrix
    :keywords: GPU, architecture, hardware, compatibility, system, requirements, components, libraries

**************************************************************************************
Compatibility matrix
**************************************************************************************

Use this matrix to view the ROCm compatibility and system requirements across successive major and minor releases.

You can also refer to the :ref:`past versions of ROCm compatibility matrix<past-rocm-compatibility-matrix>`.

Accelerators and GPUs listed in the following table support compute workloads (no display
information or graphics). If you’re using ROCm with AMD Radeon or Radeon Pro GPUs for graphics
workloads, see the `Use ROCm on Radeon GPU documentation
<https://rocm.docs.amd.com/projects/radeon/en/latest/docs/compatibility.html>`_ to verify
compatibility and system requirements.

.. |br| raw:: html

   <br/>

.. container:: format-big-table

  .. csv-table::
      :header: "ROCm Version", "6.3.3", "6.3.2", "6.2.0"
      :stub-columns: 1

      :ref:`Operating systems & kernels <OS-kernel-versions>`,Ubuntu 24.04.2,Ubuntu 24.04.2,Ubuntu 24.04
      ,Ubuntu 22.04.5,Ubuntu 22.04.5,"Ubuntu 22.04.5, 22.04.4"
      ,"RHEL 9.5, 9.4","RHEL 9.5, 9.4","RHEL 9.4, 9.3"
      ,RHEL 8.10,RHEL 8.10,"RHEL 8.10, 8.9"
      ,"SLES 15 SP6, SP5","SLES 15 SP6, SP5","SLES 15 SP6, SP5"
      ,Oracle Linux 8.10 [#mi300x]_,Oracle Linux 8.10 [#mi300x]_,Oracle Linux 8.9 [#mi300x]_
      ,Debian 12 [#single-node]_,Debian 12 [#single-node]_,
      ,Azure Linux 3.0 [#mi300x]_,Azure Linux 3.0 [#mi300x]_,
      ,.. _architecture-support-compatibility-matrix:,,
      :doc:`Architecture <rocm-install-on-linux:reference/system-requirements>`,CDNA3,CDNA3,CDNA3
      ,CDNA2,CDNA2,CDNA2
      ,CDNA,CDNA,CDNA
      ,RDNA3,RDNA3,RDNA3
      ,RDNA2,RDNA2,RDNA2
      ,.. _gpu-support-compatibility-matrix:,,
      :doc:`GPU / LLVM target <rocm-install-on-linux:reference/system-requirements>`,gfx1100,gfx1100,gfx1100
      ,gfx1030,gfx1030,gfx1030
      ,gfx942,gfx942,gfx942 [#mi300_620]_
      ,gfx90a,gfx90a,gfx90a
      ,gfx908,gfx908,gfx908
      ,,,
      FRAMEWORK SUPPORT,.. _framework-support-compatibility-matrix:,,
      :doc:`PyTorch <../compatibility/ml-compatibility/pytorch-compatibility>`,"2.4, 2.3, 2.2, 1.13","2.4, 2.3, 2.2, 1.13","2.3, 2.2, 2.1, 2.0, 1.13"
      :doc:`TensorFlow <../compatibility/ml-compatibility/tensorflow-compatibility>`,"2.17.0, 2.16.2, 2.15.1","2.17.0, 2.16.2, 2.15.1","2.16.1, 2.15.1, 2.14.1"
      :doc:`JAX <../compatibility/ml-compatibility/jax-compatibility>`,0.4.31,0.4.31,0.4.26
      `ONNX Runtime <https://onnxruntime.ai/docs/build/eps.html#amd-migraphx>`_,1.17.3,1.17.3,1.17.3
      ,,,
      THIRD PARTY COMMS,.. _thirdpartycomms-support-compatibility-matrix:,,
      `UCC <https://github.com/ROCm/ucc>`_,>=1.3.0,>=1.3.0,>=1.3.0
      `UCX <https://github.com/ROCm/ucx>`_,>=1.15.0,>=1.15.0,>=1.15.0
      ,,,
      THIRD PARTY ALGORITHM,.. _thirdpartyalgorithm-support-compatibility-matrix:,,
      Thrust,2.3.2,2.3.2,2.2.0
      CUB,2.3.2,2.3.2,2.2.0
      ,,,
      KMD & USER SPACE [#kfd_support]_,.. _kfd-userspace-support-compatibility-matrix:,,
      Tested user space versions,"6.3.x, 6.2.x, 6.1.x","6.3.x, 6.2.x, 6.1.x","6.3.x, 6.2.x, 6.1.x, 6.0.x"
      ,,,
      ML & COMPUTER VISION,.. _mllibs-support-compatibility-matrix:,,
      :doc:`Composable Kernel <composable_kernel:index>`,1.1.0,1.1.0,1.1.0
      :doc:`MIGraphX <amdmigraphx:index>`,2.11.0,2.11.0,2.10.0
      :doc:`MIOpen <miopen:index>`,3.3.0,3.3.0,3.2.0
      :doc:`MIVisionX <mivisionx:index>`,3.1.0,3.1.0,3.0.0
      :doc:`rocAL <rocal:index>`,2.1.0,2.1.0,1.0.0
      :doc:`rocDecode <rocdecode:index>`,0.8.0,0.8.0,0.6.0
      :doc:`rocJPEG <rocjpeg:index>`,0.6.0,0.6.0,N/A
      :doc:`rocPyDecode <rocpydecode:index>`,0.2.0,0.2.0,0.1.0
      :doc:`RPP <rpp:index>`,1.9.1,1.9.1,1.8.0
      ,,,
      COMMUNICATION,.. _commlibs-support-compatibility-matrix:,,
      :doc:`RCCL <rccl:index>`,2.21.5,2.21.5,2.20.5
      ,,,
      MATH LIBS,.. _mathlibs-support-compatibility-matrix:,,
      `half <https://github.com/ROCm/half>`_ ,1.12.0,1.12.0,1.12.0
      :doc:`hipBLAS <hipblas:index>`,2.3.0,2.3.0,2.2.0
      :doc:`hipBLASLt <hipblaslt:index>`,0.10.0,0.10.0,0.8.0
      :doc:`hipFFT <hipfft:index>`,1.0.17,1.0.17,1.0.14
      :doc:`hipfort <hipfort:index>`,0.5.1,0.5.1,0.4.0
      :doc:`hipRAND <hiprand:index>`,2.11.1,2.11.1,2.11.0
      :doc:`hipSOLVER <hipsolver:index>`,2.3.0,2.3.0,2.2.0
      :doc:`hipSPARSE <hipsparse:index>`,3.1.2,3.1.2,3.1.1
      :doc:`hipSPARSELt <hipsparselt:index>`,0.2.2,0.2.2,0.2.1
      :doc:`rocALUTION <rocalution:index>`,3.2.1,3.2.1,3.2.0
      :doc:`rocBLAS <rocblas:index>`,4.3.0,4.3.0,4.2.0
      :doc:`rocFFT <rocfft:index>`,1.0.31,1.0.31,1.0.28
      :doc:`rocRAND <rocrand:index>`,3.2.0,3.2.0,3.1.0
      :doc:`rocSOLVER <rocsolver:index>`,3.27.0,3.27.0,3.26.0
      :doc:`rocSPARSE <rocsparse:index>`,3.3.0,3.3.0,3.2.0
      :doc:`rocWMMA <rocwmma:index>`,1.6.0,1.6.0,1.5.0
      :doc:`Tensile <tensile:src/index>`,4.42.0,4.42.0,4.41.0
      ,,,
      PRIMITIVES,.. _primitivelibs-support-compatibility-matrix:,,
      :doc:`hipCUB <hipcub:index>`,3.3.0,3.3.0,3.2.0
      :doc:`hipTensor <hiptensor:index>`,1.4.0,1.4.0,1.3.0
      :doc:`rocPRIM <rocprim:index>`,3.3.0,3.3.0,3.2.0
      :doc:`rocThrust <rocthrust:index>`,3.3.0,3.3.0,3.0.1
      ,,,
      SUPPORT LIBS,,,
      `hipother <https://github.com/ROCm/hipother>`_,6.3.42134,6.3.42134,6.2.41133
      `rocm-core <https://github.com/ROCm/rocm-core>`_,6.3.3,6.3.2,6.2.0
      `ROCT-Thunk-Interface <https://github.com/ROCm/ROCT-Thunk-Interface>`_,N/A [#ROCT-rocr]_,N/A [#ROCT-rocr]_,20240607.1.4246
      ,,,
      SYSTEM MGMT TOOLS,.. _tools-support-compatibility-matrix:,,
      :doc:`AMD SMI <amdsmi:index>`,24.7.1,24.7.1,24.6.2
      :doc:`ROCm Data Center Tool <rdc:index>`,0.3.0,0.3.0,0.3.0
      :doc:`rocminfo <rocminfo:index>`,1.0.0,1.0.0,1.0.0
      :doc:`ROCm SMI <rocm_smi_lib:index>`,7.4.0,7.4.0,7.3.0
      :doc:`ROCm Validation Suite <rocmvalidationsuite:index>`,1.1.0,1.1.0,1.0.60200
      ,,,
      PERFORMANCE TOOLS,,,
      :doc:`ROCm Bandwidth Test <rocm_bandwidth_test:index>`,1.4.0,1.4.0,1.4.0
      :doc:`ROCm Compute Profiler <rocprofiler-compute:index>`,3.0.0,3.0.0,2.0.1
      :doc:`ROCm Systems Profiler <rocprofiler-systems:index>`,0.1.2,0.1.1,1.11.2
      :doc:`ROCProfiler <rocprofiler:index>`,2.0.60303,2.0.60302,2.0.60200
      :doc:`ROCprofiler-SDK <rocprofiler-sdk:index>`,0.5.0,0.5.0,0.4.0
      :doc:`ROCTracer <roctracer:index>`,4.1.60303,4.1.60302,4.1.60200
      ,,,
      DEVELOPMENT TOOLS,,,
      :doc:`HIPIFY <hipify:index>`,18.0.0.25012,18.0.0.25012,18.0.0.24232
      :doc:`ROCm CMake <rocmcmakebuildtools:index>`,0.14.0,0.14.0,0.13.0
      :doc:`ROCdbgapi <rocdbgapi:index>`,0.77.0,0.77.0,0.76.0
      :doc:`ROCm Debugger (ROCgdb) <rocgdb:index>`,15.2.0,15.2.0,14.2.0
      `rocprofiler-register <https://github.com/ROCm/rocprofiler-register>`_,0.4.0,0.4.0,0.4.0
      :doc:`ROCr Debug Agent <rocr_debug_agent:index>`,2.0.3,2.0.3,2.0.3
      ,,,
      COMPILERS,.. _compilers-support-compatibility-matrix:,,
      `clang-ocl <https://github.com/ROCm/clang-ocl>`_,N/A,N/A,N/A
      :doc:`hipCC <hipcc:index>`,1.1.1,1.1.1,1.1.1
      `Flang <https://github.com/ROCm/flang>`_,18.0.0.25012,18.0.0.25012,18.0.0.24232
      :doc:`llvm-project <llvm-project:index>`,18.0.0.25012,18.0.0.25012,18.0.0.24232
      `OpenMP <https://github.com/ROCm/llvm-project/tree/amd-staging/openmp>`_,18.0.0.25012,18.0.0.25012,18.0.0.24232
      ,,,
      RUNTIMES,.. _runtime-support-compatibility-matrix:,,
      :doc:`AMD CLR <hip:understand/amd_clr>`,6.3.42134,6.3.42134,6.2.41133
      :doc:`HIP <hip:index>`,6.3.42134,6.3.42134,6.2.41133
      `OpenCL Runtime <https://github.com/ROCm/clr/tree/develop/opencl>`_,2.0.0,2.0.0,2.0.0
      :doc:`ROCr Runtime <rocr-runtime:index>`,1.14.0,1.14.0,1.13.0




.. rubric:: Footnotes

.. [#mi300x] Oracle Linux and Azure Linux are supported only on AMD Instinct MI300X.
.. [#single-node] Debian 12 is supported only on AMD Instinct MI300X for single-node functionality. 
.. [#mi300_620] **For ROCm 6.2.0** - MI300X (gfx942) is supported on listed operating systems *except* Ubuntu 22.04.5 [6.8 HWE] and Ubuntu 22.04.4 [6.5 HWE].
.. [#kfd_support] ROCm provides forward and backward compatibility between the AMD Kernel-mode GPU Driver (KMD) and its user space software for +/- 2 releases. These are the compatibility combinations that are currently supported.
.. [#ROCT-rocr] Starting from ROCm 6.3.0, the ROCT Thunk Interface is included as part of the ROCr runtime package.

.. _OS-kernel-versions:

Operating systems and kernel versions
*************************************

Use this lookup table to confirm which operating system and kernel versions are supported with ROCm.

.. csv-table::
   :header: "OS", "Version", "Kernel"
   :widths: 40, 20, 40
   :stub-columns: 1

   `Ubuntu <https://ubuntu.com/about/release-cycle#ubuntu-kernel-release-cycle>`_, 24.04.2, "6.8 GA, 6.11 HWE"
   , 24.04, "6.8 GA"
   ,,
   `Ubuntu <https://ubuntu.com/about/release-cycle#ubuntu-kernel-release-cycle>`_, 22.04.5, "5.15 GA, 6.8 HWE"
   , 22.04.4, "5.15 GA, 6.5 HWE"
   ,,
   `Red Hat Enterprise Linux (RHEL 9) <https://access.redhat.com/articles/3078#RHEL9>`_, 9.5, 5.14.0
   ,9.4, 5.14.0
   ,9.3, 5.14.0
   ,,
   `Red Hat Enterprise Linux (RHEL 8) <https://access.redhat.com/articles/3078#RHEL8>`_, 8.10, 4.18.0
   ,8.9, 4.18.0
   ,,
   `SUSE Linux Enterprise Server (SLES) <https://www.suse.com/support/kb/doc/?id=000019587#SLE15SP4>`_, 15 SP6, 6.4.0
   ,15 SP5, 5.14.21
   ,,
   `Oracle Linux <https://blogs.oracle.com/scoter/post/oracle-linux-and-unbreakable-enterprise-kernel-uek-releases>`_, 8.10, 5.15.0
   ,8.9, 5.15.0
   ,,
   `Debian <https://www.debian.org/download>`_,12, 6.1
   ,,
   `Azure Linux <https://techcommunity.microsoft.com/blog/linuxandopensourceblog/azure-linux-3-0-now-in-preview-on-azure-kubernetes-service-v1-31/4287229>`_,3.0, 6.6
   ,,

..
   Footnotes and ref anchors in below historical tables should be appended with "-past-60", to differentiate from the
   footnote references in the above, latest, compatibility matrix.  It also allows to easily find & replace.
   An easy way to work is to download the historical.CSV file, and update open it in excel. Then when content is ready,
   delete the columns you don't need, to build the current compatibility matrix to use in above table.  Find & replace all
   instances of "-past-60" to make it ready for above table.


.. _past-rocm-compatibility-matrix:

Past versions of ROCm compatibility matrix
***************************************************

Expand for full historical view of:

.. dropdown:: ROCm 6.0 - Present

   You can `download the entire .csv <../downloads/compatibility-matrix-historical-6.0.csv>`_ for offline reference.

   .. csv-table::
      :file: compatibility-matrix-historical-6.0.csv
      :header-rows: 1
      :stub-columns: 1

   .. rubric:: Footnotes

   .. [#mi300x-past-60] Oracle Linux and Azure Linux are supported only on AMD Instinct MI300X.
   .. [#single-node-past-60] Debian 12 is supported only on AMD Instinct MI300X for single-node functionality. 
   .. [#mi300_624-past-60] **For ROCm 6.2.4** - MI300X (gfx942) is supported on listed operating systems *except* Ubuntu 22.04.5 [6.8 HWE] and Ubuntu 22.04.4 [6.5 HWE].
   .. [#mi300_622-past-60] **For ROCm 6.2.2** - MI300X (gfx942) is supported on listed operating systems *except* Ubuntu 22.04.5 [6.8 HWE] and Ubuntu 22.04.4 [6.5 HWE].
   .. [#mi300_621-past-60] **For ROCm 6.2.1** - MI300X (gfx942) is supported on listed operating systems *except* Ubuntu 22.04.5 [6.8 HWE] and Ubuntu 22.04.4 [6.5 HWE].
   .. [#mi300_620-past-60] **For ROCm 6.2.0** - MI300X (gfx942) is supported on listed operating systems *except* Ubuntu 22.04.5 [6.8 HWE] and Ubuntu 22.04.4 [6.5 HWE].
   .. [#mi300_612-past-60] **For ROCm 6.1.2** - MI300A (gfx942) is supported on Ubuntu 22.04.4, RHEL 9.4, RHEL 9.3, RHEL 8.9, and SLES 15 SP5. MI300X (gfx942) is only supported on Ubuntu 22.04.4 and Oracle Linux.
   .. [#mi300_611-past-60] **For ROCm 6.1.1** - MI300A (gfx942) is supported on Ubuntu 22.04.4, RHEL 9.4, RHEL 9.3, RHEL 8.9, and SLES 15 SP5. MI300X (gfx942) is only supported on Ubuntu 22.04.4 and Oracle Linux.
   .. [#mi300_610-past-60] **For ROCm 6.1.0** - MI300A (gfx942) is supported on Ubuntu 22.04.4, RHEL 9.4, RHEL 9.3, RHEL 8.9, and SLES 15 SP5. MI300X (gfx942) is only supported on Ubuntu 22.04.4.
   .. [#mi300_602-past-60] **For ROCm 6.0.2** - MI300A (gfx942) is supported on Ubuntu 22.04.3, RHEL 8.9, and SLES 15 SP5. MI300X (gfx942) is only supported on Ubuntu 22.04.3.
   .. [#mi300_600-past-60] **For ROCm 6.0.0** - MI300A (gfx942) is supported on Ubuntu 22.04.3, RHEL 8.9, and SLES 15 SP5. MI300X (gfx942) is only supported on Ubuntu 22.04.3.
   .. [#kfd_support-past-60] ROCm provides forward and backward compatibility between the AMD Kernel-mode GPU Driver (KMD) and its user space software for +/- 2 releases. These are the compatibility combinations that are currently supported.
   .. [#ROCT-rocr-past-60] Starting from ROCm 6.3.0, the ROCT Thunk Interface is included as part of the ROCr runtime package.
