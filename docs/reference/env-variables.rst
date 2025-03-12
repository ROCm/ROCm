.. meta::
    :description: Environment variables reference
    :keywords: AMD, ROCm, environment variables, environment, reference, settings

.. role:: cpp(code)
   :language: cpp

.. _env-variables-reference:

*************************************************************
ROCm environment variables
*************************************************************

ROCm provides a set of environment variables that allow users to configure and optimize their development
and runtime experience. These variables define key settings such as installation paths, platform selection,
and runtime behavior for applications running on AMD accelerators and GPUs.

This page outlines commonly used environment variables across different components of the ROCm software stack,
including HIP and ROCR-Runtime. Understanding these variables can help streamline software development and
execution in ROCm-based environments.

Commonly used environment variables
===================================

The table below provides an overview of key environment variables used in the ROCm software stack.
These variables configure various aspects of ROCm, such as specifying installation paths and
selecting the target platform for applications running on AMD accelerators and GPUs.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - Environment variable
      - Value

    * - | ``HIP_DIR``
        | The path of the HIP SDK on Microsoft Windows. This variable is ignored, if ``HIP_PATH`` is set.
      - Default: ``C:/hip``

    * - | ``HIP_PATH``
        | The path of the HIP SDK on Microsoft Windows.
      - Default: ``C:/hip``

    * - | ``HIP_PLATFORM``
        | The platform targeted by HIP. If ``HIP_PLATFORM`` isn't set, then :doc:`HIPCC <hipcc:index>` attempts to auto-detect the platform, if it can find NVCC.
      - ``amd``, ``nvidia``

    * - | ``ROCM_PATH``
        | The path of the installed ROCm software stack on Linux.
      - Default: ``/opt/rocm``

HIP environment variables
=========================

The following tables list the HIP environment variables:

.. remote-content::
   :repo: ROCm/HIP
   :path: docs/data/env_variables_hip.rst
   :default_branch: docs/develop
   :tag_prefix: docs/

ROCR-Runtime environment variables
==================================

The following table lists the ROCR-Runtime environment variables:

.. remote-content::
   :repo: ROCm/ROCR-Runtime
   :path: runtime/docs/data/env_variables.rst
   :default_branch: amd-staging
   :tag_prefix: docs/

HIPCC environment variables
=========================

.. remote-content::
   :repo: ROCm/llvm-project
   :path: amd/hipcc/docs/env.rst
   :default_branch: amd-staging
   :start_line: 10
   :tag_prefix: docs/