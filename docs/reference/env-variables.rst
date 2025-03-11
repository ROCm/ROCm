.. meta::
    :description: Environment variables reference
    :keywords: AMD, ROCm, environment variables, environment, reference

.. role:: cpp(code)
   :language: cpp

.. _env-variables-reference:

*************************************************************
ROCm environment variables
*************************************************************

The following table lists the most commonly used environment variables in the ROCm software stack. These variables help to perform simple tasks such as building a ROCm library or running applications on AMDGPUs.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Value**

    * - | ``HIP_PATH``
        | The path of the HIP SDK on Microsoft Windows.
      - Default: ``C:/hip``

    * - | ``HIP_DIR``
        | The path of the HIP SDK on Microsoft Windows. This variable is ignored, if ``HIP_PATH`` is set.
      - Default: ``C:/hip``

    * - | ``ROCM_PATH``
        | The path of the installed ROCm software stack on Linux.
      - Default: ``/opt/rocm``

    * - | ``HIP_PLATFORM``
        | The platform targeted by HIP. If ``HIP_PLATFORM`` is not set, then HIPCC attempts to auto-detect the platform, if it can find NVCC.
      - ``amd``, ``nvidia``

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