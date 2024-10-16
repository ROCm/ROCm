.. meta::
   :description: Learn how to validate LLM inference performance on MI300X accelerators using AMD MAD and a the unified
                 ROCm Docker image.
   :keywords: model, MAD, automation, dashboarding, validate

***********************************************************
LLM inference performance validation on AMD Instinct MI300X
***********************************************************

ROCm offers a prebuilt, optimized Docker image designed for validating large language model (LLM) inference performance
on the AMD Instinct™ MI300X accelerator. This unified Docker image integrates vLLM and PyTorch tailored specifically
for the MI300X accelerator and includes the following components:

* ROCm

* vLLM

* PyTorch

* Tuning files (in CSV format)

The unified Docker image is hosted on Docker Hub at :fab:`docker` `rocm/vllm <https://hub.docker.com/r/rocm/vllm/tags>`_.

For detailed guidance on performance validation and benchmarking, refer to the ROCm Model Automation and Dashboarding
(MAD) documentation at `<https://github.com/ROCm/MAD/tree/develop/benchmark/vllm>`__. The MAD documentation explains how
to reproduce and validate benchmark results using this Docker image through two methods:

* `MAD-integrated benchmarking <https://github.com/ROCm/MAD/tree/develop/benchmark/vllm#standalone-benchmarking>`_

* `Standalone benchmarking <https://github.com/ROCm/MAD/tree/develop/benchmark/vllm#standalone-benchmarking>`_

The documentation also provides helpful resources so that you can get optimal performance with popular AI models.

.. note::

   vLLM is a toolkit and library for large language model (LLM) inference and serving. It deploys the PagedAttention
   algorithm, which reduces memory consumption and increases throughput by leveraging dynamic key and value allocation
   in GPU memory. vLLM also incorporates many recent LLM acceleration and quantization algorithms. In addition, AMD
   implements high-performance custom kernels and modules in vLLM to enhance performance further. See
   :ref:`fine-tuning-llms-vllm` and :ref:`mi300x-vllm-optimization` for more information.
