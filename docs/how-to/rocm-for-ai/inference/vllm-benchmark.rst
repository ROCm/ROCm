.. meta::
   :description: Learn how to validate LLM inference performance on MI300X accelerators using AMD MAD and the
                 ROCm vLLM Docker image.
   :keywords: model, MAD, automation, dashboarding, validate

***********************************************************
LLM inference performance validation on AMD Instinct MI300X
***********************************************************

.. _vllm-benchmark-unified-docker:

The `ROCm vLLM Docker <https://hub.docker.com/r/rocm/vllm/tags>`_ image offers
a prebuilt, optimized environment designed for validating large language model
(LLM) inference performance on the AMD Instinct™ MI300X accelerator. This
ROCm vLLM Docker image integrates vLLM and PyTorch tailored specifically for the
MI300X accelerator and includes the following components:

* `ROCm 6.2.1 <https://github.com/ROCm/ROCm>`_

* `vLLM 0.6.4 <https://docs.vllm.ai/en/latest>`_

* `PyTorch 2.5.0 <https://github.com/pytorch/pytorch>`_

* Tuning files (in CSV format)

With this Docker image, you can quickly validate the expected inference
performance numbers on the MI300X accelerator. This topic also provides tips on
optimizing performance with popular AI models.

.. hlist::
   :columns: 6

   * Llama 3.1 8B

   * Llama 3.1 70B

   * Llama 3.1 405B

   * Llama 2 7B

   * Llama 2 70B

   * Mixtral 8x7B

   * Mixtral 8x22B

   * Mixtral 7B

   * Qwen2 7B

   * Qwen2 72B

   * JAIS 13B

   * JAIS 30B

.. _vllm-benchmark-vllm:

.. note::

   vLLM is a toolkit and library for LLM inference and serving. AMD implements
   high-performance custom kernels and modules in vLLM to enhance performance.
   See :ref:`fine-tuning-llms-vllm` and :ref:`mi300x-vllm-optimization` for
   more information.

Getting started
===============

Use the following procedures to reproduce the benchmark results on an
MI300X accelerator with the prebuilt vLLM Docker image.

.. _vllm-benchmark-get-started:

1. Disable NUMA auto-balancing.

   To optimize performance, disable automatic NUMA balancing. Otherwise, the GPU
   might hang until the periodic balancing is finalized. For more information,
   see :ref:`AMD Instinct MI300X system optimization <mi300x-disable-numa>`.

   .. code-block:: shell

      # disable automatic NUMA balancing
      sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'
      # check if NUMA balancing is disabled (returns 0 if disabled)
      cat /proc/sys/kernel/numa_balancing
      0

2. Download the :ref:`ROCm vLLM Docker image <vllm-benchmark-unified-docker>`.

   Use the following command to pull the Docker image from Docker Hub.

   .. code-block:: shell

      docker pull rocm/vllm:rocm6.2_mi300_ubuntu20.04_py3.9_vllm_0.6.4

Once setup is complete, you can choose between two options to reproduce the
benchmark results:

-  :ref:`MAD-integrated benchmarking <vllm-benchmark-mad>`

-  :ref:`Standalone benchmarking <vllm-benchmark-standalone>`

.. _vllm-benchmark-mad:

MAD-integrated benchmarking
===========================

Clone the ROCm Model Automation and Dashboarding (`<https://github.com/ROCm/MAD>`__) repository to a local
directory and install the required packages on the host machine.

.. code-block:: shell

   git clone https://github.com/ROCm/MAD
   cd MAD
   pip install -r requirements.txt

Use this command to run a performance benchmark test of the Llama 3.1 8B model
on one GPU with ``float16`` data type in the host machine.

.. code-block:: shell

   export MAD_SECRETS_HFTOKEN="your personal Hugging Face token to access gated models"
   python3 tools/run_models.py --tags pyt_vllm_llama-3.1-8b --keep-model-dir --live-output --timeout 28800

ROCm MAD launches a Docker container with the name
``container_ci-pyt_vllm_llama-3.1-8b``. The latency and throughput reports of the
model are collected in the following path: ``~/MAD/reports_float16/``.

Although the following models are preconfigured to collect latency and
throughput performance data, you can also change the benchmarking parameters.
Refer to the :ref:`Standalone benchmarking <vllm-benchmark-standalone>` section.

Available models
----------------

.. hlist::
   :columns: 3

   * ``pyt_vllm_llama-3.1-8b``

   * ``pyt_vllm_llama-3.1-70b``

   * ``pyt_vllm_llama-3.1-405b``

   * ``pyt_vllm_llama-2-7b``

   * ``pyt_vllm_llama-2-70b``

   * ``pyt_vllm_mixtral-8x7b``

   * ``pyt_vllm_mixtral-8x22b``

   * ``pyt_vllm_mistral-7b``

   * ``pyt_vllm_qwen2-7b``

   * ``pyt_vllm_qwen2-72b``

   * ``pyt_vllm_jais-13b``

   * ``pyt_vllm_jais-30b``

   * ``pyt_vllm_llama-3.1-8b_fp8``

   * ``pyt_vllm_llama-3.1-70b_fp8``

   * ``pyt_vllm_llama-3.1-405b_fp8``

   * ``pyt_vllm_mixtral-8x7b_fp8``

   * ``pyt_vllm_mixtral-8x22b_fp8``

.. _vllm-benchmark-standalone:

Standalone benchmarking
=======================

You can run the vLLM benchmark tool independently by starting the
:ref:`Docker container <vllm-benchmark-get-started>` as shown in the following
snippet.

.. code-block::

   docker pull rocm/vllm:rocm6.2_mi300_ubuntu20.04_py3.9_vllm_0.6.4
   docker run -it --device=/dev/kfd --device=/dev/dri --group-add video --shm-size 128G --security-opt seccomp=unconfined --security-opt apparmor=unconfined --cap-add=SYS_PTRACE -v $(pwd):/workspace --env HUGGINGFACE_HUB_CACHE=/workspace --name vllm_v0.6.4 rocm/vllm:rocm6.2_mi300_ubuntu20.04_py3.9_vllm_0.6.4

In the Docker container, clone the ROCm MAD repository and navigate to the
benchmark scripts directory at ``~/MAD/scripts/vllm``.

.. code-block::

   git clone https://github.com/ROCm/MAD
   cd MAD/scripts/vllm

Command
-------

To start the benchmark, use the following command with the appropriate options.
See :ref:`Options <vllm-benchmark-standalone-options>` for the list of
options and their descriptions.

.. code-block:: shell

   ./vllm_benchmark_report.sh -s $test_option -m $model_repo -g $num_gpu -d $datatype

See the :ref:`examples <vllm-benchmark-run-benchmark>` for more information.

.. note::

   The input sequence length, output sequence length, and tensor parallel (TP) are
   already configured. You don't need to specify them with this script.

.. note::

   If you encounter the following error, pass your access-authorized Hugging
   Face token to the gated models.

   .. code-block:: shell

      OSError: You are trying to access a gated repo.

      # pass your HF_TOKEN
      export HF_TOKEN=$your_personal_hf_token

.. _vllm-benchmark-standalone-options:

Options
-------

.. list-table::
   :header-rows: 1
   :align: center

   * - Name
     - Options
     - Description

   * - ``$test_option``
     - latency
     - Measure decoding token latency

   * -
     - throughput
     - Measure token generation throughput

   * -
     - all
     - Measure both throughput and latency

   * - ``$model_repo``
     - ``meta-llama/Meta-Llama-3.1-8B-Instruct``
     - Llama 3.1 8B

   * - (``float16``)
     - ``meta-llama/Meta-Llama-3.1-70B-Instruct``
     - Llama 3.1 70B

   * -
     - ``meta-llama/Meta-Llama-3.1-405B-Instruct``
     - Llama 3.1 405B

   * -
     - ``meta-llama/Llama-2-7b-chat-hf``
     - Llama 2 7B

   * -
     - ``meta-llama/Llama-2-70b-chat-hf``
     - Llama 2 70B

   * -
     - ``mistralai/Mixtral-8x7B-Instruct-v0.1``
     - Mixtral 8x7B

   * -
     - ``mistralai/Mixtral-8x22B-Instruct-v0.1``
     - Mixtral 8x22B

   * -
     - ``mistralai/Mistral-7B-Instruct-v0.3``
     - Mixtral 7B

   * -
     - ``Qwen/Qwen2-7B-Instruct``
     - Qwen2 7B

   * -
     - ``Qwen/Qwen2-72B-Instruct``
     - Qwen2 72B

   * -
     - ``core42/jais-13b-chat``
     - JAIS 13B

   * -
     - ``core42/jais-30b-chat-v3``
     - JAIS 30B

   * - ``$model_repo``
     - ``amd/Meta-Llama-3.1-8B-Instruct-FP8-KV``
     - Llama 3.1 8B

   * - (``float8``)
     - ``amd/Meta-Llama-3.1-70B-Instruct-FP8-KV``
     - Llama 3.1 70B

   * -
     - ``amd/Meta-Llama-3.1-405B-Instruct-FP8-KV``
     - Llama 3.1 405B

   * -
     - ``amd/Mixtral-8x7B-Instruct-v0.1-FP8-KV``
     - Mixtral 8x7B

   * -
     - ``amd/Mixtral-8x22B-Instruct-v0.1-FP8-KV``
     - Mixtral 8x22B

   * - ``$num_gpu``
     - 1 or 8
     - Number of GPUs

   * - ``$datatype``
     - ``float16`` or ``float8``
     - Data type

.. _vllm-benchmark-run-benchmark:

Running the benchmark on the MI300X accelerator
-----------------------------------------------

Here are some examples of running the benchmark with various options.
See :ref:`Options <vllm-benchmark-standalone-options>` for the list of
options and their descriptions.

Example 1: latency benchmark
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 
Use this command to benchmark the latency of the Llama 3.1 8B model on one GPU with the ``float16`` and ``float8`` data types.

.. code-block::

   ./vllm_benchmark_report.sh -s latency -m meta-llama/Meta-Llama-3.1-8B-Instruct -g 1 -d float16
   ./vllm_benchmark_report.sh -s latency -m amd/Meta-Llama-3.1-8B-Instruct-FP8-KV -g 1 -d float8

Find the latency reports at:

- ``./reports_float16/summary/Meta-Llama-3.1-8B-Instruct_latency_report.csv``

- ``./reports_float8/summary/Meta-Llama-3.1-8B-Instruct-FP8-KV_latency_report.csv``

Example 2: throughput benchmark
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use this command to benchmark the throughput of the Llama 3.1 8B model on one GPU with the ``float16`` and ``float8`` data types.

.. code-block:: shell

   ./vllm_benchmark_report.sh -s throughput -m meta-llama/Meta-Llama-3.1-8B-Instruct -g 1 -d float16
   ./vllm_benchmark_report.sh -s throughput -m amd/Meta-Llama-3.1-8B-Instruct-FP8-KV -g 1 -d float8

Find the throughput reports at:

- ``./reports_float16/summary/Meta-Llama-3.1-8B-Instruct_throughput_report.csv``

- ``./reports_float8/summary/Meta-Llama-3.1-8B-Instruct-FP8-KV_throughput_report.csv``

.. raw:: html

   <style>
   mjx-container[jax="CHTML"][display="true"] {
       text-align: left;
       margin: 0;
   }
   </style>

.. note::

   Throughput is calculated as:

   - .. math:: throughput\_tot = requests \times (\mathsf{\text{input lengths}} + \mathsf{\text{output lengths}}) / elapsed\_time

   - .. math:: throughput\_gen = requests \times \mathsf{\text{output lengths}} / elapsed\_time

Further reading
===============

- For application performance optimization strategies for HPC and AI workloads,
  including inference with vLLM, see :doc:`../inference-optimization/workload`.

- To learn more about the options for latency and throughput benchmark scripts,
  see `<https://github.com/ROCm/vllm/tree/main/benchmarks>`_.

- To learn more about system settings and management practices to configure your system for
  MI300X accelerators, see :doc:`../../system-optimization/mi300x`.

- To learn how to run LLM models from Hugging Face or your own model, see
  :doc:`Using ROCm for AI <../index>`.

- To learn how to optimize inference on LLMs, see
  :doc:`Inference optimization <../inference-optimization/index>`.

- To learn how to fine-tune LLMs, see
  :doc:`Fine-tuning LLMs <../fine-tuning/index>`.

- To compare with the previous version of the ROCm vLLM Docker image for performance validation, refer to
  `LLM inference performance validation on AMD Instinct MI300X (ROCm 6.2.0) <https://rocm.docs.amd.com/en/docs-6.2.0/how-to/performance-validation/mi300x/vllm-benchmark.html>`_.

