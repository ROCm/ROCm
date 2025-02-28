.. meta::
   :description: Learn how to validate LLM inference performance on MI300X accelerators using AMD MAD and the
                 ROCm vLLM Docker image.
   :keywords: model, MAD, automation, dashboarding, validate

***********************************************************
LLM inference performance validation on AMD Instinct MI300X
***********************************************************

.. _vllm-benchmark-unified-docker:

.. datatemplate:yaml:: /data/how-to/rocm-for-ai/inference/vllm-benchmark-models.yaml

   {% set unified_docker = data.vllm_benchmark.unified_docker.latest %}
   {% set model_groups = data.vllm_benchmark.model_groups %}

   The `ROCm vLLM Docker <{{ unified_docker.docker_hub_url }}>`_ image offers
   a prebuilt, optimized environment for validating large language model (LLM)
   inference performance on the AMD Instinct™ MI300X accelerator. This ROCm vLLM
   Docker image integrates vLLM and PyTorch tailored specifically for the MI300X
   accelerator and includes the following components:

   * `ROCm {{ unified_docker.rocm_version }} <https://github.com/ROCm/ROCm>`_

   * `vLLM {{ unified_docker.vllm_version }} <https://docs.vllm.ai/en/latest>`_

   * `PyTorch {{ unified_docker.pytorch_version }} <https://github.com/pytorch/pytorch>`_

   With this Docker image, you can quickly validate the expected inference
   performance numbers for the MI300X accelerator. This topic also provides tips on
   optimizing performance with popular AI models.

   .. _vllm-benchmark-available-models:

   Available models
   ================

   .. raw:: html

      <div id="vllm-benchmark-ud-params-picker" class="container-fluid">
        <div class="row">
          <div class="col-2 me-2 model-param-head">Model</div>
          <div class="row col-10">
   {% for model_group in model_groups %}
            <div class="col-3 model-param" data-param-k="model-group" data-param-v="{{ model_group.tag }}" tabindex="0">{{ model_group.group }}</div>
   {% endfor %}
          </div>
        </div>

        <div class="row mt-1">
          <div class="col-2 me-2 model-param-head">Model variant</div>
          <div class="row col-10">
   {% for model_group in model_groups %}
      {% set models = model_group.models %}
      {% for model in models %}
         {% if models|length % 3 == 0 %}
            <div class="col-4 model-param" data-param-k="model" data-param-v="{{ model.mad_tag }}" data-param-group="{{ model_group.tag }}" tabindex="0">{{ model.model }}</div>
         {% else %}
            <div class="col-6 model-param" data-param-k="model" data-param-v="{{ model.mad_tag }}" data-param-group="{{ model_group.tag }}" tabindex="0">{{ model.model }}</div>
         {% endif %}
      {% endfor %}
   {% endfor %}
          </div>
        </div>
      </div>

   .. _vllm-benchmark-vllm:

   {% for model_group in model_groups %}
      {% for model in model_group.models %}

   .. container:: model-doc {{model.mad_tag}}

      .. note::

         See the `{{ model.model }} model card on Hugging Face <{{ model.url }}>`_ to learn more about your selected model.
         Some models require access authorization prior to use via an external license agreement through a third party.

      {% endfor %}
   {% endfor %}


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

   2. Download the `ROCm vLLM Docker image <{{ unified_docker.docker_hub_url }}>`_.

      Use the following command to pull the Docker image from Docker Hub.

      .. code-block:: shell

         docker pull {{ unified_docker.pull_tag }}

   Benchmarking
   ============

   Once the setup is complete, choose between two options to reproduce the
   benchmark results:

   .. _vllm-benchmark-mad:

   {% for model_group in model_groups %}
      {% for model in model_group.models %}

   .. container:: model-doc {{model.mad_tag}}

      .. tab-set::

         .. tab-item:: MAD-integrated benchmarking

            Clone the ROCm Model Automation and Dashboarding (`<https://github.com/ROCm/MAD>`__) repository to a local
            directory and install the required packages on the host machine.

            .. code-block:: shell

               git clone https://github.com/ROCm/MAD
               cd MAD
               pip install -r requirements.txt

            Use this command to run the performance benchmark test on the `{{model.model}} <{{ model.url }}>`_ model
            using one GPU with the ``{{model.precision}}`` data type on the host machine.

            .. code-block:: shell

               export MAD_SECRETS_HFTOKEN="your personal Hugging Face token to access gated models"
               python3 tools/run_models.py --tags {{model.mad_tag}} --keep-model-dir --live-output --timeout 28800

            MAD launches a Docker container with the name
            ``container_ci-{{model.mad_tag}}``. The latency and throughput reports of the
            model are collected in the following path: ``~/MAD/reports_{{model.precision}}/``.

            Although the :ref:`available models <vllm-benchmark-available-models>` are preconfigured
            to collect latency and throughput performance data, you can also change the benchmarking
            parameters. See the standalone benchmarking tab for more information.

         .. tab-item:: Standalone benchmarking

            Run the vLLM benchmark tool independently by starting the
            `Docker container <https://hub.docker.com/layers/rocm/vllm/rocm6.3.1_mi300_ubuntu22.04_py3.12_vllm_0.6.6/images/sha256-9a12ef62bbbeb5a4c30a01f702c8e025061f575aa129f291a49fbd02d6b4d6c9>`_
            as shown in the following snippet.

            .. code-block::

               docker pull rocm/vllm:rocm6.3.1_mi300_ubuntu22.04_py3.12_vllm_0.6.6
               docker run -it --device=/dev/kfd --device=/dev/dri --group-add video --shm-size 16G --security-opt seccomp=unconfined --security-opt apparmor=unconfined --cap-add=SYS_PTRACE -v $(pwd):/workspace --env HUGGINGFACE_HUB_CACHE=/workspace --name vllm_v0.6.6 rocm/vllm:rocm6.3.1_mi300_ubuntu22.04_py3.12_vllm_0.6.6

            In the Docker container, clone the ROCm MAD repository and navigate to the
            benchmark scripts directory at ``~/MAD/scripts/vllm``.

            .. code-block::

               git clone https://github.com/ROCm/MAD
               cd MAD/scripts/vllm

            To start the benchmark, use the following command with the appropriate options.

            .. code-block::

               ./vllm_benchmark_report.sh -s $test_option -m {{model.model_repo}} -g $num_gpu -d {{model.precision}}

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

               * - ``$num_gpu``
                 - 1 or 8
                 - Number of GPUs

               * - ``$datatype``
                 - ``float16`` or ``float8``
                 - Data type

            .. note::

               The input sequence length, output sequence length, and tensor parallel (TP) are
               already configured. You don't need to specify them with this script.

            .. note::

               If you encounter the following error, pass your access-authorized Hugging
               Face token to the gated models.

               .. code-block::

                  OSError: You are trying to access a gated repo.

                  # pass your HF_TOKEN
                  export HF_TOKEN=$your_personal_hf_token

            Here are some examples of running the benchmark with various options.

            * Latency benchmark

              Use this command to benchmark the latency of the {{model.model}} model on eight GPUs with the ``{{model.precision}}`` data type.

              .. code-block::

                 ./vllm_benchmark_report.sh -s latency -m {{model.model_repo}} -g 8 -d {{model.precision}}

              Find the latency report at ``./reports_{{model.precision}}_vllm_rocm{{unified_docker.rocm_version}}/summary/{{model.model_repo.split('/', 1)[1] if '/' in model.model_repo else model.model_repo}}_latency_report.csv``.

            * Throughput benchmark

              Use this command to throughput the latency of the {{model.model}} model on eight GPUs with the ``{{model.precision}}`` data type.

              .. code-block:: shell

                 ./vllm_benchmark_report.sh -s latency -m {{model.model_repo}} -g 8 -d {{model.precision}}

              Find the throughput report at ``./reports_{{model.precision}}_vllm_rocm{{unified_docker.rocm_version}}/summary/{{model.model_repo.split('/', 1)[1] if '/' in model.model_repo else model.model_repo}}_throughput_report.csv``.

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
      {% endfor %}
   {% endfor %}

Further reading
===============

- For application performance optimization strategies for HPC and AI workloads,
  including inference with vLLM, see :doc:`../inference-optimization/workload`.

- To learn more about the options for latency and throughput benchmark scripts,
  see `<https://github.com/ROCm/vllm/tree/main/benchmarks>`_.

- To learn more about system settings and management practices to configure your system for
  MI300X accelerators, see :doc:`../../system-optimization/mi300x`.

- To learn how to run LLM models from Hugging Face or your own model, see
  :doc:`Running models from Hugging Face <hugging-face-models>`.

- To learn how to optimize inference on LLMs, see
  :doc:`Inference optimization <../inference-optimization/index>`.

- To learn how to fine-tune LLMs, see
  :doc:`Fine-tuning LLMs <../fine-tuning/index>`.
