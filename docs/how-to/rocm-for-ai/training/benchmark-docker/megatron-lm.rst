:orphan:

.. meta::
   :description: How to train a model using Megatron-LM for ROCm.
   :keywords: ROCm, AI, LLM, train, Megatron-LM, megatron, Llama, tutorial, docker, torch

******************************************
Training a model with Megatron-LM for ROCm
******************************************

The Megatron-LM framework for ROCm is a specialized fork of the robust Megatron-LM,
designed to enable efficient training of large-scale language models on AMD
GPUs. By leveraging AMD Instinct™ MI300X series accelerators, Megatron-LM delivers
enhanced scalability, performance, and resource utilization for AI workloads.
It is purpose-built to support models like Llama 2, Llama 3, Llama 3.1, and
DeepSeek, enabling developers to train next-generation AI models more
efficiently. See the GitHub repository at `<https://github.com/ROCm/Megatron-LM>`__.

AMD provides a ready-to-use Docker image for MI300X accelerators containing
essential components, including PyTorch, ROCm libraries, and Megatron-LM
utilities. It contains the following software components to accelerate training
workloads:

+--------------------------+--------------------------------+
| Software component       | Version                        |
+==========================+================================+
| ROCm                     | 6.3.0                          |
+--------------------------+--------------------------------+
| PyTorch                  | 2.7.0a0+git637433              |
+--------------------------+--------------------------------+
| Python                   | 3.10                           |
+--------------------------+--------------------------------+
| Transformer Engine       | 1.11                           |
+--------------------------+--------------------------------+
| Flash Attention          | 3.0.0                          |
+--------------------------+--------------------------------+
| hipBLASLt                | git258a2162                    |
+--------------------------+--------------------------------+
| Triton                   | 3.1                            |
+--------------------------+--------------------------------+

Supported features and models
=============================

Megatron-LM provides the following key features to train large language models efficiently:

- Transformer Engine (TE)

- APEX

- GEMM tuning

- Torch.compile

- 3D parallelism: TP + SP + CP

- Distributed optimizer

- Flash Attention (FA) 3

- Fused kernels

- Pre-training

.. _amd-megatron-lm-model-support:

The following models are pre-optimized for performance on the AMD Instinct MI300X accelerator.

* Llama 2 7B

* Llama 2 70B

* Llama 3 8B

* Llama 3 70B

* Llama 3.1 8B

* Llama 3.1 70B

* DeepSeek-V2-Lite

.. note::

   Some models, such as Llama 3, require an external license agreement through
   a third party (for example, Meta).

System validation
=================

If you have already validated your system settings, skip this step. Otherwise,
complete the :ref:`system validation and optimization steps <train-a-model-system-validation>`
to set up your system before starting training.

Disable NUMA auto-balancing
---------------------------

Generally, application performance can benefit from disabling NUMA auto-balancing. However,
it might be detrimental to performance with certain types of workloads.

Run the command ``cat /proc/sys/kernel/numa_balancing`` to check your current NUMA (Non-Uniform
Memory Access) settings. Output ``0`` indicates this setting is disabled. If there is no output or
the output is ``1``, run the following command to disable NUMA auto-balancing.

.. code-block:: shell

   sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'

See :ref:`mi300x-disable-numa` for more information.

.. _mi300x-amd-megatron-lm-training:

Environment setup
=================

The pre-built ROCm Megatron-LM environment allows users to quickly validate system performance, conduct
training benchmarks, and achieve superior performance for models like Llama 3.1, Llama 2, and DeepSeek V2.

Use the following instructions to set up the environment, configure the script to train models, and
reproduce the benchmark results on the MI300X accelerators with the AMD Megatron-LM Docker
image.

.. _amd-megatron-lm-requirements:
 
Download the Docker image
-------------------------

1. Use the following command to pull the Docker image from Docker Hub.

   .. code-block:: shell

      docker pull rocm/megatron-lm:v25.3

2. Launch the Docker container.

   .. code-block:: shell

      docker run -it --device /dev/dri --device /dev/kfd --network host --ipc host --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged -v $HOME:$HOME -v  $HOME/.ssh:/root/.ssh --shm-size 64G --name megatron_training_env rocm/megatron-lm:v25.3

3. Use these commands if you exit the ``megatron_training_env`` container and need to return to it.

   .. code-block:: shell

      docker start megatron_training_env
      docker exec -it megatron_training_env bash

The Docker container includes a pre-installed, verified version of Megatron-LM from the `release branch <https://github.com/ROCm/Megatron-LM/tree/megatron_release_v25.3>`_.

.. _amd-megatron-lm-environment-setup:

Configuration scripts
---------------------

.. tab-set::

   .. tab-item:: Llama
      :sync: llama

      If you're working with Llama 2 7B or Llama 2 70 B, use the ``train_llama2.sh`` configuration
      script in the ``examples/llama`` directory of
      `<https://github.com/ROCm/Megatron-LM/tree/megatron_release_v25.3/examples/llama>`__.
      Likewise, if you're working with Llama 3 or Llama 3.1, then use ``train_llama3.sh`` and update
      the configuration script accordingly.

   .. tab-item:: DeepSeek V2
      :sync: deepseek

      Use the ``train_deepseek_v2.sh`` configuration script in the ``examples/deepseek_v2``
      directory of
      `<https://github.com/ROCm/Megatron-LM/tree/megatron_release_v25.3/examples/deepseek_v2>`__
      and update the configuration script accordingly.

Network interface
^^^^^^^^^^^^^^^^^

.. tab-set::

   .. tab-item:: Llama
      :sync: llama

      To avoid connectivity issues in multi-node deployments, ensure the correct network interface
      is set in your training scripts.

      1. Run the following command (outside the container) to find the active network interface on your system.

         .. code-block:: shell

            ip a

      2. Update the ``NCCL_SOCKET_IFNAME`` and ``GLOO_SOCKET_IFNAME`` variables with your system’s network interface. For
         example:

         .. code-block:: shell

            export NCCL_SOCKET_IFNAME=ens50f0np0

            export GLOO_SOCKET_IFNAME=ens50f0np0

Dataset options
^^^^^^^^^^^^^^^

.. tab-set::

   .. tab-item:: Llama
      :sync: llama

      You can use either mock data or real data for training.

      * Mock data can be useful for testing and validation. Use the ``MOCK_DATA`` variable to toggle between mock and real data. The default
        value is ``1`` for enabled.

        .. code-block:: bash

           MOCK_DATA=1

      * If you're using a real dataset, update the ``DATA_PATH`` variable to point to the location of your dataset.

        .. code-block:: bash

           MOCK_DATA=0

           DATA_PATH=${DATA_PATH:-"/data/bookcorpus_text_sentence"}  # Change to where your dataset is stored

        Ensure that the files are accessible inside the Docker container.

   .. tab-item:: DeepSeek V2
      :sync: deepseek

      If you don't already have the dataset, download the DeepSeek dataset using the following
      commands:

      .. code-block:: shell

         mkdir deepseek-datasets
         cd deepseek-datasets
         wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-datasets/SlimPajama.json
         wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-datasets/alpaca_zh-train.json
         wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-datasets/alpaca_zh-valid.json
         wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-datasets/mmap_deepseekv2_datasets_text_document.bin
         wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/deepseek-datasets/mmap_deepseekv2_datasets_text_document.idx

      You can use either mock data or real data for training.

      * Mock data can be useful for testing and validation. Use the ``MOCK_DATA`` variable to toggle between mock and real data. The default
        value is ``1`` for enabled.

        .. code-block:: bash

           MOCK_DATA=1

      * If you're using a real dataset, update the ``DATA_DIR`` variable to point to the location of your dataset.

        .. code-block:: bash

           MOCK_DATA=0

           DATA_DIR="/root/data/deepseek-datasets"  # Change to where your dataset is stored

        Ensure that the files are accessible inside the Docker container.

Tokenizer
^^^^^^^^^

Tokenization is the process of converting raw text into tokens that can be processed by the model. For Llama
models, this typically involves sub-word tokenization, where words are broken down into smaller units based on
a fixed vocabulary. The tokenizer is trained along with the model on a large corpus of text, and it learns a
fixed vocabulary that can represent a wide range of text from different domains. This allows Llama models to
handle a variety of input sequences, including unseen words or domain-specific terms.

.. tab-set::

   .. tab-item:: Llama
      :sync: llama

      To train any of the Llama 2 models that :ref:`this Docker image supports <amd-megatron-lm-model-support>`, use the ``Llama2Tokenizer``.

      To train any of Llama 3 and Llama 3.1 models that this Docker image supports, use the ``HuggingFaceTokenizer``.
      Set the Hugging Face model link in the ``TOKENIZER_MODEL`` variable.

      For example, if you're using the Llama 3.1 8B model:

      .. code-block:: shell

         TOKENIZER_MODEL=meta-llama/Llama-3.1-8B

   .. tab-item:: DeepSeek V2
      :sync: deepseek

      To train any of the DeepSeek V2 models that :ref:`this Docker image supports <amd-megatron-lm-model-support>`, use the ``DeepSeekV2Tokenizer``.

Multi-node training
^^^^^^^^^^^^^^^^^^^

.. tab-set::

   .. tab-item:: Llama
      :sync: llama

      If you're running multi-node training, update the following environment variables. They can
      also be passed as command line arguments.

      * Change ``localhost`` to the master node's hostname:

        .. code-block:: shell

           MASTER_ADDR="${MASTER_ADDR:-localhost}"

      * Set the number of nodes you want to train on (for instance, ``2``, ``4``, ``8``):

        .. code-block:: shell

           NNODES="${NNODES:-1}"

      * Set the rank of each node (0 for master, 1 for the first worker node, and so on):

        .. code-block:: shell

           NODE_RANK="${NODE_RANK:-0}"

      * Set ``DATA_CACHE_PATH`` to a common directory accessible by all the nodes (for example, an
        NFS directory) for multi-node runs:

        .. code-block:: shell

           DATA_CACHE_PATH=/root/cache # Set to a common directory for multi-node runs

      * For multi-node runs, make sure the correct network drivers are installed on the nodes. If
        inside a Docker, either install the drivers inside the Docker container or pass the network
        drivers from the host while creating the Docker container.

Start training on AMD Instinct accelerators
===========================================

The prebuilt Megatron-LM with ROCm training environment allows users to quickly validate
system performance, conduct training benchmarks, and achieve superior
performance for models like Llama 3.1 and Llama 2. This container should not be
expected to provide generalized performance across all training workloads. You
can expect the container to perform in the model configurations described in
the following section, but other configurations are not validated by AMD.

Use the following instructions to set up the environment, configure the script
to train models, and reproduce the benchmark results on MI300X series
accelerators with the AMD Megatron-LM Docker image.

.. tab-set::

   .. tab-item:: Llama
      :sync: llama

      .. tab-set::

         .. tab-item:: Single node training
            :sync: single-node

            To run training on a single node, navigate to the Megatron-LM folder and use the
            following command:

            .. code-block:: shell

               TEE_OUTPUT=1 MBS=2 BS=128 TP=1 TE_FP8=1 SEQ_LENGTH=8192 MODEL_SIZE=8 bash examples/llama/train_llama3.sh

         .. tab-item:: Multi-node training
            :sync: multi-node

            To run training on multiple nodes, launch the Docker container on each node. For example, for a two node setup (``NODE0`` as the master node), use these commands.

            * On the master node ``NODE0``:

              .. code-block:: shell

                 TEE_OUTPUT=1 MBS=2 BS=256 TP=1 TE_FP8=1 SEQ_LENGTH=8192 MODEL_SIZE=8 MASTER_ADDR=IP_NODE0 NNODES=2 NODE_RANK=0 bash examples/llama/train_llama3.sh

            * On the worker node ``NODE1``:

              .. code-block:: shell

                 TEE_OUTPUT=1 MBS=2 BS=256 TP=1 TE_FP8=1 SEQ_LENGTH=8192 MODEL_SIZE=8 MASTER_ADDR=IP_NODE0 NNODES=2 NODE_RANK=1 bash examples/llama/train_llama3.sh


   .. tab-item:: DeepSeek V2
      :sync: deepseek

      To run the training on a single node, go to ``/Megatron-LM`` folder and use the following command:

      .. code-block:: shell

         cd /workspace/Megatron-LM
         GEMM_TUNING=1 PR=bf16 MBS=4 AC=none bash examples/deepseek_v2/train_deepseekv2.sh

Key options
-----------

.. _amd-megatron-lm-benchmark-test-vars:

The benchmark tests support the following sets of variables:

.. tab-set::

   .. tab-item:: Llama
      :sync: llama

      ``TEE_OUTPUT``
        ``1`` to enable training logs or ``0`` to disable.

      ``TE_FP8``
        ``0`` for BP16 (default) or ``1`` for FP8 GEMMs.

      ``GEMM_TUNING``
        ``1`` to enable GEMM tuning, which boosts performance by using the best GEMM kernels.

      ``USE_FLASH_ATTN``
        ``1`` to enable Flash Attention.

      ``ENABLE_PROFILING``
        ``1`` to enable PyTorch profiling for performance analysis.

      ``transformer-impl``
        ``transformer_engine`` to use the Transformer Engine (TE) or ``local`` to disable TE.

      ``MODEL_SIZE``
        ``8B`` or ``70B`` for Llama 3 and 3.1. ``7B`` or ``70B`` for Llama 2.

      ``TOTAL_ITERS``
        The total number of iterations -- ``10`` by default.

      ``MOCK_DATA``
        ``1`` to use mock data or ``0`` to use real data provided by you.

      ``MBS``
        Micro batch size.

      ``BS``
        Global batch size.

      ``TP``
        Tensor parallel (``1``, ``2``, ``4``, ``8``).

      ``SEQ_LENGTH``
        Input sequence length.

   .. tab-item:: DeepSeek V2
      :sync: deepseek

      ``PR``
        Precision for training. ``bf16`` for BF16 (default) or ``fp8`` for FP8 GEMMs.

      ``GEMM_TUNING``
        ``1`` to enable GEMM tuning, which boosts performance by using the best GEMM kernels.

      ``TOTAL_ITERS``
        The total number of iterations -- ``10`` by default.

      ``MOCK_DATA``
        ``1`` to use mock data or ``0`` to use real data provided by you.

      ``MBS``
        Micro batch size.

      ``GBS``
        Global batch size.

Benchmarking examples
---------------------

.. tab-set::

   .. tab-item:: Llama
      :sync: llama

      .. tab-set::

         .. tab-item:: Single node training
            :sync: single-node

            Use this command to run training with Llama 2 7B model on a single node. You can specify MBS, BS, FP,
            datatype, and so on.

            .. code-block:: bash

               TEE_OUTPUT=1 MBS=5 BS=120 TP=8 TE_FP8=0 NO_TORCH_COMPILE=1
               SEQ_LENGTH=4096 bash examples/llama/train_llama2.sh

            You can find the training logs at the location defined in ``$TRAIN_LOG`` in the :ref:`configuration script <amd-megatron-lm-environment-setup>`.

            See the sample output:

            .. image:: ../../../../data/how-to/rocm-for-ai/llama2-7b-training-log-sample.png
               :width: 800

         .. tab-item:: Multi-node training
            :sync: multi-node

            Launch the Docker container on each node.

            In this example, run training with Llama 2 7B model on 2 nodes with specific MBS, BS, FP, datatype, and
            so on.

            On the master node:

            .. code-block:: bash

               TEE_OUTPUT=1 MBS=4 BS=64 TP=8 TE_FP8=0 NO_TORCH_COMPILE=1
               SEQ_LENGTH=4096 bash examples/llama/train_llama2.sh

            On the worker node:

            .. code-block:: bash

               TEE_OUTPUT=1 MBS=4 BS=64 TP=8 TE_FP8=0 NO_TORCH_COMPILE=1
               SEQ_LENGTH=4096 bash examples/llama/train_llama2.sh

            You can find the training logs at the location defined in ``$TRAIN_LOG`` in the :ref:`configuration script <amd-megatron-lm-environment-setup>`.

            Sample output for 2-node training:

            Master node:

            .. image:: ../../../../data/how-to/rocm-for-ai/2-node-training-master.png
               :width: 800

            Worker node:

            .. image:: ../../../../data/how-to/rocm-for-ai/2-node-training-worker.png
               :width: 800

Previous versions
=================

This table lists previous versions of the ROCm Megatron-LM Docker image for training
performance testing. For detailed information about available models for
benchmarking, see the version-specific documentation.

.. list-table::
   :header-rows: 1
   :stub-columns: 1

   * - ROCm version
     - Megatron-LM version
     - PyTorch version
     - Resources

   * - 6.1
     - 24.12-dev
     - 2.4.0
     - 
       * `Documentation <https://rocm.docs.amd.com/en/docs-6.3.0/how-to/rocm-for-ai/train-a-model.html>`_
       * `Docker Hub <https://hub.docker.com/layers/rocm/megatron-lm/24.12-dev/images/sha256-5818c50334ce3d69deeeb8f589d83ec29003817da34158ebc9e2d112b929bf2e>`_
