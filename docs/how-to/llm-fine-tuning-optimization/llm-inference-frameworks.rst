.. meta::
   :description: How to fine-tune LLMs with ROCm
   :keywords: ROCm, LLM, fine-tuning, usage, tutorial, inference, vLLM, TGI, text generation inference

************************
LLM inference frameworks
************************

This section discusses how to implement `vLLM <https://docs.vllm.ai/en/latest>`_ and `Hugging Face TGI
<https://huggingface.co/docs/text-generation-inference/en/index>`_ using
:doc:`single-accelerator <single-gpu-fine-tuning-and-inference>` and
:doc:`multi-accelerator <multi-gpu-fine-tuning-and-inference>` systems.

.. _fine-tuning-llms-vllm:

vLLM inference
==============

vLLM is renowned for its PagedAttention algorithm that can reduce memory consumption and increase throughput thanks to
its paging scheme. Instead of allocating GPU high-bandwidth memory (HBM) for the maximum output token lengths of the
models, the paged attention of vLLM allocates GPU HBM dynamically for its actual decoding lengths. This paged attention
is also effective when multiple requests share the same key and value contents for a large value of beam search or
multiple parallel requests.

vLLM also incorporates many modern LLM acceleration and quantization algorithms, such as Flash Attention, HIP and CUDA
graphs, tensor parallel multi-GPU, GPTQ, AWQ, and token speculation.

Installing vLLM
---------------

.. _fine-tuning-llms-vllm-rocm-docker-image:

1. Run the following commands to build a Docker image ``vllm-rocm``.

   .. code-block:: shell

      git clone https://github.com/vllm-project/vllm.git
      cd vllm
      docker build -f Dockerfile.rocm -t vllm-rocm .

.. tab-set::

   .. tab-item:: vLLM on a single-accelerator system
      :sync: single

      2. To use vLLM as an API server to serve reference requests, first start a container using the :ref:`vllm-rocm
         Docker image <fine-tuning-llms-vllm-rocm-docker-image>`.

         .. code-block:: shell

            docker run -it \
               --network=host \
               --group-add=video \
               --ipc=host \
               --cap-add=SYS_PTRACE \
               --security-opt seccomp=unconfined \
               --device /dev/kfd \
               --device /dev/dri \
               -v <path/to/model>:/app/model \
               vllm-rocm \
               bash

      3. Inside the container, start the API server to run on a single accelerator on port 8000 using the following command.

         .. code-block:: shell

            python -m vllm.entrypoints.api_server --model /app/model --dtype float16 --port 8000 &

         The following log message is displayed in your command line indicates that the server is listening for requests.

         .. image:: ../../data/how-to/llm-fine-tuning-optimization/vllm-single-gpu-log.png
            :alt: vLLM API server log message
            :align: center

      4. To test, send it a curl request containing a prompt.

         .. code-block:: shell

            curl http://localhost:8000/generate -H "Content-Type: application/json" -d '{"prompt": "What is AMD Instinct?", "max_tokens": 80, "temperature": 0.0 }'

         You should receive a response like the following.

         .. code-block:: text

            {"text":["What is AMD Instinct?\nAmd Instinct is a brand new line of high-performance computing (HPC) processors from Advanced Micro Devices (AMD). These processors are designed to deliver unparalleled performance for HPC workloads, including scientific simulations, data analytics, and machine learning.\nThe Instinct lineup includes a range of processors, from the entry-level Inst"]}

   .. tab-item:: vLLM on a multi-accelerator system
      :sync: multi

      2. To use vLLM as an API server to serve reference requests, first start a container using the :ref:`vllm-rocm
         Docker image <fine-tuning-llms-vllm-rocm-docker-image>`.

         .. code-block:: shell

            docker run -it \
               --network=host \
               --group-add=video \
               --ipc=host \
               --cap-add=SYS_PTRACE \
               --security-opt seccomp=unconfined \
               --device /dev/kfd \
               --device /dev/dri \
               -v <path/to/model>:/app/model \
               vllm-rocm \
               bash


      3. To run API server on multiple GPUs, use the ``-tp``  or ``--tensor-parallel-size``  parameter. For example, to use two
         GPUs, start the API server using the following command.

         .. code-block:: shell

            python -m vllm.entrypoints.api_server --model /app/model --dtype float16 -tp 2 --port 8000 &

      4. To run multiple instances of API Servers, specify different ports for each server, and use ``ROCR_VISIBLE_DEVICES`` to
         isolate each instance to a different accelerator.

         For example, to run two API servers, one on port 8000 using GPU 0 and 1, one on port 8001 using GPU 2 and 3, use a
         a command like the following.

         .. code-block:: shell

            ROCR_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.api_server --model /data/llama-2-7b-chat-hf --dtype float16 –tp 2 --port 8000 &
            ROCR_VISIBLE_DEVICES=2,3 python -m vllm.entrypoints.api_server --model /data/llama-2-7b-chat-hf --dtype float16 –tp 2--port 8001 &

      5. To test, send it a curl request containing a prompt.

         .. code-block:: shell

            curl http://localhost:8000/generate -H "Content-Type: application/json" -d '{"prompt": "What is AMD Instinct?", "max_tokens": 80, "temperature": 0.0 }'

         You should receive a response like the following.

         .. code-block:: text

            {"text":["What is AMD Instinct?\nAmd Instinct is a brand new line of high-performance computing (HPC) processors from Advanced Micro Devices (AMD). These processors are designed to deliver unparalleled performance for HPC workloads, including scientific simulations, data analytics, and machine learning.\nThe Instinct lineup includes a range of processors, from the entry-level Inst"]}

.. seealso::

   See :ref:`mi300x-vllm-optimization` for performance optimization tips.

   ROCm provides a prebuilt optimized Docker image for validating the performance of LLM inference with vLLM
   on the MI300X accelerator. The Docker image includes ROCm, vLLM, PyTorch, and tuning files in CSV
   format. For more information, see :doc:`/how-to/performance-validation/mi300x/vllm-benchmark`.

.. _fine-tuning-llms-tgi:

Hugging Face TGI
================

Text Generation Inference (TGI) is LLM serving framework from Hugging
Face, and it also supports the majority of high-performance LLM
acceleration algorithms such as Flash Attention, Paged Attention,
CUDA/HIP graph, tensor parallel multi-GPU, GPTQ, AWQ, and token
speculation.

.. tip::

   In addition to LLM serving capability, TGI also provides the `Text Generation Inference benchmarking tool
   <https://github.com/huggingface/text-generation-inference/blob/main/benchmark/README.md>`_.

Install TGI
-----------

1. Launch the TGI Docker container in the host machine.

   .. code-block:: shell

      docker run --name tgi --rm -it --cap-add=SYS_PTRACE --security-opt seccomp=unconfined
      --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 256g
      --net host -v $PWD:/data
      --entrypoint "/bin/bash"
      --env HUGGINGFACE_HUB_CACHE=/data
      ghcr.io/huggingface/text-generation-inference:latest-rocm

.. tab-set::

   .. tab-item:: TGI on a single-accelerator system
      :sync: single

      2. Inside the container, launch a model using TGI server on a single accelerator.

         .. code-block:: shell

            export ROCM_USE_FLASH_ATTN_V2_TRITON=True
            text-generation-launcher --model-id NousResearch/Meta-Llama-3-70B --dtype float16 --port 8000 &

      3. To test, send it a curl request containing a prompt.

         .. code-block:: shell

            curl http://localhost:8000/generate_stream -X POST -d '{"inputs":"What is AMD Instinct?","parameters":{"max_new_tokens":20}}' -H 'Content-Type: application/json'

         You should receive a response like the following.

         .. code-block:: shell

            data:{"index":20,"token":{"id":304,"text":" in","logprob":-1.2822266,"special":false},"generated_text":" AMD Instinct is a new family of data center GPUs designed to accelerate the most demanding workloads in","details":null}

   .. tab-item:: TGI on a multi-accelerator system

      2. Inside the container, launch a model using TGI server on multiple accelerators (4 in this case).

         .. code-block:: shell

            export ROCM_USE_FLASH_ATTN_V2_TRITON=True
            text-generation-launcher --model-id NousResearch/Meta-Llama-3-8B --dtype float16 --port 8000 --num-shard 4 &

      3. To test, send it a curl request containing a prompt.

         .. code-block:: shell

            curl http://localhost:8000/generate_stream -X POST -d '{"inputs":"What is AMD Instinct?","parameters":{"max_new_tokens":20}}' -H 'Content-Type: application/json'

         You should receive a response like the following.

         .. code-block:: shell

            data:{"index":20,"token":{"id":304,"text":" in","logprob":-1.2773438,"special":false},"generated_text":" AMD Instinct is a new family of data center GPUs designed to accelerate the most demanding workloads in","details":null}
