.. meta::
   :description: How to deploy your model for AI inference using vLLM and Hugging Face TGI.
   :keywords: ROCm, AI, LLM, train, fine-tune, deploy, FSDP, DeepSpeed, LLaMA, tutorial

********************
Deploying your model
********************

ROCm enables inference and deployment for various classes of models including CNN, RNN, LSTM, MLP, and transformers.
This section focuses on deploying transformers-based LLM models.

ROCm supports vLLM and Hugging Face TGI as major LLM-serving frameworks.

.. _rocm-for-ai-serve-vllm:

Serving using vLLM
==================

vLLM is a fast and easy-to-use library for LLM inference and serving. vLLM officially supports ROCm versions 5.7 and
6.0. AMD is actively working with the vLLM team to improve performance and support later ROCm versions.

See the `GitHub repository <https://github.com/vllm-project/vllm>`_ and `official vLLM documentation
<https://docs.vllm.ai/>`_ for more information.

For guidance on using vLLM with ROCm, refer to `Installation with ROCm
<https://docs.vllm.ai/en/latest/getting_started/amd-installation.html>`_.

vLLM installation
-----------------

vLLM supports two ROCm-capable installation methods. Refer to the official documentation use the following links.

-  `Build from source with Docker
   <https://docs.vllm.ai/en/latest/getting_started/amd-installation.html#build-from-source-docker-rocm>`_ (recommended)

-  `Build from source <https://docs.vllm.ai/en/latest/getting_started/amd-installation.html#build-from-source-rocm>`_

vLLM walkthrough
----------------

Refer to this developer blog for guidance on serving with vLLM `Inferencing and serving with vLLM on AMD GPUs — ROCm
Blogs <https://rocm.blogs.amd.com/artificial-intelligence/vllm/README.html>`_

Validating vLLM performance
---------------------------

ROCm provides a prebuilt optimized Docker image for validating the performance of LLM inference with vLLM 
on the MI300X accelerator. The Docker image includes ROCm, vLLM, PyTorch, and tuning files in the CSV 
format. For more information, see the guide to 
`LLM inference performance testing with vLLM on the AMD Instinct™ MI300X accelerator <https://github.com/ROCm/MAD/blob/develop/benchmark/vllm/README.md>`_ 
on the ROCm GitHub repository.

.. _rocm-for-ai-serve-hugging-face-tgi:

Serving using Hugging Face TGI
==============================

The `Hugging Face Text Generation Inference <https://huggingface.co/docs/text-generation-inference/index>`_
(TGI) library is optimized for serving LLMs with low latency. Refer to the `Quick tour of TGI
<https://huggingface.co/docs/text-generation-inference/quicktour>`_ for more details.

TGI installation
----------------

The easiest way to use Hugging Face TGI with ROCm on AMD Instinct accelerators is to use the official Docker image at
`<https://github.com/huggingface/text-generation-inference/pkgs/container/text-generation-inference>`__.

TGI walkthrough
---------------

#. Set up the LLM server.

   Deploy the Llama2 7B model with TGI using the official Docker image.

   .. code-block:: shell

      model=TheBloke/Llama-2-7B-fp16
      volume=$PWD
      docker run --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 1g -p 8080:80 -v $volume:/data --name tgi_amd ghcr.io/huggingface/text-generation-inference:1.2-rocm --model-id $model

#. Set up the client.

   a. Open another shell session and run the following command to access the server with the client URL.

   .. code-block:: shell

      curl 127.0.0.1:8080/generate \\
      -X POST \\
      -d '{"inputs":"What is Deep
      Learning?","parameters":{"max_new_tokens":20}}' \\
      -H 'Content-Type: application/json'

   b. Access the server with request endpoints.

   .. code-block:: shell

      pip install request
      PYTHONPATH=/usr/lib/python3/dist-packages python requests_model.py

      ``requests_model.py`` should look like:

      .. code-block:: python

         import requests

         headers = {
           "Content-Type": "application/json",
         }

         data = {
            'inputs': 'What is Deep Learning?',
            'parameters': { 'max_new_tokens': 20 },
         }

         response = requests.post('http://127.0.0.1:8080/generate', headers=headers, json=data)

         print(response.json())

vLLM and Hugging Face TGI are robust solutions for anyone looking to deploy LLMs for applications that demand high
performance, low latency, and scalability.

Visit the topics in :doc:`Using ROCm for AI <../index>` to learn about other ROCm-aware solutions for AI development.
