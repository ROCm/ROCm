.. meta::
   :description: How to run models from Hugging Face on AMD GPUs.
   :keywords: ROCm, AI, LLM, Hugging Face, Optimum, Flash Attention, GPTQ, ONNX, tutorial

********************************
Running models from Hugging Face
********************************

`Hugging Face <https://huggingface.co>`_ hosts the world’s largest AI model repository for developers to obtain
transformer models. Hugging Face models and tools significantly enhance productivity, performance, and accessibility in
developing and deploying AI solutions.

This section describes how to run popular community transformer models from Hugging Face on AMD accelerators and GPUs.

.. _rocm-for-ai-hugging-face-transformers:

Using Hugging Face Transformers
-------------------------------

First, `install the Hugging Face Transformers library <https://huggingface.co/docs/transformers/en/installation>`_,
which lets you easily import any of the transformer models into your Python application.

.. code-block:: shell

   pip install transformers

Here is an example of running `GPT2 <https://huggingface.co/openai-community/gpt2>`_:

.. code-block:: python

   from transformers import GPT2Tokenizer, GPT2Model

   tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

   model = GPT2Model.from_pretrained('gpt2')

   text = "Replace me with any text you'd like."

   encoded_input = tokenizer(text, return_tensors='pt')

   output = model(**encoded_input)

Mainstream transformer models are regularly tested on supported hardware platforms. Models derived from those core
models should also function correctly.

Here are some mainstream models to get you started:

- `BERT <https://huggingface.co/bert-base-uncased>`_

- `BLOOM <https://huggingface.co/bigscience/bloom>`_

- `Llama <https://huggingface.co/huggyllama/llama-7b>`_

- `OPT <https://huggingface.co/facebook/opt-66b>`_

- `T5 <https://huggingface.co/t5-base>`_

.. _rocm-for-ai-hugging-face-optimum:

Using Hugging Face with Optimum-AMD
-----------------------------------

Optimum-AMD is the interface between Hugging Face libraries and the ROCm software stack.

For a deeper dive into using Hugging Face libraries on AMD accelerators and GPUs, refer to the
`Optimum-AMD <https://huggingface.co/docs/optimum/main/en/amd/amdgpu/overview>`_ page on Hugging Face for guidance on
using Flash Attention 2, GPTQ quantization and the ONNX Runtime integration.

Hugging Face libraries natively support AMD Instinct accelerators. For other
:doc:`ROCm-capable hardware <rocm-install-on-linux:reference/system-requirements>`, support is currently not
validated, but most features are expected to work without issues.

.. _rocm-for-ai-install-optimum-amd:

Installation
~~~~~~~~~~~~

Install Optimum-AMD using pip.

.. code-block:: shell

   pip install --upgrade --upgrade-strategy eager optimum[amd]

Or, install from source.

.. code-block:: shell

   git clone https://github.com/huggingface/optimum-amd.git
   cd optimum-amd
   pip install -e .

.. _rocm-for-ai-flash-attention:

Flash Attention
---------------

#. Use `the Hugging Face team's example Dockerfile
   <https://github.com/huggingface/optimum-amd/blob/main/docker/transformers-pytorch-amd-gpu-flash/Dockerfile>`_ to use
   Flash Attention with ROCm.

   .. code-block:: shell

      docker build -f Dockerfile -t transformers_pytorch_amd_gpu_flash .
      volume=$PWD
      docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $volume:/workspace --name transformer_amd
      transformers_pytorch_amd_gpu_flash:latest

#. Use Flash Attention 2 with `Transformers
   <https://huggingface.co/docs/transformers/perf_infer_gpu_one#flashattention-2>`_ by adding the
   ``use_flash_attention_2`` parameter to ``from_pretrained()``:

   .. code-block:: python

      import torch
      from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

      tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b")

      with torch.device("cuda"):
        model = AutoModelForCausalLM.from_pretrained(
        "tiiuae/falcon-7b",
        torch_dtype=torch.float16,
        use_flash_attention_2=True,
        )

.. _rocm-for-ai-gptq:

GPTQ
----

To enable `GPTQ <https://arxiv.org/abs/2210.17323>`_, hosted wheels are available for ROCm.

#. First, :ref:`install Optimum-AMD <rocm-for-ai-install-optimum-amd>`.

#. Install AutoGPTQ using pip. Refer to `AutoGPTQ Installation <https://github.com/AutoGPTQ/AutoGPTQ#Installation>`_ for
   in-depth guidance.

   .. code-block:: shell

      pip install auto-gptq --no-build-isolation --extra-index-url https://huggingface.github.io/autogptq-index/whl/rocm573/

   Or, to install from source for AMD accelerators supporting ROCm, specify the ``ROCM_VERSION`` environment variable.

   .. code-block:: shell

      ROCM_VERSION=6.1 pip install -vvv --no-build-isolation -e .


#. Load GPTQ-quantized models in Transformers using the backend `AutoGPTQ library
   <https://github.com/PanQiWei/AutoGPTQ>`_:

   .. code-block:: python

      import torch
      from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

      tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-7B-Chat-GPTQ")

      with torch.device("cuda"):
        model = AutoModelForCausalLM.from_pretrained(
        "TheBloke/Llama-2-7B-Chat-GPTQ",
        torch_dtype=torch.float16,
        )

.. _rocm-for-ai-onnx:

ONNX
----

Hugging Face Optimum also supports the `ONNX Runtime <https://onnxruntime.ai>`_ integration. For ONNX models, usage is
straightforward.

#. Specify the provider argument in the ``ORTModel.from_pretrained()`` method:

   .. code-block:: python

      from optimum.onnxruntime import ORTModelForSequenceClassification
      ..
      ort_model = ORTModelForSequenceClassification.from_pretrained(
      ..
      provider="ROCMExecutionProvider"
      )

#. Try running a `BERT text classification
   <https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english>`_ ONNX model with ROCm:

   .. code-block:: python

      from optimum.onnxruntime import ORTModelForSequenceClassification
      from optimum.pipelines import pipeline
      from transformers import AutoTokenizer
      import onnxruntime as ort

      session_options = ort.SessionOptions()

      session_options.log_severity_level = 0

      ort_model = ORTModelForSequenceClassification.from_pretrained(
         "distilbert-base-uncased-finetuned-sst-2-english",
         export=True,
         provider="ROCMExecutionProvider",
         session_options=session_options
         )

      tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

      pipe = pipeline(task="text-classification", model=ort_model, tokenizer=tokenizer, device="cuda:0")

      result = pipe("Both the music and visual were astounding, not to mention the actors performance.")

