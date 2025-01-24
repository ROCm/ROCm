.. meta::
   :description: Model fine-tuning and inference on a multi-GPU system
   :keywords: ROCm, LLM, fine-tuning, usage, tutorial, multi-GPU, distributed, inference, accelerators, PyTorch, HuggingFace, torchtune

*****************************************************
Fine-tuning and inference using multiple accelerators
*****************************************************

This section explains how to fine-tune a model on a multi-accelerator system. See
:doc:`Single-accelerator fine-tuning <single-gpu-fine-tuning-and-inference>` for a single accelerator or GPU setup.

.. _fine-tuning-llms-multi-gpu-env:

Environment setup
=================

This section was tested using the following hardware and software environment.

.. list-table::
   :stub-columns: 1

   * - Hardware
     - 4 AMD Instinct MI300X accelerators

   * - Software
     - ROCm 6.1, Ubuntu 22.04, PyTorch 2.1.2, Python 3.10

   * - Libraries
     - ``transformers`` ``datasets`` ``accelerate`` ``huggingface-hub`` ``peft`` ``trl`` ``scipy``

   * - Base model
     - ``meta-llama/Llama-2-7b-chat-hf``

.. _fine-tuning-llms-multi-gpu-env-setup:

Setting up the base implementation environment
----------------------------------------------

#. Install PyTorch for ROCm. Refer to the
   :doc:`PyTorch installation guide <rocm-install-on-linux:install/3rd-party/pytorch-install>`. For consistent
   installation, it’s recommended to use official ROCm prebuilt Docker images with the framework pre-installed.

#. In the Docker container, check the availability of ROCM-capable accelerators using the following command.

   .. code-block:: shell

      rocm-smi --showproductname

#. Check that your accelerators are available to PyTorch.

   .. code-block:: python

      import torch
      print("Is a ROCm-GPU detected? ", torch.cuda.is_available())
      print("How many ROCm-GPUs are detected? ", torch.cuda.device_count())

   If successful, your output should look like this:

   .. code-block:: shell

      >>> print("Is a ROCm-GPU detected? ", torch.cuda.is_available())
      Is a ROCm-GPU detected?  True
      >>> print("How many ROCm-GPUs are detected? ", torch.cuda.device_count())
      How many ROCm-GPUs are detected?  4

.. tip::

   During training and inference, you can check the memory usage by running the ``rocm-smi`` command in your terminal.
   This tool helps you see shows which accelerators or GPUs are involved.


.. _fine-tuning-llms-multi-gpu-hugging-face-accelerate:

Hugging Face Accelerate for fine-tuning and inference
===========================================================

`Hugging Face Accelerate <https://huggingface.co/docs/accelerate/en/index>`_ is a library that simplifies turning raw
PyTorch code for a single accelerator into code for multiple accelerators for LLM fine-tuning and inference. It is
integrated with `Transformers <https://huggingface.co/docs/transformers/en/index>`_ allowing you to scale your PyTorch
code while maintaining performance and flexibility.

As a brief example of model fine-tuning and inference using multiple GPUs, let's use Transformers and load in the Llama
2 7B model.

Here, let's reuse the code in :ref:`Single-accelerator fine-tuning <fine-tuning-llms-single-gpu-download-model-dataset>`
to load the base model and tokenizer.

Now, it's important to adjust how you load the model. Add the ``device_map`` parameter to your base model configuration.

.. code-block:: python

   ...
   base_model_name = "meta-llama/Llama-2-7b-chat-hf"
   
   # Load base model to GPU memory
   base_model = AutoModelForCausalLM.from_pretrained(
           base_model_name, 
           device_map = "auto",
           trust_remote_code = True)
   ...
   # Run training
   sft_trainer.train()

.. note::

   You can let Accelerate handle the device map computation by setting ``device_map`` to one of the supported options
   (``"auto"``, ``"balanced"``, ``"balanced_low_0"``, ``"sequential"``).

   It's recommended to set the ``device_map`` parameter to ``“auto”`` to allow Accelerate to automatically and
   efficiently allocate the model given the available resources (4 accelerators in this case).

   When you have more GPU memory available than the model size, here is the difference between each ``device_map``
   option:

   * ``"auto"`` and ``"balanced"`` evenly split the model on all available GPUs, making it possible for you to use a
     batch size greater than 1.

   * ``"balanced_low_0"`` evenly splits the model on all GPUs except the first
     one, and only puts on GPU 0 what does not fit on the others. This
     option is great when you need to use GPU 0 for some processing of the
     outputs, like when using the generate function for Transformers
     models.

   * ``"sequential"`` will fit what it can on GPU 0, then move on GPU 1 and so forth. Not all GPUs might be used.

After loading the model in this way, the model is fully ready to use the resources available to it.

.. _fine-tuning-llms-multi-gpu-torchtune:

torchtune for fine-tuning and inference
=============================================

`torchtune <https://pytorch.org/torchtune/main/>`_ is a PyTorch-native library for easy single and multi-accelerator or
GPU model fine-tuning and inference with LLMs.

#. Install torchtune using pip.

   .. code-block:: shell

      # Install torchtune with PyTorch release 2.2.2+
      pip install torchtune
      
      # To confirm that the package is installed correctly
      tune --help

   The output should look like this:

   .. code-block:: shell

      usage: tune [-h] {download,ls,cp,run,validate} ...
      
      Welcome to the TorchTune CLI!
      
      options:
        -h, --help            show this help message and exit
      
      subcommands:
        {download,ls,cp,run,validate}

#. torchtune recipes are designed around easily composable components and workable training loops, with minimal abstraction
   getting in the way of fine-tuning. Run ``tune ls`` to show built-in torchtune configuration recipes.

   .. code-block:: shell

      RECIPE                                   CONFIG
      full_finetune_single_device              llama2/7B_full_low_memory
                                               llama3/8B_full_single_device
                                               mistral/7B_full_low_memory
      full_finetune_distributed                llama2/7B_full
                                               llama2/13B_full
                                               llama3/8B_full
                                               mistral/7B_full
                                               gemma/2B_full
      lora_finetune_single_device              llama2/7B_lora_single_device
                                               llama2/7B_qlora_single_device
                                               llama3/8B_lora_single_device
                                               llama3/8B_qlora_single_device
                                               llama2/13B_qlora_single_device
                                               mistral/7B_lora_single_device

   The ``RECIPE`` column shows the easy-to-use and workable fine-tuning and inference recipes for popular fine-tuning
   techniques (such as LoRA). The ``CONFIG`` column lists the YAML configurations for easily configuring training,
   evaluation, quantization, or inference recipes.

   The snippet shows the architecture of a model's YAML configuration file:

   .. code-block:: yaml

      # Model arguments
      model:
        _component_: torchtune.models.llama2.lora_llama2_7b
        lora_attn_modules: ['q_proj', 'v_proj']
        apply_lora_to_mlp: False
        apply_lora_to_output: False
        lora_rank: 8
        lora_alpha: 16
      
      tokenizer:
        _component_: torchtune.models.llama2.llama2_tokenizer
        path: /tmp/Llama-2-7b-hf/tokenizer.model
      
      # Dataset and sampler
      dataset:
        _component_: torchtune.datasets.alpaca_cleaned_dataset
        train_on_input: True

#. This configuration file defines the fine-tuning base model path, data set, hyper-parameters for optimizer and scheduler,
   and training data type. To download the base model for fine-tuning, run the following command:

   .. code-block:: shell

      tune download meta-llama/Llama-2-7b-hf --output-dir /tmp/Llama-2-7b-hf --hf-token

   The output directory argument for ``--output-dir`` should map the model path specified in YAML config file.

#. To launch ``lora_finetune_distributed`` on four devices, run the following
   command:

   .. code-block:: shell

      tune run --nnodes 1 --nproc_per_node 4 lora_finetune_distributed --config llama2/7B_lora

   If successful, you should something like the following output:

   .. code-block:: shell

      INFO:torchtune.utils.logging:FSDP is enabled. Instantiating Model on CPU for Rank 0 ...
      INFO:torchtune.utils.logging:Model instantiation took 7.32 secs
      INFO:torchtune.utils.logging:Memory Stats after model init:
      {'peak_memory_active': 9.478172672, 'peak_memory_alloc': 8.953868288, 'peak_memory_reserved': 11.112808448}
      INFO:torchtune.utils.logging:Optimizer and loss are initialized.
      INFO:torchtune.utils.logging:Dataset and Sampler are initialized.
      INFO:torchtune.utils.logging:Learning rate scheduler is initialized.
      1|111|Loss: 1.5790324211120605:   7%|█                                          | 114/1618

Read more about inference frameworks in :doc:`LLM inference frameworks <../inference/llm-inference-frameworks>`.
