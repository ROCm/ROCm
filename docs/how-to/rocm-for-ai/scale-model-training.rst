.. meta::
   :description: How to scale and accelerate model training
   :keywords: ROCm, AI, LLM, train, fine-tune, deploy, FSDP, DeepSpeed, LLaMA, tutorial

**********************
Scaling model training
**********************

To train a large-scale model like OpenAI GPT-2 or Meta Llama 2 70B, a single accelerator or GPU cannot store all the
model parameters required for training. This immense scale presents a fundamental challenge: no single GPU or
accelerator can simultaneously store and process the entire model's parameters during training. PyTorch
provides an answer to this computational constraint through its distributed training frameworks.

.. _rocm-for-ai-pytorch-distributed:

PyTorch distributed
===================

Features in ``torch.distributed`` are categorized into three main components:

- `Distributed data-parallel training
  <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`_ (DDP)

- `RPC-Based distributed training <https://pytorch.org/docs/stable/rpc.html>`_ (RPC)

- `Collective communication <https://pytorch.org/docs/stable/distributed.html>`_

In this topic, the focus is on the distributed data-parallelism strategy as it’s the most popular. To get started with DDP,
you need to first understand how to coordinate the model and its training data across multiple accelerators or GPUs.

The DDP workflow on multiple accelerators or GPUs is as follows:

#. Split the current global training batch into small local batches on each GPU. For instance, if you have 8 GPUs and
   the global batch is set at 32 samples, each of the 8 GPUs will have a local batch size of 4 samples.

#. Copy the model to every device so each can process its local batches independently.

#. Run a forward pass, then a backward pass, and output the gradient of the weights with respect to the loss of the
   model for that local batch. This happens in parallel on multiple devices.

#. Synchronize the local gradients computed by each device and combine them to update the model weights. The updated
   weights are then redistributed to each device.

In DDP training, each process or worker owns a replica of the model and processes a batch of data, and then the reducer uses
``allreduce`` to sum up gradients over different workers.

See the following developer blogs for more in-depth explanations and examples.

*  `Multi GPU training with DDP — PyTorch Tutorials <https://pytorch.org/tutorials/beginner/ddp_series_multigpu.html>`_

*  `Building a decoder transformer model on AMD GPUs — ROCm Blogs
   <https://rocm.blogs.amd.com/artificial-intelligence/decoder-transformer/README.html#distributed-training-on-multiple-gpus>`_

.. _rocm-for-ai-pytorch-fsdp:

PyTorch FSDP
------------

As noted in :ref:`PyTorch distributed <rocm-for-ai-pytorch-distributed>`, DDP model weights and optimizer states
are evenly replicated across all workers. Fully Sharded Data Parallel (FSDP) is a type of data parallelism that shards
model parameters, optimizer states, and gradients across DDP ranks.

When training with FSDP, the GPU memory footprint is smaller than when training with DDP across all workers. This makes
training some very large models feasible by allowing larger models or batch sizes to fit on-device. However, this
comes with the cost of increased communication volume. The communication overhead is reduced by internal optimizations
like overlapping communication and computation.

For a high-level overview of how FSDP works, review `Getting started with Fully Sharded Data Parallel
<https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html#how-fsdp-works>`_.

For detailed training steps, see `PyTorch FSDP examples
<https://github.com/pytorch/examples/tree/main/distributed/FSDP>`_.

.. _rocm-for-ai-deepspeed:

DeepSpeed
---------

`DeepSpeed <https://deepspeed.ai>`_ offers system innovations that make large-scale deep learning training effective,
efficient, and easy to use. Innovations such as ZeRO, 3D-Parallelism, DeepSpeed-MoE, ZeRO-Infinity, and so on fall under
the training pillar.

See `Pre-training a large language model with Megatron-DeepSpeed on multiple AMD GPUs
<https://rocm.blogs.amd.com/artificial-intelligence/megatron-deepspeed-pretrain/README.html>`_ for a detailed example of
training with DeepSpeed on an AMD accelerator or GPU.

.. _rocm-for-ai-automatic-mixed-precision:

Automatic mixed precision (AMP)
-------------------------------

As models increase in size, so do the time and memory needed to train them; their cost also increases. Any measure we
can take to reduce training time and memory usage through `automatic mixed precision
<https://pytorch.org/docs/stable/amp.html>`_ (AMP) is highly beneficial for most use cases.

See `Automatic mixed precision in PyTorch using AMD GPUs — ROCm Blogs
<https://rocm.blogs.amd.com/artificial-intelligence/automatic-mixed-precision/README.html#automatic-mixed-precision-in-pytorch-using-amd-gpus>`_
for more information about running AMP on an AMD accelerator.

.. _rocm-for-ai-fine-tune:

Fine-tuning your model
======================

ROCm supports multiple techniques for :ref:`optimizing fine-tuning <fine-tuning-llms-concept-optimizations>`, for
example, LoRA, QLoRA, PEFT, and FSDP.

Learn more about challenges and solutions for model fine-tuning in :doc:`../llm-fine-tuning-optimization/index`.

The following developer blogs showcase examples of fine-tuning a model on an AMD accelerator or GPU.

* Fine-tuning Llama2 with LoRA

  * `Fine-tune Llama 2 with LoRA: Customizing a large language model for question-answering
    <https://rocm.blogs.amd.com/artificial-intelligence/llama2-lora/README.html>`_

* Fine-tuning Llama2 with QLoRA

  * `Enhancing LLM accessibility: A deep dive into QLoRA through fine-tuning Llama 2 on a single AMD GPU
    <https://rocm.blogs.amd.com/artificial-intelligence/llama2-Qlora/README.html>`_

* Fine-tuning a BERT-based LLM for a text classification task using JAX

  * `LLM distributed supervised fine-tuning with JAX
    <https://rocm.blogs.amd.com/artificial-intelligence/distributed-sft-jax/README.html>`_

* Fine-tuning StarCoder using PEFT

  * `Instruction fine-tuning of StarCoder with PEFT on multiple AMD GPUs
    <https://rocm.blogs.amd.com/artificial-intelligence/starcoder-fine-tune/README.html>`_

* Recipes for fine-tuning Llama2 and 3 with ``llama-recipes``

  * `meta-llama/llama-recipes: Scripts for fine-tuning Meta Llama3 with composable FSDP & PEFT methods to cover
    single/multi-node GPUs <https://github.com/meta-llama/llama-recipes/tree/main/recipes/quickstart/finetuning>`_
