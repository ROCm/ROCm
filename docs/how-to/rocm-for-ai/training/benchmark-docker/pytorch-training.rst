:orphan:

.. meta::
   :description: How to train a model using ROCm PyTorch
   :keywords: ROCm, AI, LLM, train, Megatron-LM, megatron, Llama, tutorial, docker, torch

**********************************
Training a model with ROCm PyTorch
**********************************

PyTorch is an open-source machine learning framework that is widely used for
model training with GPU-optimized components for transformer-based models.

The ROCm PyTorch Training Docker (``rocm/pytorch-training:v25.3``) image
provides a prebuilt optimized environment for fine-tuning and pretraining a
model on AMD Instinct MI325X and MI300X accelerators. It includes the following
software components to accelerate training workloads:

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

.. _amd-pytorch-training-model-support:

Supported models
================

The following models are pre-optimized for performance on the AMD Instinct MI300X accelerator.
Only the models listed here are supported in the following workflows.

* Llama 3.1 8B

* Llama 3.1 70B

* FLUX.1-dev

.. note::

   Some models, such as Llama 3, require an external license agreement through
   a third party (for instance, Meta).

System validation
=================

If you have already validated your system settings, skip this step. Otherwise,
complete the :ref:`system validation and optimization steps <train-a-model-system-validation>`
to set up your system before starting training.

Environment setup
=================

This Docker image is optimized for specific model configurations outlined
below. Performance may vary for other training workloads, as configurations and
run conditions outside of those described are not validated by AMD.

Download the Docker image
-------------------------

1. Use the following command to pull the Docker image from Docker Hub.

   .. code-block:: shell

      docker pull rocm/pytorch-training:v25.3

2. Run the Docker container.

   .. code-block:: shell

      docker run -it --device /dev/dri --device /dev/kfd --network host --ipc host --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged -v $HOME:$HOME -v  $HOME/.ssh:/root/.ssh --shm-size 64G --name training_env rocm/pytorch-training:v25.3

3. Use these commands if you exit the ``training_env`` container and need to return to it.

   .. code-block:: shell

      docker start training_env
      docker exec -it training_env bash

4. In the Docker container, clone the `<https://github.com/ROCm/MAD>`__ repository and navigate to the benchmark scripts directory.

   .. code-block:: shell

      git clone https://github.com/ROCm/MAD
      cd MAD/scripts/pytorch-train

Prepare training datasets and dependencies
------------------------------------------

The following benchmarking examples may require downloading models and datasets
from Hugging Face. To ensure successful access to gated repos, set your
``HF_TOKEN``.

Run the setup script to install libraries and datasets needed for benchmarking.

.. code-block:: shell

   ./pytorch_benchmark_setup.sh

``pytorch_benchmark_setup.sh`` will be install the following libraries:

.. list-table::
   :header-rows: 1

   * - Library
     - Benchmark model
     - Reference

   * - ``accelerate``
     - Llama 3.1 8B, FLUX
     - `Hugging Face Accelerate <https://huggingface.co/docs/accelerate/en/index>`_

   * - ``datasets``
     - Llama 3.1 8B, 70B, FLUX
     - `Hugging Face Datasets <https://huggingface.co/docs/datasets/v3.2.0/en/index>`_ 3.2.0

   * - ``torchdata``
     - Llama 3.1 70B
     - `TorchData <https://pytorch.org/data/beta/index.html>`_

   * - ``tomli``
     - Llama 3.1 70B
     - `Tomli <https://pypi.org/project/tomli/>`_

   * - ``tiktoken``
     - Llama 3.1 70B
     - `tiktoken <https://github.com/openai/tiktoken>`_

   * - ``blobfile``
     - Llama 3.1 70B
     - `blobfile <https://pypi.org/project/blobfile/>`_

   * - ``tabulate``
     - Llama 3.1 70B
     - `tabulate <https://pypi.org/project/tabulate/>`_

   * - ``wandb``
     - Llama 3.1 70B
     - `Weights & Biases <https://github.com/wandb/wandb>`_

   * - ``sentencepiece``
     - Llama 3.1 70B, FLUX
     - `SentencePiece <https://github.com/google/sentencepiece>`_ 0.2.0

   * - ``tensorboard``
     - Llama 3.1 70 B, FLUX
     - `TensorBoard <https://www.tensorflow.org/tensorboard>`_ 2.18.0

   * - ``csvkit``
     - FLUX
     - `csvkit <https://csvkit.readthedocs.io/en/latest/>`_ 2.0.1

   * - ``deepspeed``
     - FLUX
     - `DeepSpeed <https://github.com/deepspeedai/DeepSpeed>`_ 0.16.2

   * - ``diffusers``
     - FLUX
     - `Hugging Face Diffusers <https://huggingface.co/docs/diffusers/en/index>`_ 0.31.0

   * - ``GitPython``
     - FLUX
     - `GitPython <https://github.com/gitpython-developers/GitPython>`_ 3.1.44

   * - ``opencv-python-headless``
     - FLUX
     - `opencv-python-headless <https://pypi.org/project/opencv-python-headless/>`_ 4.10.0.84

   * - ``peft``
     - FLUX
     - `PEFT <https://huggingface.co/docs/peft/en/index>`_ 0.14.0

   * - ``protobuf``
     - FLUX
     - `Protocol Buffers <https://github.com/protocolbuffers/protobuf>`_ 5.29.2

   * - ``pytest``
     - FLUX
     - `PyTest <https://docs.pytest.org/en/stable/>`_ 8.3.4

   * - ``python-dotenv``
     - FLUX
     - `python-dotenv <https://pypi.org/project/python-dotenv/>`_ 1.0.1

   * - ``seaborn``
     - FLUX
     - `Seaborn <https://seaborn.pydata.org/>`_ 0.13.2

   * - ``transformers``
     - FLUX
     - `Transformers <https://huggingface.co/docs/transformers/en/index>`_ 4.47.0

``pytorch_benchmark_setup.sh`` will download the following models from Hugging Face:

* `meta-llama/Llama-3.1-70B-Instruct <https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct>`_

* `black-forest-labs/FLUX.1-dev <https://huggingface.co/black-forest-labs/FLUX.1-dev>`_

Along with the following datasets:

* `WikiText <https://huggingface.co/datasets/Salesforce/wikitext>`_

* `bghira/pseudo-camera-10k <https://huggingface.co/datasets/bghira/pseudo-camera-10k>`_

Start training
==============

Once your environment is set up, use the following commands and examples to start benchmarking pretraining and
fine-tuning performance.

Pretraining
-----------

To start the pretraining benchmark, use the following command with the
appropriate options. See the following list of options and their descriptions.

.. code-block:: shell

   ./pytorch_benchmark_report.sh -t $training_mode -m $model_repo -p $datatype -s $sequence_length

Options and available models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Name
     - Options
     - Description

   * - ``$training_mode``
     - ``pretrain``
     - Benchmark pretraining

   * -
     - ``finetune_fw``
     - Benchmark full weight fine-tuning (Llama 3.1 70B with BF16)

   * -
     - ``finetune_lora``
     - Benchmark LoRA fine-tuning (Llama 3.1 70B with BF16)

   * - ``$datatype``
     - FP8 or BF16
     - Only Llama 3.1 8B supports FP8 precision.

   * - ``$model_repo``
     - Llama-3.1-8B
     - `Llama 3.1 8B <https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct>`_

   * - 
     - Llama-3.1-70B
     - `Llama 3.1 70B <https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct>`_

   * - 
     - Flux
     - `FLUX.1 [dev] <https://huggingface.co/black-forest-labs/FLUX.1-dev>`_

Fine-tuning
-----------

To start the fine-tuning benchmark, use the following command. It will run the benchmarking example of Llama 2 70B
with the WikiText dataset using the AMD fork of `torchtune <https://github.com/AMD-AIG-AIMA/torchtune>`_.

.. code-block:: shell

   ./pytorch_benchmark_report.sh -t {finetune_fw, finetune_lora} -p BF16 -m Llama-3.1-70B

Benchmarking examples
---------------------

Here are some examples of how to use the command.

* Example 1: Llama 3.1 70B with BF16 precision with `torchtitan <https://github.com/ROCm/torchtitan>`_.

  .. code-block:: shell

     ./pytorch_benchmark_report.sh -t pretrain -p BF16 -m Llama-3.1-70B -s 8192

* Example 2: Llama 3.1 8B with FP8 precision using Transformer Engine (TE) and Hugging Face Accelerator.

  .. code-block:: shell

     ./pytorch_benchmark_report.sh -t pretrain -p FP8 -m Llama-3.1-70B -s 8192

* Example 3: FLUX.1-dev with BF16 precision with FluxBenchmark.

  .. code-block:: shell

     ./pytorch_benchmark_report.sh -t pretrain -p BF16 -m Flux

* Example 4: Torchtune full weight fine-tuning with Llama 3.1 70B

  .. code-block:: shell

     ./pytorch_benchmark_report.sh -t finetune_fw -p BF16 -m Llama-3.1-70B

* Example 5: Torchtune LoRA fine-tuning with Llama 3.1 70B

  .. code-block:: shell

     ./pytorch_benchmark_report.sh -t finetune_lora -p BF16 -m Llama-3.1-70B
