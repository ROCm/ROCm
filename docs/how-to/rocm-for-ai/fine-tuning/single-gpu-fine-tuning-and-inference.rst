.. meta::
   :description: Model fine-tuning and inference on a single-GPU system
   :keywords: ROCm, LLM, fine-tuning, usage, tutorial, single-GPU, LoRA, PEFT, inference, SFTTrainer

****************************************************
Fine-tuning and inference using a single accelerator
****************************************************

This section explains model fine-tuning and inference techniques on a single-accelerator system. See
:doc:`Multi-accelerator fine-tuning <multi-gpu-fine-tuning-and-inference>` for a setup with multiple accelerators or
GPUs.

.. _fine-tuning-llms-single-gpu-env:

Environment setup
=================

This section was tested using the following hardware and software environment.

.. list-table::
   :stub-columns: 1

   * - Hardware
     - AMD Instinct MI300X accelerator

   * - Software
     - ROCm 6.1, Ubuntu 22.04, PyTorch 2.1.2, Python 3.10

   * - Libraries
     - ``transformers`` ``datasets`` ``huggingface-hub`` ``peft`` ``trl`` ``scipy``

   * - Base model
     - ``meta-llama/Llama-2-7b-chat-hf``

.. _fine-tuning-llms-single-gpu-env-setup:

Setting up the base implementation environment
----------------------------------------------

#. Install PyTorch for ROCm. Refer to the
   :doc:`PyTorch installation guide <rocm-install-on-linux:install/3rd-party/pytorch-install>`. For a consistent
   installation, it’s recommended to use official ROCm prebuilt Docker images with the framework pre-installed.

#. In the Docker container, check the availability of ROCm-capable accelerators using the following command.

   .. code-block:: shell

      rocm-smi --showproductname

   Your output should look like this:

   .. code-block:: shell

      ============================ ROCm System Management Interface ============================
      ====================================== Product Info ======================================
      GPU[0]          : Card series:          AMD Instinct MI300X OAM
      GPU[0]          : Card model:           0x74a1
      GPU[0]          : Card vendor:          Advanced Micro Devices, Inc. [AMD/ATI]
      GPU[0]          : Card SKU:             MI3SRIOV
      ==========================================================================================
      ================================== End of ROCm SMI Log ===================================

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

#. Install the required dependencies.

   bitsandbytes is a library that facilitates quantization to improve the efficiency of deep learning models. Learn more
   about its use in :doc:`../inference-optimization/model-quantization`.

   See the :ref:`Optimizations for model fine-tuning <fine-tuning-llms-concept-optimizations>` for a brief discussion on
   PEFT and TRL.

   .. code-block:: shell

      # Install `bitsandbytes` for ROCm 6.0+.
      # Use -DBNB_ROCM_ARCH to target a specific GPU architecture.
      git clone --recurse https://github.com/ROCm/bitsandbytes.git
      cd bitsandbytes
      git checkout rocm_enabled_multi_backend
      pip install -r requirements-dev.txt
      cmake -DBNB_ROCM_ARCH="gfx942" -DCOMPUTE_BACKEND=hip -S .
      python setup.py install
      
      # To leverage the SFTTrainer in TRL for model fine-tuning.
      pip install trl
      
      # To leverage PEFT for efficiently adapting pre-trained language models .
      pip install peft
      
      # Install the other dependencies.
      pip install transformers datasets huggingface-hub scipy

#. Check that the required packages can be imported.

   .. code-block:: python

      import torch
      from datasets import load_dataset
      from transformers import (
          AutoModelForCausalLM,
          AutoTokenizer,
          TrainingArguments
      )
      from peft import LoraConfig
      from trl import SFTTrainer

.. _fine-tuning-llms-single-gpu-download-model-dataset:

Download the base model and fine-tuning dataset
-----------------------------------------------

#. Request to access to download the `Meta's official Llama model <https://huggingface.co/meta-llama>`_ from Hugging
   Face. After permission is granted, log in with the following command using your personal access tokens:

   .. code-block:: shell

      huggingface-cli login

   .. note::

      You can also use the `NousResearch Llama-2-7b-chat-hf <https://huggingface.co/NousResearch/Llama-2-7b-chat-hf>`_ 
      as a substitute. It has the same model weights as the original.

#. Run the following code to load the base model and tokenizer.

   .. code-block:: python

      # Base model and tokenizer names.
      base_model_name = "meta-llama/Llama-2-7b-chat-hf"
      
      # Load base model to GPU memory.
      device = "cuda:0"
      base_model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code = True).to(device)
      
      # Load tokenizer.
      tokenizer = AutoTokenizer.from_pretrained(
              base_model_name, 
              trust_remote_code = True)
      tokenizer.pad_token = tokenizer.eos_token
      tokenizer.padding_side = "right"

#. Now, let's fine-tune the base model for a question-and-answer task using a small dataset called
   `mlabonne/guanaco-llama2-1k <https://huggingface.co/datasets/mlabonne/guanaco-llama2-1k>`_, which is a 1000 sample
   subset of the `timdettmers/openassistant-guanaco <https://huggingface.co/datasets/OpenAssistant/oasst1>`_ dataset.

   .. code-block::

      # Dataset for fine-tuning.
      training_dataset_name = "mlabonne/guanaco-llama2-1k"
      training_dataset = load_dataset(training_dataset_name, split = "train")
      
      # Check the data.
      print(training_dataset)
      
      # Dataset 11 is a QA sample in English.
      print(training_dataset[11])

#. With the base model and the dataset, let's start fine-tuning!

.. _fine-tuning-llms-single-gpu-configure-params:

Configure fine-tuning parameters
--------------------------------

To set up ``SFTTrainer`` parameters, you can use the following code as reference.

.. code-block:: python

   # Training parameters for SFTTrainer.
   training_arguments = TrainingArguments(
       output_dir = "./results",
            num_train_epochs = 1,
            per_device_train_batch_size = 4,
            gradient_accumulation_steps = 1,
            optim = "paged_adamw_32bit",
            save_steps = 50,
            logging_steps = 50,
            learning_rate = 4e-5,
            weight_decay = 0.001,
            fp16=False,
            bf16=False,
            max_grad_norm = 0.3,
            max_steps = -1,
            warmup_ratio = 0.03,
            group_by_length = True,
            lr_scheduler_type = "constant",
            report_to = "tensorboard"
   )

.. _fine-tuning-llms-single-gpu-start:

Fine-tuning
===========

In this section, you'll see two ways of training: with the LoRA technique and without. See :ref:`Optimizations for model
fine-tuning <fine-tuning-llms-concept-optimizations>` for an introduction to LoRA. Training with LoRA uses the
``SFTTrainer`` API with its PEFT integration. Training without LoRA forgoes these benefits.

Compare the number of trainable parameters and training time under the two different methodologies.

.. tab-set::

   .. tab-item:: Fine-tuning with LoRA and PEFT
      :sync: with

      1. Configure LoRA using the following code snippet.

         .. code-block:: python

            peft_config = LoraConfig(
                    lora_alpha = 16,
                    lora_dropout = 0.1,
                    r = 64,
                    bias = "none",
                    task_type = "CAUSAL_LM"
            )
            # View the number of trainable parameters.
            from peft import get_peft_model
            peft_model = get_peft_model(base_model, peft_config)
            peft_model.print_trainable_parameters()

         The output should look like this. Compare the number of trainable parameters to that when fine-tuning without
         LoRA and PEFT.

         .. code-block:: shell

            trainable params: 33,554,432 || all params: 6,771,970,048 || trainable%: 0.49548996469513035

      2. Initialize ``SFTTrainer`` with a PEFT LoRA configuration and run the trainer.

         .. code-block:: python

            # Initialize an SFT trainer.
            sft_trainer = SFTTrainer(
                    model = base_model,
                    train_dataset = training_dataset,
                    peft_config = peft_config,
                    dataset_text_field = "text",
                    tokenizer = tokenizer,
                    args = training_arguments
            ) 
            
            # Run the trainer.
            sft_trainer.train()

         The output should look like this:

         .. code-block:: shell

            {'loss': 1.5973, 'grad_norm': 0.25271978974342346, 'learning_rate': 4e-05, 'epoch': 0.16}
            {'loss': 2.0519, 'grad_norm': 0.21817368268966675, 'learning_rate': 4e-05, 'epoch': 0.32}
            {'loss': 1.6147, 'grad_norm': 0.3046981394290924, 'learning_rate': 4e-05, 'epoch': 0.48}
            {'loss': 1.4124, 'grad_norm': 0.11534837633371353, 'learning_rate': 4e-05, 'epoch': 0.64}
            {'loss': 1.5627, 'grad_norm': 0.09108350425958633, 'learning_rate': 4e-05, 'epoch': 0.8}
            {'loss': 1.417, 'grad_norm': 0.2536439299583435, 'learning_rate': 4e-05, 'epoch': 0.96}
            {'train_runtime': 197.4947, 'train_samples_per_second': 5.063, 'train_steps_per_second': 0.633, 'train_loss': 1.6194254455566406, 'epoch': 1.0}
            100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 125/125 [03:17<00:00,  1.58s/it]

   .. tab-item:: Fine-tuning without LoRA and PEFT
      :sync: without

      1. Use the following code to get started.

         .. code-block:: python

            def print_trainable_parameters(model):
                # Prints the number of trainable parameters in the model.
                trainable_params = 0
                all_param = 0
                for _, param in model.named_parameters():
                    all_param += param.numel()
                    if param.requires_grad:
                        trainable_params += param.numel()
                print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}")
            
            sft_trainer.peft_config = None
            print_trainable_parameters(sft_trainer.model)

         The output should look like this. Compare the number of trainable parameters to that when fine-tuning with LoRA
         and PEFT.

         .. code-block:: shell

            trainable params: 6,738,415,616 || all params: 6,738,415,616 || trainable%: 100.00


      2. Run the trainer.

         .. code-block:: python

            # Trainer without LoRA config.
            trainer_full = SFTTrainer(
                    model = base_model,
                    train_dataset = training_dataset,
                    dataset_text_field = "text",
                    tokenizer = tokenizer,
                    args = training_arguments
            ) 
            
            # Training.
            trainer_full.train()

         The output should look like this:

         .. code-block:: shell

            {'loss': 1.5975, 'grad_norm': 0.25113457441329956, 'learning_rate': 4e-05, 'epoch': 0.16}
            {'loss': 2.0524, 'grad_norm': 0.2180655151605606, 'learning_rate': 4e-05, 'epoch': 0.32}
            {'loss': 1.6145, 'grad_norm': 0.2949850261211395, 'learning_rate': 4e-05, 'epoch': 0.48}
            {'loss': 1.4118, 'grad_norm': 0.11036080121994019, 'learning_rate': 4e-05, 'epoch': 0.64}
            {'loss': 1.5595, 'grad_norm': 0.08962831646203995, 'learning_rate': 4e-05, 'epoch': 0.8}
            {'loss': 1.4119, 'grad_norm': 0.25422757863998413, 'learning_rate': 4e-05, 'epoch': 0.96}
            {'train_runtime': 419.5154, 'train_samples_per_second': 2.384, 'train_steps_per_second': 0.298, 'train_loss': 1.6171623611450194, 'epoch': 1.0}
            100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 125/125 [06:59<00:00,  3.36s/it]

.. _fine-tuning-llms-single-gpu-saving:

Saving adapters or fully fine-tuned models
------------------------------------------

PEFT methods freeze the pre-trained model parameters during fine-tuning and add a smaller number of trainable
parameters, namely the adapters, on top of it. The adapters are trained to learn specific task information. The adapters
trained with PEFT are usually an order of magnitude smaller than the full base model, making them convenient to share,
store, and load.

.. tab-set::

   .. tab-item:: Saving a PEFT adapter
      :sync: with

      If you're using LoRA and PEFT, use the following code to save a PEFT adapter to your system once the fine-tuning
      is completed.

      .. code-block:: python

         # PEFT adapter name.
         adapter_name = "llama-2-7b-enhanced-adapter"
         
         # Save PEFT adapter.
         sft_trainer.model.save_pretrained(adapter_name)

      The saved PEFT adapter should look like this on your system:

      .. code-block:: shell

         # Access adapter directory.
         cd llama-2-7b-enhanced-adapter
         
         # List all adapter files.
         README.md  adapter_config.json  adapter_model.safetensors

   .. tab-item:: Saving a fully fine-tuned model
      :sync: without

      If you're not using LoRA and PEFT so there is no PEFT LoRA configuration used for training, use the following code 
      to save your fine-tuned model to your system.

      .. code-block:: python

         # Fully fine-tuned model name.
         new_model_name = "llama-2-7b-enhanced"
         
         # Save the fully fine-tuned model.
         full_trainer.model.save_pretrained(new_model_name)

      The saved new full model should look like this on your system:

      .. code-block:: shell

         # Access new model directory.
         cd llama-2-7b-enhanced
         
         # List all model files.
         config.json                       model-00002-of-00006.safetensors  model-00005-of-00006.safetensors
         generation_config.json            model-00003-of-00006.safetensors  model-00006-of-00006.safetensors
         model-00001-of-00006.safetensors  model-00004-of-00006.safetensors  model.safetensors.index.json

.. note::

   PEFT adapters can’t be loaded by ``AutoModelForCausalLM`` from the Transformers library as they do not contain
   full model parameters and model configurations, for example, ``config.json``. To use it as a normal transformer
   model, you need to merge them into the base model.

Basic model inference
=====================

A trained model can be classified into one of three types:

*  A PEFT adapter

*  A pre-trained language model in Hugging Face

*  A fully fine-tuned model not using PEFT

Let's look at achieving model inference using these types of models.

.. tab-set::

   .. tab-item:: Inference using PEFT adapters

      To use PEFT adapters like a normal transformer model, you can run the generation by loading a base model along with PEFT 
      adapters as follows.

      .. code-block:: python

         from peft import PeftModel
         from transformers import AutoModelForCausalLM
         
         # Set the path of the model or the name on Hugging face hub
         base_model_name = "meta-llama/Llama-2-7b-chat-hf"
         
         # Set the path of the adapter
         adapter_name = "Llama-2-7b-enhanced-adpater"
         
         # Load base model 
         base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
         
         # Adapt the base model with the adapter 
         new_model = PeftModel.from_pretrained(base_model, adapter_name)
         
         # Then, run generation as the same with a normal model outlined in 2.1

      The PEFT library provides a ``merge_and_unload`` method, which merges the adapter layers into the base model. This is
      needed if someone wants to save the adapted model into local storage and use it as a normal standalone model.

      .. code-block:: python

         # Load base model 
         base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
         
         # Adapt the base model with the adapter 
         new_model = PeftModel.from_pretrained(base_model, adapter_name)
         
         # Merge adapter 
         model = model.merge_and_unload()

         # Save the merged model into local
         model.save_pretrained("merged_adpaters")

   .. tab-item:: Inference using pre-trained or fully fine-tuned models

      If you have a fully fine-tuned model not using PEFT, you can load it like any other pre-trained language model in
      `Hugging Face Hub <https://huggingface.co/docs/hub/en/index>`_ using the `Transformers
      <https://huggingface.co/docs/transformers/en/index>`_ library.

      .. code-block:: python

         # Import relevant class for loading model and tokenizer
         from transformers import AutoTokenizer, AutoModelForCausalLM
         
         # Set the pre-trained model name on Hugging face hub
         model_name = "meta-llama/Llama-2-7b-chat-hf"
         
         # Set device type 
         device = "cuda:0"
         
         # Load model and tokenizer 
         model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
         tokenizer = AutoTokenizer.from_pretrained(model_name)
         
         # Input prompt encoding 
         query = "What is a large language model?"
         inputs = tokenizer.encode(query, return_tensors="pt").to(device)
         
         # Token generation  
         outputs = model.generate(inputs) 
         
         # Outputs decoding 
         print(tokenizer.decode(outputs[0]))

      In addition, pipelines from Transformers offer simple APIs to use pre-trained models for different tasks, including
      sentiment analysis, feature extraction, question answering and so on. You can use the pipeline abstraction to achieve
      model inference easily.

      .. code-block:: python

         # Import relevant class for loading model and tokenizer
         from transformers import pipeline
         
         # Set the path of your model or the name on Hugging face hub
         model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"
         
         # Set pipeline 
         # A positive device value will run the model on associated CUDA device id
         pipe = pipeline("text-generation", model=model_name_or_path, device=0)
         
         # Token generation
         print(pipe("What is a large language model?")[0]["generated_text"])

If using multiple accelerators, see
:ref:`Multi-accelerator fine-tuning and inference <fine-tuning-llms-multi-gpu-hugging-face-accelerate>` to explore
popular libraries that simplify fine-tuning and inference in a multi-accelerator system.

Read more about inference frameworks like vLLM and Hugging Face TGI in
:doc:`LLM inference frameworks <../inference/llm-inference-frameworks>`.
