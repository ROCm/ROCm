.. meta::
   :description: How to use ROCm for AI inference workloads.
   :keywords: ROCm, AI, machine learning, LLM, AI inference, NLP, GPUs, usage, tutorial

****************************
Use ROCm for AI inference
****************************
AI inference is a process of deploying a trained machine learning model to make predictions or classifications on new data. This commonly involves using the model with real-time data and making quick decisions based on the predictions made by the model.  

Understanding the ROCm™ software platform’s architecture and capabilities is vital for running AI inference. By leveraging the ROCm platform's capabilities, you can harness the power of high-performance computing and efficient resource management to run inference workloads, leading to faster predictions and classifications on real-time data.

Throughout the following topics, this section provides a comprehensive guide to setting up and deploying AI inference on AMD GPUs. This includes instructions on how to install ROCm, how to use Hugging Face Transformers to manage pre-trained models for natural language processing (NLP) tasks, how to validate vLLM on AMD Instinct™ MI300X accelerators and illustrate how to deploy trained models in production environments. 

For AI inference tutorials in Jupyter Notebook format, see `Tutorials for AI developers <https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/>`_.

- :doc:`Installing ROCm and machine learning frameworks <install>`

- :doc:`Running models from Hugging Face <hugging-face-models>`

- :doc:`LLM inference frameworks <llm-inference-frameworks>`

- :doc:`Performance validation <vllm-benchmark>`

- :doc:`Deploying your model <deploy-your-model>`
