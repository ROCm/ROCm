.. meta::
   :description: How to fine-tune models with ROCm
   :keywords: ROCm, LLM, fine-tuning, inference, usage, tutorial, deep learning, PyTorch, TensorFlow, JAX

*************************
Fine-tuning and inference
*************************

Fine-tuning using ROCm involves leveraging AMD's GPU-accelerated :doc:`libraries <rocm:reference/api-libraries>` and
:doc:`tools <rocm:reference/rocm-tools>` to optimize and train deep learning models. ROCm provides a comprehensive
ecosystem for deep learning development, including open-source libraries for optimized deep learning operations and
ROCm-aware versions of :doc:`deep learning frameworks <../../deep-learning-rocm>` such as PyTorch, TensorFlow, and JAX.

Single-accelerator systems, such as a machine equipped with a single accelerator or GPU, are commonly used for
smaller-scale deep learning tasks, including fine-tuning pre-trained models and running inference on moderately
sized datasets. See :doc:`single-gpu-fine-tuning-and-inference`.

Multi-accelerator systems, on the other hand, consist of multiple accelerators working in parallel. These systems are
typically used in LLMs and other large-scale deep learning tasks where performance, scalability, and the handling of
massive datasets are crucial. See :doc:`multi-gpu-fine-tuning-and-inference`.
